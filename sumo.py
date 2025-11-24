import torch,sys
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import gymnasium_sudoku
from dataclasses import dataclass
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from collections import deque
from torch.distributions.kl import kl_divergence as kl
from itertools import chain
import warnings
warnings.filterwarnings("ignore")


@dataclass(frozen=False)
class Hypers:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 10
    max_steps = int(1e6)
    batchsize = 512
    minibatch = 16
    e_aux = 6
    lr = 5e-4
    gamma = .99
    lambda_ = .99
    epsilon = .2
    beta = 1e-1 # entropy coeff
    beta_clone = 1 # kl coeff in the aux phase
    optim_steps = 10 # defualt 32 as seen in the original paper
    
hypers = Hypers()

def env():
    def fn():
        x = gym.make("sudoku-v0")
        return x 
    return AsyncVectorEnv([fn for _ in range(hypers.num_envs)])

def process_obs(x): # -> one hot encoding + mask
    x = torch.tensor(x,dtype=torch.int64,device=hypers.device)
    m = (x == 0).unsqueeze(1).to(torch.float32)
    x = F.one_hot(x,num_classes=10).permute(0,-1,1,2).float() 
    return torch.cat([x,m],dim=1) 

def softmax_mask(x): # action (x,y,z) -> x,y max = 8 and z min = 1 
    x = x.reshape(x.shape[0],3,9) 
    m = torch.zeros_like(x,dtype=torch.bool)  
    m[:,0,-1] = True
    m[:,1,-1] = True
    m[:,-1,0] = True
    value = -float("inf")
    return torch.masked_fill(x,m,value)
  
@torch.no_grad()
def w_init(l):
    if isinstance(l,(nn.Conv2d,nn.Linear)):
        nn.init.orthogonal_(l.weight)
        l.bias.fill_(0.0)

class p_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.LazyConv2d(64,1,1)
        self.c2 = nn.LazyConv2d(128,3,1)
        self.c3 = nn.LazyConv2d(128,3,1)
        self.l1 = nn.LazyLinear(512)
        self.l2 = nn.LazyLinear(256)
        self.l3 = nn.LazyLinear(3*9)
        self.v_aux = nn.LazyLinear(1)    # auxiliary value head
    
    def forward(self,x): 
        x = self.c1(x)
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x)) 
        x = F.relu(self.l1(x.flatten(start_dim=1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x)) 
        p_head = F.softmax(softmax_mask(x),dim=-1) # policy head output
        v_aux = self.v_aux(x)                      # auxiliary value head output
        return p_head,v_aux
       
class v_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.LazyConv2d(64,1,1)
        self.c2 = nn.LazyConv2d(64,3,1) 
        self.l1 = nn.LazyLinear(512) 
        self.l2 = nn.LazyLinear(128)
        self.v = nn.LazyLinear(1)  

    def forward(self,x):
        x = F.relu(self.c1(x)) 
        x = F.relu(self.c2(x)) # -> torch.Size([n env, 3136])
        x = F.relu(self.l1(x.flatten(start_dim=1)))
        x = F.relu(self.l2(x))
        return self.v(x) 


class memory: # Replay buffer class
    def __init__(self,env:AsyncVectorEnv,p_net,v_net):
        N = hypers.num_envs
        B = hypers.batchsize 
        self.state = torch.empty((B,N,9,9),device=hypers.device,dtype=torch.half)
        self.action = torch.empty((B,3,N),device=hypers.device,dtype=torch.float32)
        self.values = torch.empty((B,N,1),device=hypers.device,dtype=torch.float32)
        self.values_aux = torch.empty((B,N,1),device=hypers.device,dtype=torch.float32)
        self.v_target = torch.empty((B,N,1),device=hypers.device,dtype=torch.float32)
        self.prob = torch.empty((B,N,3),device=hypers.device,dtype=torch.float32) 
        self.rewards = torch.empty((B,N),device=hypers.device,dtype=torch.float32) 
        self.dones = torch.empty((B,N),device=hypers.device,dtype=torch.float32) 
        self.dist_prob = torch.empty((B,N,3,9),device=hypers.device,dtype=torch.float32) 
        self.advantages = torch.empty((B,N),device=hypers.device,dtype=torch.float32) 

        self.env = env
        self._observation = self.env.reset()[0]
        self.p_net = p_net
        self.v_net = v_net
        self.gamma = hypers.gamma   
        self._lambda_ = hypers.lambda_ 
        self.pointer = 0
        self.finished_reward = deque(maxlen=30)
        self.log_total_steps = deque(maxlen=30)
        self.episode_reward = torch.empty(self.env.num_envs).float()
        self.total_steps = torch.empty(hypers.num_envs).float()
          
    @torch.no_grad()
    def step(self,num_it):
        policy_output,v_policy = self.p_net(process_obs(self._observation))
        value = self.v_net(process_obs(self._observation))
        distribution = Categorical(policy_output)
        action = distribution.sample()
        prob = distribution.log_prob(action)
        
        assert torch.equal(action.T.T,action)
        action = action.T.cpu().numpy()
        state,reward,done,_,_ = self.env.step(action)
        
        for i in range(self.env.num_envs): # tracking episode rewards and total steps
            self.episode_reward[i] += reward[i] 
            self.total_steps[i] += 1
            if done[i]:
                self.finished_reward.append(self.episode_reward[i])
                self.log_total_steps.append(self.total_steps[i])
                self.episode_reward[i] = 0
                self.total_steps[i] = 0
     
        self.state[num_it].copy_(torch.as_tensor(self._observation)) 
        self.action[num_it].copy_(torch.as_tensor(action)) 
        self.values[num_it].copy_(value) 
        self.values_aux[num_it].copy_(v_policy)
        self.prob[num_it].copy_(prob)
        self.rewards[num_it].copy_(torch.as_tensor(reward))
        self.dones[num_it].copy_(torch.as_tensor(done)) 
        self.dist_prob[num_it].copy_(distribution.probs)
        self._observation = state        

    #@torch.compile(mode="reduce-overhead",fullgraph=True,)
    @torch.no_grad()
    def compute_advantage(self): 
        next_value = self.v_net(process_obs(self._observation)).unsqueeze(0)
        _values = torch.cat([self.values,next_value]).squeeze(-1)
        gae = torch.zeros_like(self.rewards[0], device=hypers.device)  
        td = self.rewards.clone().add_(self.gamma * _values[1:] * (1 - self.dones)).sub_(_values[:-1])
        for n in reversed(range(len(self.rewards))): 
            gae.mul_(self._lambda_ * self.gamma * (1-self.dones[n])).add_(td[n])
            self.advantages[n].copy_(gae)
    
    @torch.no_grad()
    def sample(self,minibatch): # with random sampling
        idx = torch.randperm(hypers.batchsize)[:hypers.minibatch]
        return (
            self.state[idx].flatten(0,1),
            self.action[idx],
            self.values[idx].flatten(0,1),
            self.values_aux[idx].flatten(0,1),
            self.v_target[idx].flatten(0,1),
            self.prob[idx],
            self.advantages[idx],
            self.dist_prob[idx].flatten(0,1)
        )

    def update_prob(self,x): # update (replace) the entire probability distribution
        x = x.reshape(*self.dist_prob.shape) 
        self.dist_prob = x

    def update_v_target(self,x): # update v target preallocated space 
        x = x.reshape(*self.v_target.shape) 
        self.v_target = x
 
    def traj_reward(self):
        return list(map(torch.tensor,(self.finished_reward,self.log_total_steps)))


class main:
    def init_nets(self):
        random = torch.randint(0,9,(self.env.reset()[0].shape))  
        self.p_net(process_obs(random))
        self.v_net(process_obs(random))
    
        self.p_net.apply(w_init)
        self.v_net.apply(w_init)

        self.p_net.compile()
        self.v_net.compile()

    def __init__(self):
        self.p_net = p_net().to(hypers.device)
        self.v_net = v_net().to(hypers.device)
        self.env = env() 
        self.init_nets()
        self.memory = memory(self.env,self.p_net,self.v_net)
        self.optim = Adam(
                chain(self.p_net.parameters(),self.v_net.parameters()),lr=hypers.lr
        )    
        self.writter = SummaryWriter("./")

    def save(self,n):
        data = {
            "policy state":self.p_net.state_dict(),
            "policy optim":self.p_optim.state_dict(),
            "value state":self.v_net.state_dict(),
            "value optim":self.v_optim.state_dict()
        }
        torch.save(data,f"./model-{n}")

    def process_sample(self): # sample and process parts of the sample
        states,actions,values,v_policy,v_target,probs,advantages,dist_prob = self.memory.sample(hypers.minibatch)
        actions = actions.transpose(1, 2).flatten(0,1)
        advantages = advantages.flatten().unsqueeze(-1)
        return (
            process_obs(states),
            actions,
            values,
            v_policy,
            v_target,
            probs,
            advantages,
            Categorical(probs=dist_prob)
        )

    def run(self,start=False):
        if start:
            for n in range(hypers.max_steps):
                for m in range(hypers.batchsize):
                    self.memory.step(m)  

                torch.compiler.cudagraph_mark_step_begin()
                self.memory.compute_advantage()
                frozen_probs = []
                v_target_list = []
            
                for _ in range(hypers.batchsize//hypers.minibatch):
                    states,actions,values,_,_,probs,advantages,_ = self.process_sample()
                    v_target = advantages + values
                 
                    for _ in range(hypers.optim_steps): # sample reuse N_pi = 32  
                        # -
                        p_out,_ = self.p_net(states) 
                        dist = Categorical(probs=p_out)
                        new_probs = dist.log_prob(actions)
                        ratio = torch.exp(new_probs - probs.flatten(0,1)) 
                        p1 = ratio * advantages
                        p2 = torch.clamp(ratio,1+hypers.epsilon,1-hypers.epsilon) * advantages 
                        loss_policy = - torch.mean(torch.min(p1,p2))
                        # -
                        new_values = self.v_net(states) 
                        loss_value = F.smooth_l1_loss(new_values.squeeze(), v_target)
                        # -
                        loss = loss_policy + loss_value - (hypers.beta * dist.entropy().mean())
                        self.optim.zero_grad(set_to_none=True)
                        loss.backward()
                        self.optim.step()

                    frozen_probs.append(dist.probs)
                    v_target_list.append(v_target)
                    
                    self.writter.add_scalar("ppo/Loss policy",loss_policy)
                    self.writter.add_scalar("ppo/Loss value",loss_value)
                    self.writter.add_scalar("ppo/total loss",loss)
                    self.writter.add_scalar("ppo/episode rewards",self.memory.traj_reward()[0].mean())

                frozen_probs = torch.stack(frozen_probs) 
                v_targets = torch.stack(v_target_list) 
                self.memory.update_prob(frozen_probs)
                self.memory.update_v_target(v_targets) 
           
                for _ in range(hypers.e_aux): # auxiliary phase 
                    for _ in range(hypers.batchsize//hypers.minibatch):   
                        states,actions,values,v_policy,v_targets,probs,advantages,dist_prob = self.process_sample()
                        
                        l_v_aux = F.smooth_l1_loss(v_policy,v_target) 
                        p_out,_ = self.p_net(states)
                        new_dist = Categorical(probs=p_out)
                        l_joint = l_v_aux + (hypers.beta_clone * kl(dist_prob,new_dist).mean()) 
                      
                        new_values = self.v_net(states) 
                        l_value = F.smooth_l1_loss(new_values,v_targets) 
                  
                        loss_aux = l_joint + l_value
                        self.optim.zero_grad(set_to_none=True)
                        loss_aux.backward()
                        self.optim.step()
                        
                        self.writter.add_scalar("auxiliary/loss aux value",l_v_aux)
                        self.writter.add_scalar("auxiliary/loss joint",l_joint)
                        self.writter.add_scalar("auxiliary/loss value",l_value)

                if n%400: # reset every 400 steps
                    self.env = env()

                if n%10_000:
                    self.save(n)


if __name__ == "__main__":
    main().run(start=True)
