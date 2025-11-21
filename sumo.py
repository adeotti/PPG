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


@dataclass(frozen=False)
class Hypers:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 2
    num_games = 1
    batchsize = 15
    minibatch = 5
    e_aux = 6
    lr = 1
    gamma = 1
    lambda_ = 1
    epsilon = 1
    


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
        x = self.c1(x)
        x = F.relu(self.c2(x)) # -> torch.Size([n env, 3136])
        x = F.relu(self.l1(x.flatten(start_dim=1)))
        x = F.relu(self.l2(x))
        return F.relu(self.v(x)) 


class memory: # data collection class 
    def __init__(self,env:AsyncVectorEnv,p_net,v_net):
        N = hypers.num_envs
        B = hypers.batchsize 
        self.state = torch.empty((B,N,9,9),device=hypers.device,dtype=torch.half)
        self.action = torch.empty((B,3,N),device=hypers.device,dtype=torch.float32)
        self.values = torch.empty((B,N,1),device=hypers.device,dtype=torch.float32)
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
        policy_output,_= self.p_net(process_obs(self._observation))
        value = self.v_net(process_obs(self._observation))
        distribution = Categorical(policy_output)
        action = distribution.sample()
        prob = distribution.log_prob(action)
        
        assert torch.equal(action.T.T,action)
        action = action.T.cpu().numpy()
        state,reward,done,_,_ = self.env.step(action)
next_value = self.v_net(process_obs(self._observation)).unsqueeze(0)
        print(self.values.shape)
        print(next_value.shape)
        _values = torch.cat([self.values,next_value])
        gae = torch.zeros_like(self.rewards[0], device=hypers.device) 
        print(self.dones.shape)
        print(self.rewards.shape)
        print(_values[1:].shape)
        print(_values[:-1].shape)
        td = self.rewards.clone().add_(self.gamma * _values[1:]) #* (1 - self.dones))#.sub_(_values[:-1])
        sys.exit()
        for n in reversed(range(len(self.rewards))): 
            gae.mul_(self._lambda_ * self.gamma * (1-self.dones[n])).add_(td[n])
            self.advantages[n].copy_(gae)
         
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
        self.prob[num_it].copy_(prob)
        self.rewards[num_it].copy_(torch.as_tensor(reward))
        self.dones[num_it].copy_(torch.as_tensor(done)) 
        self.dist_prob[num_it].copy_(distribution.probs)
        self._observation = state        

    # @torch.compile(mode="reduce-overhead",fullgraph=True)
    def compute_advantage(self): 
        next_value = self.v_net(process_obs(self._observation)).unsqueeze(0)
        _values = torch.cat([self.values,next_value]).squeeze(-1)
        gae = torch.zeros_like(self.rewards[0], device=hypers.device)  
        td = self.rewards.clone().add_(self.gamma * _values[1:] * (1 - self.dones)).sub_(_values[:-1])
        for n in reversed(range(len(self.rewards))): 
            gae.mul_(self._lambda_ * self.gamma * (1-self.dones[n])).add_(td[n])
            self.advantages[n].copy_(gae)
         
    def sample(self,minibatch): 
        start = self.pointer 
        end = self.pointer + minibatch 
        self.pointer = end
        return (
            self.state[start:end].flatten(0,1),
            self.action[start:end],
            self.values[start:end],
            self.prob[start:end],
            self.advantages[start:end],
            self.dist_prob[start:end]
        )

    def traj_reward(self):
        return list(map(torch.tensor,(self.finished_reward,self.log_total_steps)))


class main:
    def init_nets(self):
        random = torch.tensor(self.env.reset()[0],device=hypers.device)  
        self.p_net(process_obs(random))
        self.v_net(process_obs(random))
    
        self.p_net.apply(w_init)
        self.v_net.apply(w_init)

        # self.p_net.compile()
        # self.v_net.compile()

    def __init__(self):
        self.p_net = p_net().to(hypers.device)
        self.v_net = v_net().to(hypers.device)
        self.env = env()
        self.init_nets()
        self.memory = memory(self.env,self.p_net,self.v_net)
        
        self.p_optim = Adam(self.p_net.parameters(),lr=hypers.lr)
        self.v_optim = Adam(self.v_net.parameters(),lr=hypers.lr)

        # self.writter = SummaryWriter("./")

    def save(self,n):
        data = {
            "policy state":self.p_net.state_dict(),
            "policy optim":self.p_optim.state_dict(),
            "value state":self.v_net.state_dict(),
            "value optim":self.v_optim.state_dict()
        }
        torch.save(data,f"./model-{n}")

    def run(self,start=False):
        if start:
            for n in range(hypers.num_games):
                for m in range(hypers.batchsize):
                    self.memory.step(m)  

                torch.compiler.cudagraph_mark_step_begin()
                self.memory.compute_advantage() 

                for _ in range(hypers.batchsize//hypers.minibatch):
                    states,actions,values,probs,advantages,dist_prob = self.memory.sample(hypers.minibatch)
                    
                    # policy optim  
                    p_out,_ = self.p_net(process_obs(states)) 
                    dist = Categorical(probs=p_out)
                    new_probs = dist.log_prob(actions)
                    ratio = (new_probs - probs).exp()
                    p1 = ratio * advantages
                    p2 = torch.clamp(ratio,1+hypers.epsilon,1-hypers.epsilon) * advantages
                    loss_policy = - (torch.mean(torch.min(p1,p2)) + (hypers.beta * dist.entropy().mean()))
                    self.p_optim.zero_grad(set_to_none=True)
                    loss_policy.backward()
                    self.p_optim.step()

                    # value optim
                    new_values = self.v_net(process_obs(states))
                    vtarget
                    loss_value

                
                   
new_values = self.v_net(process_obs(states))
                for _ in range(hypers.e_aux): # auxiliary phase
                    pass

                
        

if __name__ == "__main__":
    main().run(start=True)









