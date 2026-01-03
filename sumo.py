import torch,sys,os,warnings,gymnasium_sudoku

import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.kl import kl_divergence as kl

import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from dataclasses import dataclass
from collections import deque
from itertools import chain
from tqdm import tqdm

os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYTHONOPTIMIZE"] = "1" # also disable asserts
warnings.filterwarnings("ignore")
torch.set_printoptions(precision=4, sci_mode=False)

@dataclass(frozen=False)
class Hypers:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = 100
    num_envs = 5
    max_steps = 30_000
    batchsize = 512
    minibatch = 32
    e_aux = 6
    lr = 5e-4
    gamma = .99
    lambda_ = .99
    epsilon = .2
    beta = 5e-1         # entropy coeff
    beta_clone = 1      # kl coeff in the aux phase
    optim_steps = 10    # defualt 32 as seen in the original paper
    
hypers = Hypers()

def env():
    def fn():
        x = gym.make("sudoku-v0",horizon=hypers.horizon)
        return x 
    return AsyncVectorEnv([fn for _ in range(hypers.num_envs)])

def process_obs(x): # -> one hot encoding + mask
    x = x.long() 
    m = (x == 0).unsqueeze(1).float()
    x = F.one_hot(x,num_classes=10).permute(0,-1,1,2).float() 
    return torch.cat([x,m],dim=1) 

@torch.no_grad()
def w_init(l):
    if isinstance(l,(nn.Conv2d,nn.Linear)):
        nn.init.orthogonal_(l.weight)
        l.bias.fill_(0.0)


class p_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.LazyConv2d(64,1,1)
        self.c2 = nn.LazyConv2d(128,3,1,padding=1)
        self.c3 = nn.LazyConv2d(128,3,1,padding=1)

        self.emb = nn.Parameter(torch.zeros(1,81,128))
        self.attn = nn.MultiheadAttention(128,4,batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.l1 = nn.LazyLinear(128)
        self.l2 = nn.LazyLinear(128)
 
        self.pos = nn.LazyLinear(1)
        self.num = nn.LazyLinear(10)
        self.v_aux = nn.LazyLinear(1) 
    
    def forward(self,s):
        x = self.c1(s)
        x = F.silu(self.c2(x))  
        x = F.silu(self.c3(x))
        x = x.flatten(2).transpose(-1,1) # -> torch.Size([1,81,128])

        x = x + self.emb 
        x,_= self.attn(x,x,x,average_attn_weights=True) 
        x = self.norm(x)
        x = F.silu(self.l1(x))
        x = F.silu(self.l2(x))
               
        pos = self.pos(x).squeeze(-1) # cell position block
        pos = self.pos_mask(s,pos)
        pos = F.softmax(pos,-1)
        dist_pos = Categorical(probs=pos)
        sample_pos = dist_pos.sample()
        
        num_logits = self.num(x)  # cell value block
        idx = torch.arange(x.size(0),device=x.device)
        o = num_logits[idx,sample_pos]
        o = self.action_mask(o)
        o = F.softmax(o,-1)
        dist_num = Categorical(probs=o)
        sample_num = dist_num.sample()

        v_aux = self.v_aux(x.mean(1))                        
        return (dist_pos,sample_pos),(dist_num,sample_num),v_aux.squeeze()

    def pos_mask(self,s,x): # mask untouchable cells
        s = s.argmax(1)
        mask = (s!=0).flatten(1)
        value = -float("inf")
        return torch.masked_fill(x,mask,value)

    def action_mask(self,x): # min(cell value) = 1 
        mask = torch.zeros_like(x,dtype=torch.bool)   
        mask[:,0] = True
        value = -float("inf")
        return torch.masked_fill(x,mask,value)

class v_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.LazyConv2d(64,1,1)
        self.c2 = nn.LazyConv2d(64,3,1) 
        self.l1 = nn.LazyLinear(512) 
        self.l2 = nn.LazyLinear(128)
        self.v = nn.LazyLinear(1)  

    def forward(self,x):
        x = F.silu(self.c1(x))
        x = F.silu(self.c2(x)) # -> torch.Size([n env, 3136])
        x = F.silu(self.l1(x.flatten(start_dim=1)))
        x = F.silu(self.l2(x))
        return self.v(x).squeeze()
        

class memory: # Replay buffer class
    def __init__(self,env:AsyncVectorEnv,p_net,v_net):
        N = hypers.num_envs
        B = hypers.batchsize 

        self.state = torch.empty((B,N,9,9),device=hypers.device,dtype=torch.half) 
        self.action = torch.empty((B,3,N),device=hypers.device,dtype=torch.float32)

        self.values = torch.empty((B,N),device=hypers.device,dtype=torch.float32)
        self.values_aux = torch.empty((B,N),device=hypers.device,dtype=torch.float32)
        self.v_target = torch.empty((B,N),device=hypers.device,dtype=torch.float32)
        self.rewards = torch.empty((B,N),device=hypers.device,dtype=torch.float32) 
        self.dones = torch.empty((B,N),device=hypers.device,dtype=torch.float32)

        self.pos_probs = torch.empty((B,N,81),device=hypers.device,dtype=torch.float32)
        self.num_probs = torch.empty((B,N,10),device=hypers.device,dtype=torch.float32) 
        self.log_prob = torch.empty((B,N),device=hypers.device,dtype=torch.float32)

        self.advantages = torch.empty((B,N),device=hypers.device,dtype=torch.float32) 

        self.env = env
        self._observation = torch.as_tensor(self.env.reset()[0],device=hypers.device) 
        self.p_net = p_net
        self.v_net = v_net
        self.gamma = hypers.gamma   
        self._lambda_ = hypers.lambda_ 
        self.rewards_deque = deque(maxlen=hypers.num_envs+2)
        self.total_steps_deque = deque(maxlen=hypers.num_envs+2)
        self.episode_reward = torch.empty(self.env.num_envs).float()
        self.total_steps = torch.empty(hypers.num_envs).float()
          
    @torch.no_grad()
    def step(self,num_it):
        pos_data,num_data,v_policy = self.p_net(process_obs(self._observation))
        self.pos_probs[num_it].copy_(pos_data[0].probs)
        self.num_probs[num_it].copy_(num_data[0].probs)
        # joint probability distribution
        log_prob = pos_data[0].log_prob(pos_data[1]) + num_data[0].log_prob(num_data[1])
        self.log_prob[num_it].copy_(log_prob)
        
        value = self.v_net(process_obs(self._observation))
        
        pos = pos_data[-1]
        xpos = pos // 9 ; ypos = pos % 9
        cell_value = num_data[-1]
        action = torch.stack((xpos,ypos,cell_value)).cpu().numpy() # shape -> [x_n...][y_n...][z_n...]
        # self.env.action_space.sample() >>> (array([0, 5]), array([2, 6]), array([3, 4])) 
      
        state,reward,done,_,_ = self.env.step(action)
        reward = np.round(reward,2)
                    
        self.episode_reward += reward
        self.total_steps += 1 
        done_envs = np.where(done==True)[0]
        if done_envs.shape[0] > 0:
            self.rewards_deque.append(self.episode_reward[done_envs].tolist())
            self.total_steps_deque.append(self.total_steps[done_envs].tolist())

        self.state[num_it].copy_(torch.as_tensor(self._observation)) 
        self.action[num_it].copy_(torch.as_tensor(action)) 
        self.values[num_it].copy_(value) 
        self.values_aux[num_it].copy_(v_policy)
        self.rewards[num_it].copy_(torch.as_tensor(reward))
        self.dones[num_it].copy_(torch.as_tensor(done)) 
         
        self._observation = torch.as_tensor(state,device=hypers.device)
   
    @torch.compile()
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
            self.action[idx].transpose(1, 2).flatten(0,1),
            self.values[idx].flatten(0,1),
            self.values_aux[idx].flatten(0,1),
            self.v_target[idx].flatten(0,1),
            self.advantages[idx].flatten(),

            self.log_prob[idx].flatten(0,1),
            self.pos_probs[idx],
            self.num_probs[idx]
        )

    def update_pos_prob(self,x):
        x = x.reshape(*self.pos_probs.shape) 
        self.pos_probs = x

    def update_num_prob(self,x): 
        x = x.reshape(*self.num_probs.shape)
        self.num_probs = x

    def update_v_target(self,x): 
        x = x.reshape(*self.v_target.shape) 
        self.v_target = x
 
    def traj_reward(self):
        return list(map(torch.tensor,(self.rewards_deque,self.total_steps_deque)))


class main:
    def init_nets(self):
        self.p_net = p_net().to(hypers.device)
        self.v_net = v_net().to(hypers.device)
    
        self.p_net(process_obs(torch.randint(0,9,(self.env.reset()[0].shape),device=hypers.device)))
        self.v_net(process_obs(torch.randint(0,9,(self.env.reset()[0].shape),device=hypers.device)))
    
        self.p_net.apply(w_init) ; self.p_net.compile() 
        self.v_net.apply(w_init) ; self.v_net.compile()

    def __init__(self):
        self.env = env() 
        self.init_nets()
        self.memory = memory(self.env,self.p_net,self.v_net)
        self.optim = Adam(chain(self.p_net.parameters(),self.v_net.parameters()),lr=hypers.lr)    
        self.writter = SummaryWriter("./")

    def save(self,n):
        data = {
            "policy state":self.p_net.state_dict(),
            "value state":self.v_net.state_dict(),
            "value optim":self.optim.state_dict()
        }
        torch.save(data,f"./model-{n}")

    def load(self,path):
        self.p_net.load_state_dict(torch.load(path),strict=True)
        self.v_net.load_state_dict(torch.load(),strict=True)
        self.optim.load_state_dict(torch.load())

    def run(self,start=False):
        if start:
            for n in tqdm(range(hypers.max_steps),total=hypers.max_steps):
                for m in range(hypers.batchsize):
                    self.memory.step(m) 
                
                torch.compiler.cudagraph_mark_step_begin()
                self.memory.compute_advantage()
                frozen_pos_probs = []
                frozen_num_probs = []
                v_target_list = []
            
                for i in range(hypers.batchsize//hypers.minibatch):
                    states,actions,values,_,_,advantages,log_prob,_,_ = self.memory.sample(hypers.minibatch)
                    assert advantages.shape == values.shape
                    v_target = advantages + values
                                
                    for _ in range(hypers.optim_steps): # sample reuse N_pi = 32, as seen in the paper 
                        pos_data,num_data,_ = self.p_net(process_obs(states))
                        new_log_prob = pos_data[0].log_prob(pos_data[1]) + num_data[0].log_prob(num_data[1]) 
                        assert new_log_prob.shape == log_prob.shape
                        
                        ratio = torch.exp(new_log_prob - log_prob.squeeze()) 
                        assert ratio.shape == advantages.shape
                        p1 = ratio * advantages
                        p2 = torch.clamp(ratio,1-hypers.epsilon,1+hypers.epsilon) * advantages 
                        loss_policy = - torch.mean(torch.min(p1,p2))
             
                        new_values = self.v_net(process_obs(states)) 
                        loss_value = F.smooth_l1_loss(new_values.squeeze(), v_target)
                        
                        entropy = (pos_data[0].entropy() + num_data[0].entropy()).mean() 

                        loss = loss_policy + loss_value - (hypers.beta * entropy)
                        self.optim.zero_grad(set_to_none=True)
                        loss.backward()
                        self.optim.step()
                    
                    frozen_pos_probs.append(pos_data[0].probs)
                    frozen_num_probs.append(num_data[0].probs)
                    v_target_list.append(v_target)
                    
                    if i%50 == 0:
                        self.writter.add_scalar("main/Loss policy",loss_policy)
                        self.writter.add_scalar("main/Loss value",loss_value)
                        self.writter.add_scalar("main/total loss",loss)
                        self.writter.add_scalar("main/entropy",entropy)
                        self.writter.add_scalar("main/action variance",actions.var())
                        self.writter.add_scalar("main/episode rewards",self.memory.traj_reward()[0].mean())

                self.memory.update_pos_prob(torch.stack(frozen_pos_probs))
                self.memory.update_num_prob(torch.stack(frozen_num_probs))
                self.memory.update_v_target(torch.stack(v_target_list)) 
            
                for _ in range(hypers.e_aux): # auxiliary phase 
                    for _ in range(hypers.batchsize//hypers.minibatch):   
                        states,actions,_,v_policy,v_targets,_,log_prob,pos_probs,num_probs = self.memory.sample(hypers.minibatch)
                        
                        l_v_aux = F.smooth_l1_loss(v_policy,v_targets) 
                        pos_data,num_data,_ = self.p_net(process_obs(states)) 

                        new_pos_probs = pos_data[0] # already an instance of the Categorical class
                        old_pos_probs = pos_probs.flatten(0,1)
                        assert new_pos_probs.probs.shape == old_pos_probs.shape
                        old_pos_probs = Categorical(probs=old_pos_probs)
                        pos_kl = kl(old_pos_probs,new_pos_probs)
                    
                        new_num_probs = num_data[0] # also an instance of the Categorical class
                        old_num_probs = num_probs.flatten(0,1) 
                        assert new_num_probs.probs.shape == old_num_probs.shape
                        old_num_probs = Categorical(probs=old_num_probs)
                        num_kl = kl(old_num_probs,new_num_probs)
                         
                        kl_div = (pos_kl + num_kl).mean()
                        l_joint = l_v_aux + (hypers.beta_clone * kl_div) 
                        
                        new_values = self.v_net(process_obs(states)) 
                        l_value = F.smooth_l1_loss(new_values,v_targets) 
                 
                        loss_aux = l_joint + l_value
                        self.optim.zero_grad(set_to_none=True)
                        loss_aux.backward()
                        self.optim.step()
                    
                        self.writter.add_scalar("auxiliary/loss aux value",l_v_aux)
                        self.writter.add_scalar("auxiliary/loss joint",l_joint)
                        self.writter.add_scalar("auxiliary/loss value",l_value)
                   
                if n%2_000 == 0:
                    self.save(n)

if __name__ == "__main__":
    main().run(start=False)
