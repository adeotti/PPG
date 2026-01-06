import torch,sys,os,warnings

import torch.nn as nn
from torch.distributions import Bernoulli
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.kl import kl_divergence as kl

import retro

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation,ResizeObservation
from gymnasium.vector import AsyncVectorEnv
from dataclasses import dataclass
from collections import deque
from itertools import chain
from tqdm import tqdm


@dataclass(frozen=False)
class Hypers:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 3
    max_steps = 10
    batch_size = 4
    mini_batch = 2
    optim_steps = 2
    lr = 1e-4
    gamma = .99
    lambda_ = .99

hypers = Hypers()

def vec():
    def thunk():
        e = retro.make(game="Airstriker-Genesis",render_mode=None) # single env shape (224, 320, 3)
        e = ResizeObservation(e,(100,100))
        e = GrayscaleObservation(e)
        return e
    return AsyncVectorEnv([thunk for _ in range(hypers.num_envs)])

def process_obs(x):
    return torch.as_tensor(x,device=hypers.device,dtype=torch.float).unsqueeze(1)


class policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.LazyConv2d(32,1,1,0)
        self.c2 = nn.LazyConv2d(64,2,2,0)
        self.c3 = nn.LazyConv2d(64,2,2,0)
        self.c4 = nn.LazyConv2d(64,2,2,0)
        self.l1 = nn.LazyLinear(5120)
        self.l2 = nn.LazyLinear(2048)
        self.l3 = nn.LazyLinear(512)
        self.l4 = nn.LazyLinear(12)
        self.vaux = nn.LazyLinear(1)

    def forward(self,x):
        x = F.silu(self.c1(x))
        x = F.silu(self.c2(x))
        x = F.silu(self.c3(x))
        x = F.silu(self.c4(x))
        x = F.silu(self.l1(x.flatten(1)))
        x = F.silu(self.l2(x))
        x = F.silu(self.l3(x))
        return self.l4(x), self.vaux(x) 

class value(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.LazyConv2d(16,4,2,1)  
        self.c2 = nn.LazyConv2d(32,4,2,1)
        self.c3 = nn.LazyConv2d(32,3,2,1)
        self.l1 = nn.LazyLinear(256)
        self.l2 = nn.LazyLinear(128)
        self.l3 = nn.LazyLinear(1)
        
    def forward(self, x):
        x = F.silu(self.c1(x))
        x = F.silu(self.c2(x))
        x = F.silu(self.c3(x))
        x = F.silu(self.l1(x.flatten(1)))
        x = F.silu(self.l2(x))
        return self.l3(x)


class replay_buffer:
    def __init__(self,env,policy_net,value_net):
        self.env = env

        self.states = torch.empty(
                (hypers.batch_size,*self.env.reset()[0].shape),device=hypers.device,dtype=torch.float32
        )
        self.actions = torch.empty((hypers.batch_size,hypers.num_envs,12),device=hypers.device,dtype=torch.float32)
        self.rewards = torch.empty((hypers.batch_size,hypers.num_envs),device=hypers.device,dtype=torch.float32)
        self.dones = torch.empty((hypers.batch_size,hypers.num_envs),device=hypers.device,dtype=torch.float32)
        self.advantages = torch.empty((hypers.batch_size,hypers.num_envs,1),device=hypers.device,dtype=torch.float32)
        self.probs = torch.empty((hypers.batch_size,hypers.num_envs,12),device=hypers.device,dtype=torch.float32)
        self.log_probs = torch.empty((hypers.batch_size,hypers.num_envs,12),device=hypers.device,dtype=torch.float32)
        self.values = torch.empty((hypers.batch_size,hypers.num_envs,1),device=hypers.device,dtype=torch.float32)
        self.vaux = torch.empty((hypers.batch_size,hypers.num_envs,1),device=hypers.device,dtype=torch.float32)

        self.obs = process_obs(env.reset()[0])
        self.policy_net = policy_net
        self.value_net = value_net

        self.ep_reward_buffer = torch.zeros((hypers.batch_size,))
        self.ep_reward = deque(maxlen=10)
    
    @torch.no_grad()
    def step(self,num_it):
        p_logits,vaux = self.policy_net(self.obs)
        value = self.value_net(self.obs)
        dist = Bernoulli(logits=p_logits)
        sample = dist.sample()
        dist_prob = dist.log_prob(sample)

        nx_states,reward,dones,_,_ = self.env.step(sample.cpu().numpy())
        #self.ep_reward_buffer += torch.as_tensor(reward)
        #if np.all(reward) : self.ep_reward.append(self.ep_reward_buffer.tolist())::
    
        self.states[num_it].copy_(self.obs.squeeze())
        self.actions[num_it].copy_(torch.as_tensor(sample,device=hypers.device))
        self.rewards[num_it].copy_(torch.as_tensor(reward,device=hypers.device))
        self.dones[num_it].copy_(torch.as_tensor(dones,device=hypers.device))
        self.probs[num_it].copy_(dist.probs)
        self.log_probs[num_it].copy_(dist.log_prob(sample))
        self.values[num_it].copy_(value)
        self.vaux[num_it].copy_(vaux)

        self.obs = process_obs(nx_states)
        
    #@torch.compile()
    @torch.no_grad()
    def compute_advantage(self): 
        next_value = self.value_net(self.obs).unsqueeze(0) 
        _values = torch.cat([self.values,next_value]).squeeze(-1)
        gae = torch.zeros_like(self.rewards[0], device=hypers.device)
        td = self.rewards.clone().add_(hypers.gamma * _values[1:] * (1 - self.dones)).sub_(_values[:-1])
        for n in reversed(range(len(self.rewards))): 
            gae.mul_(hypers.lambda_ * hypers.gamma * (1-self.dones[n])).add_(td[n])
            self.advantages[n].copy_(gae.unsqueeze(-1)) 

    def sample(self,minibatch):
        idx = torch.randperm(hypers.batch_size)[:minibatch]
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.probs[idx],
            self.log_probs[idx],
            self.values[idx],
            self.vaux[idx]
        )


class main:
    def init_nets(self):
        self.policy_net = policy().to(hypers.device)
        self.value_net = value().to(hypers.device)
        
        self.policy_net(process_obs(torch.rand(*(self.env.reset()[0].shape),device=hypers.device)))
        self.value_net(process_obs(torch.rand(*(self.env.reset()[0].shape),device=hypers.device)))
        
        #self.policy_net.apply(init) ; self.policy_net.compile()
        #self.value_net.apply(init) ; self.value_net.compile()

    def __init__(self):
        self.env = vec()
        self.init_nets()
        self.optim = Adam(chain(self.policy_net.parameters(),self.value_net.parameters()),lr=hypers.lr)
        self.buffer = replay_buffer(self.env,self.policy_net,self.value_net)
        # self.writer = SummaryWriter("./")

    def save(self,n):
        data = {
            "policy state": self.policy_net.state_dict(),
            "value state" : self.value_net.state_dict(),
            "optim state" : self.optim.state_dict()
        }
        torch.save(data,f"./model-{n}")

    def train(self,start=False):
        if start:
            for n in tqdm(range(hypers.max_steps),total=hypers.max_steps):
                
                for i in range(hypers.batch_size):
                    self.buffer.step(i)

                self.buffer.compute_advantage()
                sys.exit()
                
                for _ in range(hypers.batch_size//hypers.mini_batch):
                    data = self.buffer.sample(hypers.mini_batch)
                
                    for _ in range(hypers.optim_steps):
                       pass


                

