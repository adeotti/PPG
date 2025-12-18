import torch,sys,os,warnings

import torch.nn as nn
from torch.distributions import Categorical
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
    num_envs = 2

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
        return F.sigmoid(self.l4(x)), self.vaux(x) 

class value(nn.module):
    def __init__(self):
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

class replay_buffer
    def __init__(self,env,policy,value):
        pass
    
    @torch.no_grad()
    def step(self):
        pass

    def advantages(self)
        pass




