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
        env = retro.make(game="Airstriker-Genesis",render_mode=None)
        # single env shape (224, 320, 3)
        return env
    return AsyncVectorEnv([thunk for _ in range(hypers.num_envs)])

def process_obs(x):
    return torch.as_tensor(x,device=hypers.device).permute(0,-1,1,2)

class policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.LazyConv2d(64,1,1)
        self.c2 = nn.LazyConv2d(64,1,1)
        self.c3 = nn.LazyConv2d(64,1,1)
        self.vaux = nn.LazyLinear(1)

    def forward(self,x):
        pass



    
