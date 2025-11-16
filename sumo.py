import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import gymnasium_sudoku
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass(frozen=False)
class Hypers:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hypers = Hypers()

env = gym.make("sudoku-v0")

def process_obs(x):
    x = torch.tensor(x,dtype=torch.float32,device=hypers.device)
    m = (x == 0).to(torch.float32)
    return torch.stack([x,m],dim=0)

def w_init(l):
    if isinstance(l,(nn.Conv2d,nn.Linear)):
        nn.init.orthogonal_(l.weight)
        l.bias.fill_(0.0)

class policy_n(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d()
        self.c2 = nn.Conv2d()
        self.c3 = nn.Conv2d()
    
        self.l1 = nn.LazyLinear()
        self.l2 = nn.LazyLinear()
        self.l3 = nn.LazyLinear()

        self.p_head = nn.LazyLinear() # policy head
        self.v_head = nn.LazyLinear() # value head
    
    def forward(self,x): 
        x = self.c1(x)
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        p = self.p_head(x) 
        v = self.v_head(x)
        return x


