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
    num_envs = 2

hypers = Hypers()

def env():
    def fn():
        x = gym.make("sudoku-v0")
        return x 
    return AsyncVectorEnv([fn for _ in range(hypers.num_envs)])

def process_obs(x): # - > one hot encoding + mask
    x = torch.tensor(x,dtype=torch.int64,device=hypers.device)
    m = (x == 0).unsqueeze(1).to(torch.float32)
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
        self.c2 = nn.LazyConv2d(128,3,1)
        self.c3 = nn.LazyConv2d(128,3,1)
        self.l1 = nn.LazyLinear(512)
        self.l2 = nn.LazyLinear(256)
        self.l3 = nn.LazyLinear(128)
        self.p_head = nn.LazyLinear(30) # policy head
        self.v_aux = nn.LazyLinear(1)   # auxiliary value head
    
    def forward(self,x): 
        x = self.c1(x)
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x)) 
        x = F.relu(self.l1(x.flatten(start_dim=1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        p = self.p_head(x) 
        v = self.v_aux(x)
        return p  

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


if __name__ == "__main__":
    n = v_net()
    e = env()
    d = torch.tensor(e.reset()[0],dtype=torch.float32)
    print(n(process_obs(d)))
    # print(n(process_obs(x).unsqueeze(0)).shape)

