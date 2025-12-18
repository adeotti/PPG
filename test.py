import gymnasium_sudoku,torch,sys
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

def envi():
    x = gym.make("sudoku-v0",render_mode="human",horizon=100)
    return x 

def process_obs(x): 
    m = (x == 0).unsqueeze(1).float()
    x = F.one_hot(x,num_classes=10).permute(0,-1,1,2).float() 
    return torch.cat([x,m],dim=1) 

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
    
    def forward(self,x):
        x = self.c1(x)
        x = F.silu(self.c2(x))  
        x = F.silu(self.c3(x))
        x = x.flatten(2).transpose(-1,1) 
        x = x + self.emb 
        x,_ = self.attn(x,x,x)  
        x = self.norm(x)
        x = F.silu(self.l1(x))
        x = F.silu(self.l2(x))
        pos = self.pos(x).squeeze(-1)
        pos = F.softmax(pos,-1)
        pos = pos.argmax().item() 
        num_logits = self.num(x)  
        idx = torch.arange(x.size(0),device=x.device)
        o = num_logits[idx,pos]
        o = self.cll_mask(o)
        o = F.softmax(o,-1)
        num = o.argmax().item()                      
        return pos,num

    def cll_mask(self,x):
        m = torch.zeros_like(x,dtype=torch.bool)   
        m[:,0] = True
        value = -float("inf")
        return torch.masked_fill(x,m,value)

policy = p_net()
policy(process_obs(torch.randint(0,9,(1,9,9))))
t_policy = torch.load("./model-4000",map_location="cpu")["policy state"]
policy.load_state_dict(t_policy,strict=False)

env = envi()
obs = env.reset()[0]
r = 0
for n in range(2_000):
    pos,num = policy(process_obs(torch.tensor(obs,dtype=torch.int64).unsqueeze(0)))
    v(process_obs(torch.tensor(obs,dtype=torch.int64).unsqueeze(0)))
    xpos = pos // 9 ; ypos = pos % 9
    action = np.stack((xpos,ypos,num),axis=-1).reshape(3)
    obs,reward,done,_,_ = env.step(action)
    r+=reward
    env.render()
    if done:
        print(r)
        r = 0
        obs = env.reset()[0]


