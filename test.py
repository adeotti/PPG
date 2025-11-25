import gymnasium_sudoku,torch,sys
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

env = gym.make("sudoku-v0",render_mode="human")

def process_obs(x): 
    x = torch.tensor(x,dtype=torch.int64).unsqueeze(0) 
    m = (x == 0).unsqueeze(1).to(torch.float32)
    x = F.one_hot(x,num_classes=10).permute(0,-1,1,2).float() 
    return torch.cat([x,m],dim=1) 

def softmax_mask(x): 
    x = x.reshape(x.shape[0],3,9) 
    m = torch.zeros_like(x,dtype=torch.bool)  
    m[:,0,-1] = True
    m[:,1,-1] = True
    m[:,-1,0] = True
    value = -float("inf")
    return torch.masked_fill(x,m,value)

class p_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.LazyConv2d(64,1,1)
        self.c2 = nn.LazyConv2d(128,3,1)
        self.c3 = nn.LazyConv2d(128,3,1)
        self.l1 = nn.LazyLinear(512)
        self.l2 = nn.LazyLinear(256)
        self.l3 = nn.LazyLinear(3*9)
      
    def forward(self,x): 
        x = self.c1(x)
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x)) 
        x = F.relu(self.l1(x.flatten(start_dim=1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x)) 
        p_head = F.softmax(softmax_mask(x),dim=-1)                   
        return p_head

obs = env.reset()[0]
model = p_net()
for _ in range(200):
    dist = model(process_obs(obs))
    action = Categorical(logits=dist).sample().squeeze()
    env.step(action.numpy())
    env.render()
    
