import gymnasium_sudoku,torch,sys
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

HORIZON = int(10e3)

def envi():
    x = gym.make(
            "sudoku-v0",
            render_mode="human",
            horizon=HORIZON,
            eval_mode=True,
            render_delay=0.0
        )
    return x 

def process_obs(x): 
    m = (x == 0).unsqueeze(1).float()
    x = F.one_hot(x,num_classes=10).permute(0,-1,1,2).float() 
    return torch.cat([x,m],dim=1) 

class p_net(nn.Module):
    def __init__(self,stochastic):
        super().__init__()
        self.stochastic_policy = stochastic
        self.c1 = nn.LazyConv2d(64,1,1)
        self.c2 = nn.LazyConv2d(128,3,1,padding=1)
        self.c3 = nn.LazyConv2d(128,3,1,padding=1)
        self.emb = nn.Parameter(torch.randn(1,81,128) * 0.02)
        self.attn = nn.MultiheadAttention(128,4,batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.l1 = nn.LazyLinear(128)
        self.l2 = nn.LazyLinear(128)
        self.pos = nn.LazyLinear(1)
        self.num = nn.LazyLinear(10)
        self.v_aux = nn.LazyLinear(1)
        self.register_buffer("attn_mask",self.attn_masks())
    
    def forward(self,s):
        x = self.c1(s)
        x = F.silu(self.c2(x))  
        x = F.silu(self.c3(x))
        x = x.flatten(2).transpose(-1,1) 
        x = x + self.emb
        x,asc= self.attn(x,x,x,attn_mask=self.attn_mask,average_attn_weights=True)
        x = self.norm(x)
        x = F.silu(self.l1(x))
        x = F.silu(self.l2(x))
        pos = self.pos(x).squeeze(-1)
        pos = self.pos_mask(s,pos)
        pos = F.softmax(pos,-1)
        pos = Categorical(probs=pos).sample() if self.stochastic_policy else pos.argmax().item()
        num_logits = self.num(x)  
        idx = torch.arange(x.size(0))
        o = num_logits[idx,pos]
        o = self.action_mask(o)
        o = F.softmax(o,-1)
        num = Categorical(probs=o).sample() if self.stochastic_policy else o.argmax().item()
        return pos,num,asc

    def pos_mask(self,s,x): 
        s = s.argmax(1)
        mask = (s!=0).flatten(1)
        value = -1e9
        return torch.masked_fill(x,mask,value)

    def action_mask(self,x): 
        mask = torch.zeros_like(x,dtype=torch.bool)   
        mask[:,0] = True
        value = -float("inf")
        return torch.masked_fill(x,mask,value)
    
    def attn_masks(self,N=81):
        indices = torch.arange(N)  

        rows = indices // 9      # shape [81]
        cols = indices % 9       # shape [81]
        boxes = (rows // 3) * 3 + (cols // 3)  # shape [81]

        row_mask = (rows.unsqueeze(0) == rows.unsqueeze(1)).float()
        col_mask = (cols.unsqueeze(0) == cols.unsqueeze(1)).float()
        box_mask = (boxes.unsqueeze(0) == boxes.unsqueeze(1)).float()
        global_mask = torch.ones(N, N)
        return torch.stack([row_mask,col_mask,box_mask,global_mask],dim=0)


def test_trained(rollout_num:int=None,stochastic:bool=True):
    writter = SummaryWriter(
            f"test/stochastic_{rollout_num}_epi_{datetime.now().strftime('%Y%m%d_%H%M%S')}_hor_{HORIZON}"
    )
    policy = p_net(stochastic=stochastic)
    policy(process_obs(torch.randint(0,9,(1,9,9))))
    t_policy = torch.load("./models/sumo_v1_10k",map_location="cpu")["policy state"]
    policy.load_state_dict(t_policy,strict=False)

    env = envi()
    obs = env.reset()[0]
    steps = r = 0
    
    for n in tqdm(range(rollout_num),total=rollout_num):
        pos,num,attn = policy(process_obs(torch.tensor(obs,dtype=torch.int64).unsqueeze(0)))
        xpos = pos // 9 ; ypos = pos % 9
        action = np.stack((xpos,ypos,num),axis=-1).reshape(3)
        obs,reward,done,trunc,_ = env.step(action)
        steps+=1 ; r+=reward
        env.render()
        if done:
            writter.add_scalar("reward_per_ep",r,global_step=n/HORIZON)
            print(f"\nSteps : {steps} | Rewards : {r:.2f} \n{obs}" )
            time.sleep(5)
            steps = r = 0
            obs = env.reset()[0]
            
        
def test_random(rollout_num:int=None):
    writter = SummaryWriter(
            f"test/random_policy_{rollout_num}_epi_{datetime.now().strftime('%Y%m%d_%H%M%S')}_hor_{HORIZON}"
    )
    env = envi()
    obs = env.reset()[0]
    steps = r = 0

    for n in tqdm(range(rollout_num),total=rollout_num):
        obs,reward,done,trunc,_ = env.step(env.action_space.sample())
        steps+=1 ; r+=reward
        env.render()
        if done:
            writter.add_scalar("reward_per_ep",r,global_step=n/HORIZON)
            print(f"\nSteps : {steps} | Rewards : {r:.2f} \n{obs}" )
            time.sleep(5)
            steps = r = 0
            obs = env.reset()[0]


def plot_attn_mask():
    masks = p_net(True).attn_masks()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    titles = ["Head 0: Row","Head 1: Column","Head 2: Region","Head 3: Global"]

    for i, (ax, title) in enumerate(zip(axes, titles)):
        m = masks[i].float().cpu().numpy() 
        im = ax.imshow(m, cmap="binary", interpolation="nearest")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Key Position (0-80)", fontsize=10)
        ax.set_ylabel("Query Position (0-80)", fontsize=10)
        
        for k in range(0, 82, 9):
            ax.axhline(k - 0.5, color="grey", linewidth=1, alpha=0.3)
            ax.axvline(k - 0.5, color="grey", linewidth=1, alpha=0.3)

    plt.tight_layout()
    plt.savefig("attention_masks.png", dpi=150, bbox_inches="tight")
    plt.show()
    

if __name__ == "__main__":
    episodes = HORIZON
    test_trained(episodes,True)
    #test_random(episodes)
    #plot_attn_mask()
