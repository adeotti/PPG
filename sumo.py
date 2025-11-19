import torch,sys
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
    num_envs = 1

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
    x = x.reshape(1,3,9)
    m = torch.zeros_like(x,dtype=torch.bool)  
    m[0,0,-1] = True
    m[0,1,-1] = True
    m[0,-1,0] = True
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


class Memory:
    def __init__(self,env:AsyncVectorEnv):
        N = configs.num_env
        B = configs.batchsize 
        self.state = torch.empty((B,N,1,9,9),device=configs.device,dtype=torch.half)
        self.action = torch.empty((B,N),device=configs.device,dtype=torch.float32)
        self.values = torch.empty((B,N),device=configs.device,dtype=torch.float32)
        self.prob = torch.empty((B,N),device=configs.device,dtype=torch.float32) 
        self.rewards = torch.empty((B,N),device=configs.device,dtype=torch.float32) 
        self.dones = torch.empty((B,N),device=configs.device,dtype=torch.float32) 
        self.dist_prob = torch.empty((B,N,3),device=configs.device,dtype=torch.float32) 
        self.advantages = torch.empty((B,N),device=configs.device,dtype=torch.float32) 

        self.env = env
        self.gamma = configs.gamma
        self._lambda_ = configs.lambda_
        self.data = []
        self.pointer = 0
        self.finished_reward = deque(maxlen=30)
        self.log_total_steps = deque(maxlen=30)
        self.episode_reward = torch.empty(self.env.num_envs).float()
        self.total_steps = torch.empty(configs.num_env).float()
          
    @torch.no_grad()
    def step(self,batchsize,network:network):
        self.pointer = 0 
        self._observation,_ = self.env.reset()
        torch.compiler.cudagraph_mark_step_begin()
        
        self._observation = self.transf_obs(self._observation)
        with torch.amp.autocast(device_type="cuda",dtype=torch.half):
            policy_output, value = network(self._observation)
        distribution = Categorical(policy_output)
        action = distribution.sample()
        prob = distribution.log_prob(action)
        state,reward,done,_,_ = self.env.step(action.cpu().numpy())
        for i in range(self.env.num_envs): # tracking episode rewards and total steps
            self.episode_reward[i] += reward[i] 
            self.total_steps[i] += 1
            if done[i]:
                self.finished_reward.append(self.episode_reward[i])
                self.log_total_steps.append(self.total_steps[i])
                self.episode_reward[i] = 0
                self.total_steps[i] = 0
        
        self.state[n].copy_(self._observation)
        self.action[n].copy_(action)
        self.values[n].copy_(value)
        self.prob[n].copy_(prob)
        self.rewards[n].copy_(torch.as_tensor(reward))
        self.dones[n].copy_(torch.as_tensor(done))
        self.dist_prob[n].copy_(distribution.probs)
        self._observation = state 

    @torch.compile(mode="reduce-overhead",fullgraph=True)
    def compute_advantage(self,network,rewards:Tensor,values:Tensor,dones:Tensor):
        next_state = self.transf_obs(self._observation)
        with torch.amp.autocast(device_type="cuda",dtype=torch.half):
            _,next_value = network(next_state)
        _values = torch.cat([values,next_value.unsqueeze(0)])
        gae = torch.zeros_like(rewards[0], device=configs.device)
        td = rewards.clone().add_(self.gamma * _values[1:] * (1 - dones)).sub_(_values[:-1])
        for n in reversed(range(len(rewards))): 
            gae.mul_(self._lambda_ * self.gamma * (1-dones[n])).add_(td[n])
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



if __name__ == "__main__":
    v = v_net()
    p = p_net()
    e = env()
    d = torch.tensor(e.reset()[0],dtype=torch.float32)
    print(p(process_obs(d)))
    # print(n(process_obs(x).unsqueeze(0)).shape)



    # self.compute_advantage(network,self.rewards,self.values,self.dones)










