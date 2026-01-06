import torch,sys,os,warnings

import torch.nn as nn
from torch.distributions import Bernoulli
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.kl import kl_divergence as kl

import retro

env = retro.make(game="Airstriker-Genesis",render_mode="human")
env.reset()

for n in range(1000):
    env.step(env.action_space.sample())
    env.render()
