import torch,sys,os,warnings

import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.kl import kl_divergence as kl

import numpy as np

import gym as old_gym
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
import gymnasium as gym
import procgen
from gymnasium.vector import AsyncVectorEnv

from dataclasses import dataclass
from collections import deque
from itertools import chain
from tqdm import tqdm

os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
warnings.filterwarnings("ignore")


def vec_env():
    def make_env():
        x = old_gym.make("procgen:procgen-coinrun-v0")
        x = GymV21CompatibilityV0(env=x)
        return x
    return AsyncVectorEnv([make_env for _ in range(10)])


