import sympy as sp
import numpy as np
from abc import ABC
import gym
from gym import spaces
from env.Helicopter import Helicopter
from env.controller import Controller
from utils import save_files

def check_diverge(env):
def done_diverge(env):
    observation = list((env.all_obs[env.counter]) - 200)
    reward = 
def find_next_state(env)

