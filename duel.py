from melee import enums
from melee_env.env import MeleeEnv
from melee_env.agents.basic import *
from melee_env.agents.util import *

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from nn_list import *
import math
import random
import argparse
import time
import os

from Bot import nnAgent

observation_space = ObservationSpace()
action_space = ActionSpace()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = './models/2024-07-19-05-17-58_actor.pt'

agent = nnAgent(observation_space)
opp = CPU(enums.Character.FOX, 9)
# agent.net.load_state_dict(torch.load(model_path))

players = [agent, opp] #CPU(enums.Character.KIRBY, 5)]
env = MeleeEnv("ssbm.iso", players, fast_forward=True, ai_starts_game=True)
env.start()
agent.opponent_controller = opp.controller

state, done = env.setup(enums.Stage.FINAL_DESTINATION)
while not done:
    for i in range(len(players)):
        players[i].act(state)
    obs, _, _, _ = agent.observation_space(state, agent.controller.port, opp.controller.port)
    state, done = env.step()