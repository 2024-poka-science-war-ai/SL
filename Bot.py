import melee
import numpy as np
from abc import ABC, abstractmethod
from nn_list import GRU, DeepResLSTMActor
from melee_env.agents.util import *
from melee import enums
import MovesList
from melee_env.agents.util import ObservationSpace, ActionSpace, from_action_space

import torch
import torch.nn.functional as F

from train import SEQ_LEN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Bot:
    def __init__(self, model, controller: melee.Controller, opponent_controller: melee.Controller):
        self.opponent_controller = opponent_controller
        self.drop_every = 180
        self.model = model
        self.controller = controller
        self.frame_counter = 0
        self.observ = ObservationSpace()
        self.action_space = ActionSpace()
        self.delay = 0
        self.pause_delay = 0
        self.firefoxing = False
        
    @from_action_space
    def act(self, gamestate: melee.GameState):
        self.controller.release_all()

        player: melee.PlayerState = gamestate.players.get(self.controller.port)
        opponent: melee.PlayerState = gamestate.players.get(self.opponent_controller.port)

        if opponent.action in MovesList.dead_list and player.on_ground:
            return

        self.frame_counter += 1

        inp, _, _, _ = self.observ(gamestate, self.controller.port, self.opponent_controller.port)
        del self.states[SEQ_LEN-1]
        self.states.insert(0, np.array(inp))
        temp = torch.tensor(np.float32([self.states])).to(device=DEVICE)
        temp2 = torch.tensor(np.float32(np.array([self.actions]))).to(device=DEVICE)
        
        a = self.model(temp, temp2)

        action = torch.argmax(a.detach().cpu())
        
        del self.actions[SEQ_LEN-2]
        zero = np.zeros(45)
        zero[action] = 1
        self.actions.insert(0, zero)
        
        return action

class Agent(ABC):
    def __init__(self):
        self.agent_type = "AI"
        self.controller = None
        self.port = None  # this is also in controller, maybe redundant?
        self.action = 0
        self.press_start = False
        self.self_observation = None
        self.current_frame = 0

    @abstractmethod
    def act(self):
        pass

class nnAgent(Agent):
    def __init__(self, obs_space, controller=None, opponent_controller=None, path=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DeepResLSTMActor(512, 57, 3).to(self.device)
        if path!=None:
            self.net.load_state_dict(torch.load(path))
            self.net.eval()
            print(path+" succesfully loaded.")
        self.character = enums.Character.FOX
        self.controller = controller
        self.opponent_controller = opponent_controller

        self.action_space = ActionSpace()
        self.observation_space = obs_space
        self.action = 0
        self.prev_state = self.net.initial_state(1, self.device)
        self.prev_action = torch.zeros((1,1,8)).to(device=DEVICE)
        # self.states = [np.zeros(36) for _ in range(SEQ_LEN)]
        # self.actions = [np.zeros(45) for _ in range(SEQ_LEN-1)]

    @from_action_space    
    def act(self, gamestate):
        inp, _, _, _ = self.observation_space(gamestate, self.controller.port, self.opponent_controller.port)
        temp = torch.tensor(inp).reshape((1,1,-1)).float().to(device=DEVICE)

        buttons, main_stick_output, c_stick_output, trigger_output, self.prev_state = self.net(torch.concat([temp, self.prev_action], axis=2), self.prev_state)
        
        sample_action = False
        if sample_action == True:
            # Multinomial을 이용하여 인덱스 선택
            button_indices = [torch.multinomial(F.softmax(button), 1).item() for button in buttons]
            main_stick_index = torch.multinomial(F.softmax(main_stick_output), 1).item()
            c_stick_index = torch.multinomial(F.softmax(c_stick_output), 1).item()
            trigger_index = torch.multinomial(F.softmax(trigger_output), 1).item()
        
        else:
            # Argmax를 이용하여 인덱스 선택
            button_indices = [torch.argmax(button).item() for button in buttons]
            main_stick_index = torch.argmax(main_stick_output).item()
            c_stick_index = torch.argmax(c_stick_output).item()
            trigger_index = torch.argmax(trigger_output).item()
            
        self.prev_action = torch.zeros((1,1,8)).to(device=DEVICE)
        for i in range(5):
            self.prev_action[0,0,i] = button_indices[i]
        self.prev_action[0,0,5] = main_stick_index
        self.prev_action[0,0,6] = c_stick_index
        self.prev_action[0,0,7] = trigger_index
        
        
        # 버튼 인덱스를 이진수로 변환하여 하나의 숫자로 만듦
        button_value = sum([bit * (2 ** i) for i, bit in enumerate(button_indices)])
    
        # 최종 매핑
        action_index = button_value * 17 * 17 * 5 + main_stick_index * 17 * 5 + c_stick_index * 5 + trigger_index
        
        self.action = action_index
        return self.action