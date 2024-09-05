
import melee
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssbm_bot"))

from melee_env.agents.util import ObservationSpace, ActionSpace
from DataHandler_meleenv import get_ports, controller_states_different, generate_output
import json



class Slp2MemmapConverter:
    def __init__(self):
        self.obs_state_extractor = ObservationSpace()
    
    def convert(self, file_path:str, memmap_path:str, config_path:str) -> bool:
        console = melee.Console(system="file",
                                allow_old_version=True,
                                path=file_path)
        try:
            console.connect()
        except:
            console.stop()
            return False
        
        self.obs_state_extractor._reset() # TODO  ObservationSpace의 self.done이 정상 동작하지 않음. 
        gamestate = console.step()
        p1_tot_obsstate = []
        p2_tot_obsstate = []
        p1_tot_action = []
        p2_tot_action = []
        p1_tot_reward = []
        p2_tot_reward = []
        while gamestate is not None:
            ## for each gamestate, 
            ## read observationstate(state), action, reward for p1 and p2
            assert len(gamestate.players.keys()) == 2, "일대일 경기만 쓸꺼임."
            p1_port, p2_port = sorted(gamestate.players.keys())
            
            # observation state and reward
            p1_observstate, p1_reward, _, _ = self.obs_state_extractor(gamestate, p1_port, p2_port)
            p2_observstate, p2_reward, _, _ = self.obs_state_extractor(gamestate, p2_port, p1_port)
            p1_tot_obsstate.append(p1_observstate)
            p1_tot_reward.append(p1_reward)
            p2_tot_obsstate.append(p2_observstate)
            p2_tot_reward.append(p2_reward)
            
            # action
            p1_playerstate = gamestate.players[p1_port]
            p2_playerstate = gamestate.players[p2_port]
            
            p1_action = generate_output(p1_playerstate)
            p2_action = generate_output(p2_playerstate)
            
            p1_tot_action.append(p1_action)
            p2_tot_action.append(p2_action)
            
            gamestate = console.step()
        
        ## stack to single numpy array
        p1_tot_obsstate = np.stack(p1_tot_obsstate, axis=0, dtype=np.float32)
        p2_tot_obsstate = np.stack(p2_tot_obsstate, axis=0, dtype=np.float32)
        
        p1_tot_reward = np.stack(p1_tot_reward, axis=0, dtype=np.float32).reshape(-1, 1)
        p2_tot_reward = np.stack(p2_tot_reward, axis=0, dtype=np.float32).reshape(-1, 1)
        
        p1_tot_action = np.stack(p1_tot_action, axis=0, dtype=np.float32)
        p2_tot_action = np.stack(p2_tot_action, axis=0, dtype=np.float32)
        ## concat all information and save index
        index_map = [
        {"p1_state" : p1_tot_obsstate.shape[1]},
        {"p2_state" : p2_tot_obsstate.shape[1]},
        {"p1_reward": p1_tot_reward.shape[1]},
        {"p2_reward": p2_tot_reward.shape[1]},
        {"p1_action": p1_tot_action.shape[1]},
        {"p2_action": p2_tot_action.shape[1]},
        ]
        json.dump(index_map, open(f"{config_path}", "w"))
        
        
        tot_data = np.concatenate((p1_tot_obsstate, p2_tot_obsstate, \
                                  p1_tot_reward, p2_tot_reward, \
                                  p1_tot_action, p2_tot_action), axis=1)
        
        ## convert numpy.ndarray to numpy.memmap
        tot_memmap = make_memmap(memmap_path, tot_data)
        del tot_data
        return tot_memmap
    
        
        
        
def make_memmap(mem_file_name:str, ndarray_to_copy:np.ndarray):
    memmap_configs = dict()
    memmap_configs['shape'] = shape = tuple(ndarray_to_copy.shape)
    memmap_configs['dtype'] = dtype = str(ndarray_to_copy.dtype)
    json.dump(memmap_configs, open(f"{mem_file_name}.conf", "w"))
    mm = np.memmap(mem_file_name, mode='w+', shape=shape, dtype=dtype)
    mm[:] = ndarray_to_copy[:]
    mm.flush()

            
def read_memmap(mem_file_name):
    with open(f"{mem_file_name}.conf", "r") as file:
        memmap_configs = json.load(file)
        return np.memmap(mem_file_name, mode='r+', \
            shape=tuple(memmap_configs['shape']), \
                dtype=memmap_configs['dtype'])
            
            