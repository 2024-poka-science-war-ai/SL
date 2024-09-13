from torch.utils.data import Dataset
from configuration_dataset import DatasetConfig
import numpy as np
import os
import json
import torch
import time
import psutil
import sys
from contextlib import contextmanager

def close_memmap(mm):
    mm.flush()  # Flush changes if using mode='r+'
    del mm  # Remove reference

@contextmanager
def open_memmap(mem_file_name):
    mm = read_memmap(mem_file_name)
    try:
        yield mm
    finally:
        close_memmap(mm)
        
TIME_LIST = [[], [], []]

class SlippiDataset(Dataset):
    def __init__(self, config:DatasetConfig, mm_filelist=None):
        assert mm_filelist is not None
        self.config = config
        self.mm_file_list = self.get_mm_file_names() if mm_filelist is None else mm_filelist
        mm_list = [read_memmap(file) for file, _ in self.mm_file_list]
        
        self.start_indices = [0]
        self.total_len = 0
        self.seq_len = config.seq_len
        
        for mm in mm_list[:-1]:
            if config.duplicate_ok:
                n_elem = mm.shape[0]
            else:
                n_elem = (mm.shape[0] // config.seq_len) + (mm.shape[0]%config.seq_len!=0) ## full data 개수 + padding 있는거 개수
            self.start_indices.append(self.start_indices[-1] + n_elem)
            self.total_len += n_elem
        self.start_indices = torch.tensor(self.start_indices).detach()
        if config.duplicate_ok:
            self.total_len += mm_list[-1].shape[0]
        else:
            self.total_len += (mm_list[-1].shape[0] // config.seq_len) + (mm_list[-1].shape[0]%config.seq_len!=0)
        del mm_list
        
    def get_mm_file_names(self):
        mm_base_path = self.config.basepath
        mm_file_names = []
        for file_name in os.listdir(mm_base_path):
            if file_name.endswith("dat"):
                mm_file_names.append(os.path.join(mm_base_path, file_name))
        return mm_file_names
    def __len__(self):
        return self.total_len
    def __getitem__(self, index):
        target_game_id = np.arange(len(self.start_indices))[index >= self.start_indices][-1]
        filename, player_id = self.mm_file_list[target_game_id]
        with open_memmap(filename) as target_game:
            if self.config.duplicate_ok:
                start_idx = (index - self.start_indices[target_game_id])
            else:
                start_idx = (index - self.start_indices[target_game_id]) * self.config.seq_len
            end_idx = min(start_idx + self.seq_len, len(target_game))
            if (end_idx - start_idx) < self.seq_len:  # add padding
                
                padding = np.zeros((self.seq_len - (end_idx - start_idx), target_game.shape[1]), dtype=target_game.dtype)
                final_mm = np.concatenate([target_game[start_idx:end_idx], padding], axis=0)
                del padding
            else:
                final_mm = target_game[start_idx:end_idx]
            
        temp = torch.from_numpy(final_mm)
        if player_id == 0:
            return temp[:, :960], temp[:, 1922].reshape(-1, 1), temp[:, 1920].reshape(-1, 1) 
        else:
            return temp[:, 960:1920], temp[:, 1923].reshape(-1, 1), temp[:, 1921].reshape(-1, 1) 
        
    


def read_memmap(mem_file_name):
    with open(f"{mem_file_name}.conf", "r") as file:
        memmap_configs = json.load(file)    
        return np.memmap(mem_file_name, mode='r+', \
            shape=tuple(memmap_configs['shape']), \
  dtype=memmap_configs['dtype'])
        