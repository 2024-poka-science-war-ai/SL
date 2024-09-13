from dataclasses import dataclass

@dataclass
class DatasetConfig:
    basepath: str = "/root/code/processed_data/"
    agent_name: str = "LINK"
    batch_size: int = 64
    num_workers: int = 0
    seq_len: int = 31
    