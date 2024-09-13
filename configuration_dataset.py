from dataclasses import dataclass

@dataclass
class DatasetConfig:
    basepath: str = "/root/multi_purpose/slp_compact_data/processed_data"
    agent_name: str = "LINK"
    batch_size: int = 1024
    num_workers: int = 0
    seq_len: int = 31
    duplicate_ok: bool = True
    