from dataclasses import dataclass, field
import os
@dataclass
class DatasetConfig:
    """Configuration class for the SlippiDataset.
    """

    # Input / output structure.
    dataset_basepath: str = "/root/data/skyview/mm"
    seq_len: int = 64
    
    def __post_init__(self):
        """Input validation (not exhaustive)."""
        assert os.path.exists(self.dataset_basepath)
        pass