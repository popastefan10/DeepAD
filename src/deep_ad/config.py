import os

from dotenv import dotenv_values
from torch import Generator


class Config:
    def __init__(self, env_path: str | None = None) -> None:
        # Paths
        env = dotenv_values(env_path)
        self.datasets_dir = env["datasets_dir"]
        self.DAGM_dir = os.path.join(self.datasets_dir, "DAGM")

        # Datasets
        self.dagm_lengths = [0.8, 0.1, 0.1]

        # PyTorch
        self.seed: int = 42
        self.generator = Generator().manual_seed(self.seed)
        self.batch_size: int = 32
