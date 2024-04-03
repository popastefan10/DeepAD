import os

from dotenv import dotenv_values
from torch import Generator


class Config:
    def __init__(self, root_dir: str = ".", env_path: str | None = None) -> None:
        # Paths
        env = dotenv_values(env_path)
        self.root_dir = root_dir
        self.datasets_dir = env["datasets_dir"]
        self.DAGM_raw_dir = os.path.join(self.datasets_dir, "DAGM")
        self.DAGM_processed_dir = os.path.join(root_dir, "data", "processed", "DAGM")

        # Datasets
        self.dagm_lengths = [0.8, 0.1, 0.1] # Train, Val, Test
        self.raw_patch_size = 160           # Patches larger than 128 need to be cropped to avoid border effects when applying random transforms
        self.ppi = 4                        # Patches per image - Number of patches to extract from each image
        self.patches_iou_threshold = 0      # Maximum IOU between patches to consider them different

        # PyTorch
        self.seed: int = 42
        self.generator = Generator().manual_seed(self.seed)
        self.batch_size: int = 32
