import os
import torch

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
        self.dagm_lengths = [0.8, 0.1, 0.1]  # Train, Val, Test
        self.raw_patch_size = (
            176  # Patches larger than 128 need to be cropped to avoid border effects when applying random transforms
        )
        self.patch_size = 128  # Patch size specified in the paper; keep in sync with loss_N
        self.ppi = 4  # Patches per image - Number of patches to extract from each image
        self.patches_iou_threshold = 0.05  # Maximum IOU between patches to consider them different

        # PyTorch
        self.seed: int = 42
        self.generator = Generator().manual_seed(self.seed)
        self.batch_size: int = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Training
        self.loss_Lambda = 0.9
        self.loss_N = 128 ** 2 # Keep in sync with patch_size
        self.optim_lr = 2e-4
        self.optim_adam_betas = (0.9, 0.999)
        self.optim_adam_eps = 1e-8
        self.train_epochs = 1
