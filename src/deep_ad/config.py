import os
import torch
import yaml

from dotenv import dotenv_values
from torch import Generator
from typing import Literal


class Config:
    def __init__(self, root_dir: str = ".", env_path: str | None = None, config_path: str | None = None) -> None:
        # Paths
        env = dotenv_values(env_path)
        self.root_dir = root_dir
        self.datasets_dir = env["datasets_dir"]
        self.DAGM_raw_dir = os.path.join(self.datasets_dir, "DAGM")
        self.DAGM_processed_dir = os.path.join(root_dir, "data", "processed", "DAGM")

        # Datasets
        self.dagm_lengths = [0.8, 0.1, 0.1]  # Train, Val, Test
        # Patches larger than 128 need to be cropped to avoid border effects when applying random transforms
        self.raw_patch_size = 176
        self.patch_size = 128  # Patch size specified in the paper; keep in sync with loss_N
        self.ppi = 4  # Patches per image - Number of patches to extract from each image
        self.patches_iou_threshold = 0.05  # Maximum IOU between patches to consider them different

        # PyTorch
        self.seed: int = 42
        self.generator = Generator().manual_seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model
        self.batch_norm: bool = True
        self.init_weights: bool = True

        # Training
        self.batch_size = 32
        self.loss_type: Literal["l1_norm", "l1_loss"] = "l1_norm"
        self.loss_Lambda = 0.9
        self.loss_N = 128**2  # Keep in sync with patch_size
        self.optim_lr = 2e-4
        self.optim_adam_betas = (0.9, 0.999)
        self.optim_adam_eps = 1e-8
        self.train_epochs = 1
        self.train_classes = [10]

        # Save
        self.save_dir = os.path.join(root_dir, "save")

        # Load predefined config
        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            yml_config = yaml.safe_load(f) or {}

        # Datasets
        self.dagm_lengths = yml_config.get("dagm_lengths") or self.dagm_lengths
        self.raw_patch_size = yml_config.get("raw_patch_size") or self.raw_patch_size
        self.patch_size = yml_config.get("patch_size") or self.patch_size
        self.ppi = yml_config.get("ppi") or self.ppi
        self.patches_iou_threshold = yml_config.get("patches_iou_threshold") or self.patches_iou_threshold

        # PyTorch
        self.seed = yml_config.get("seed") or self.seed
        self.generator = Generator().manual_seed(self.seed)

        # Model
        self.batch_norm = yml_config.get("batch_norm") or self.batch_norm
        self.init_weights = yml_config.get("init_weights") or self.init_weights

        # Training
        self.batch_size = yml_config.get("batch_size") or self.batch_size
        self.device = yml_config.get("device") or self.device
        self.loss_type = yml_config.get("loss_type") or self.loss_type
        self.loss_Lambda = yml_config.get("loss_Lambda") or self.loss_Lambda
        self.loss_N = yml_config.get("loss_N") or self.loss_N
        self.loss_N = eval(self.loss_N) if isinstance(self.loss_N, str) else self.loss_N
        self.optim_lr = yml_config.get("optim_lr") or self.optim_lr
        self.optim_lr = eval(self.optim_lr) if isinstance(self.optim_lr, str) else self.optim_lr
        self.optim_adam_betas = tuple(yml_config.get("optim_adam_betas")) or self.optim_adam_betas
        self.optim_adam_eps = yml_config.get("optim_adam_eps") or self.optim_adam_eps
        self.optim_adam_eps = eval(self.optim_adam_eps) if isinstance(self.optim_adam_eps, str) else self.optim_adam_eps
        self.train_epochs = yml_config.get("train_epochs") or self.train_epochs

    def save(self, config_path: str) -> None:
        """
        `config_path` - Path to save the configuration file, e.g. "save/config.yml".
        """
        with open(config_path, "w") as f:
            yaml.safe_dump(self.__getstate__(), f, default_flow_style=False)

    def __str__(self) -> str:
        return (
            "Datasets:"
            + f"\ndagm_lengths: {self.dagm_lengths}"
            + f"\nraw_patch_size: {self.raw_patch_size}"
            + f"\npatch_size: {self.patch_size}"
            + f"\nppi: {self.ppi}"
            + f"\npatches_iou_threshold: {self.patches_iou_threshold}"
            + "\n\nPyTorch:"
            + f"\nseed: {self.seed}"
            + f"\ndevice: {self.device}"
            + "\n\nModel:"
            + f"\nbatch_norm: {self.batch_norm}"
            + f"\ninit_weights: {self.init_weights}"
            + "\n\nTraining:"
            + f"\nbatch_size: {self.batch_size}"
            + f"\nloss_type: {self.loss_type}"
            + f"\nloss_Lambda: {self.loss_Lambda}"
            + f"\nloss_N: {self.loss_N}"
            + f"\noptim_lr: {self.optim_lr}"
            + f"\noptim_adam_betas: {self.optim_adam_betas}"
            + f"\noptim_adam_eps: {self.optim_adam_eps}"
            + f"\ntrain_epochs: {self.train_epochs}"
            + f"\ntrain_classes: {self.train_classes}"
        )

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        del (
            state["root_dir"],
            state["datasets_dir"],
            state["DAGM_raw_dir"],
            state["DAGM_processed_dir"],
            state["generator"],
            state["save_dir"],
        )

        return state
