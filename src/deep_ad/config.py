import os

from torch import Generator


class Config:
    def __init__(self) -> None:
        self.root_dir = "."

        # Datasets
        self.datasets_dir = "C:\\Stefan\\Facultate\\Licenta\\Datasets"
        self.DAGM_dir = os.path.join(self.datasets_dir, "DAGM")
        self.KSDD_dir = os.path.join(self.datasets_dir, "KSDD")
        self.KSDD2_dir = os.path.join(self.datasets_dir, "KSDD2")
        self.MVTecAD_dir = os.path.join(self.datasets_dir, "MVTecAD")
        self.Severstal_dir = os.path.join(self.datasets_dir, "Severstal")

        # PyTorch
        self.seed: int = 42
        self.generator = Generator().manual_seed(self.seed)
