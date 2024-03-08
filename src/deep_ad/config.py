import os


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

def hello():
    print('hello from config.py..')


__all__ = ['Config', 'Test', 'hello']