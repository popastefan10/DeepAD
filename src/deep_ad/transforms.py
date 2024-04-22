import torch

from torchvision.transforms import v2
from torchvision.transforms.v2._transform import Transform

from src.deep_ad.config import Config


def create_training_transform(config: Config) -> Transform:
    return v2.Compose(
        [
            v2.RandomAffine(degrees=15, shear=(-15, 15, -15, 15), scale=(1, 1.1)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.CenterCrop(config.patch_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def create_validation_transform(config: Config) -> Transform:
    return v2.Compose(
        [
            v2.CenterCrop(config.patch_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
