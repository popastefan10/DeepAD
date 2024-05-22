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


def create_to_image_transform() -> Transform:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def normalize_to_mean_std(tensor: torch.Tensor, mean: torch.tensor, std: torch.tensor) -> torch.Tensor:
    """
    Normalize a tensor to a given mean and standard deviation.
    The tensor is assumed to have shape `(B, C, H, W)`.
    """
    output = tensor - tensor.mean(dim=(0, 2, 3), keepdim=True)
    output = output / output.std(dim=(0, 2, 3), keepdim=True)
    output = output * std + mean

    return output
