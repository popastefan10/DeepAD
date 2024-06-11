import glob
import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from typing import Callable, Literal

from src.deep_ad.data.dagm_utils import (
    dagm_get_class,
    dagm_get_image_name,
    dagm_get_label_name,
    dagm_get_image_key,
    dagm_get_label_key,
    dagm_get_image_path,
    dagm_get_patch_key,
)

DAGM_dataset_type = Literal["Original", "Defect-free", "Defect-only"]


# Dataset class for DAGM 2007 dataset
class DAGMDataset(Dataset):
    all_classes: list[int] = list(range(1, 11))

    # Returns all image and label paths for given classes and type
    @staticmethod
    def __get_images_and_labels_paths(
        img_dir: str, classes: list[int], type: DAGM_dataset_type, train: bool = True
    ) -> tuple[list[str], list[str], list[str]]:
        image_paths: list[str] = []
        label_paths: list[str] = []
        better_label_paths: list[str] = []
        for cls in classes:
            subset = "Train" if train else "Test"
            image_paths.extend(glob.glob(os.path.join(img_dir, f"Class{cls}", subset, "*.png")))
            label_paths.extend(glob.glob(os.path.join(img_dir, f"Class{cls}", subset, "Label", "*_label.png")))
            better_label_paths.extend(
                glob.glob(os.path.join(img_dir, f"Class{cls}", subset, "Label", "*_label_better.png"))
            )

        if type == "Defect-free":
            # Remove images with labels
            label_images_paths: list[str] = [
                dagm_get_image_path(img_dir, dagm_get_class(label_path), dagm_get_label_name(label_path))
                for label_path in label_paths
            ]
            image_paths = list(set(image_paths) - set(label_images_paths))
            label_paths = []
            better_label_paths = []
        elif type == "Defect-only":
            # Keep only images with labels
            label_images_paths: list[str] = [
                dagm_get_image_path(img_dir, dagm_get_class(label_path), dagm_get_label_name(label_path), train)
                for label_path in label_paths
            ]
            image_paths = list(set(label_images_paths))

        # Sort paths by class and image name
        sort_fn: Callable[[str], tuple[int, str]] = lambda path: (dagm_get_class(path), dagm_get_image_name(path))
        image_paths.sort(key=sort_fn)
        label_paths.sort(key=sort_fn)
        better_label_paths.sort(key=sort_fn)

        return image_paths, label_paths, better_label_paths

    def __init__(
        self,
        img_dir: str,
        transform=None,
        target_transform=None,
        classes: list[int] | None = None,
        type: DAGM_dataset_type = "Original",
        train: bool = True,
    ):
        self.classes: list[int] = DAGMDataset.all_classes if not classes else [*classes]
        self.img_dir = img_dir
        self.type = type

        image_paths, label_paths, better_label_paths = DAGMDataset.__get_images_and_labels_paths(
            img_dir, self.classes, type, train
        )
        self.image_paths: list[str] = image_paths
        self.label_paths: dict[str, str] = dict(
            [(dagm_get_label_key(label_path), label_path) for label_path in label_paths]
        )
        self.better_label_paths: dict[str, str] = dict(
            [(dagm_get_label_key(label_path), label_path) for label_path in better_label_paths]
        )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        image = read_image(image_path).squeeze()
        label_key = dagm_get_image_key(image_path)
        # Choose the better label if possible
        label_path = self.better_label_paths.get(label_key, None) or self.label_paths.get(label_key, None)
        label = read_image(label_path).squeeze() if label_key in self.label_paths else torch.zeros(image.shape)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_index_of_image(self, cls: int, image_name: str) -> int:
        return self.image_paths.index(dagm_get_image_path(self.img_dir, cls, image_name))


# Choose DAGMDataset constructor
def use_dagm() -> type[DAGMDataset]:
    return DAGMDataset


# Dev dataset returns the class and name of the image as well
class DAGMDatasetDev(DAGMDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        cls = dagm_get_class(self.image_paths[idx])
        name = dagm_get_image_name(self.image_paths[idx])
        image, label = super().__getitem__(idx)

        return image, label, cls, name


# Choose DAGMDatasetDev constructor
def use_dagm_dev() -> type[DAGMDatasetDev]:
    return DAGMDatasetDev


# Dataset class for patches obtained from DAGM 2007 dataset
class DAGMPatchDataset:
    def __init__(
        self,
        img_dir: str | None = None,
        classes: list[int] | None = None,
        patch_paths: list[str] | None = None,
        patch_classes: list[int] | None = None,
        transform=None,
        cache_patches: bool = False,
    ) -> None:
        """
        If you want to use the patches from the whole dataset (all classes), provide img_dir. \\
        If you want to use the patches from specific classes, provide img_dir and classes as a list of integers. \\
        If you want to use specific patches, as in train, val and test datasets, provide patch_paths and patch_classes.
        """
        if not img_dir and not patch_paths:
            raise ValueError("Either img_dir or patch_paths must be provided")
        if img_dir and patch_paths:
            raise ValueError("Only one of img_dir or patch_paths must be provided")
        if patch_paths and not patch_classes:
            raise ValueError("patch_classes must be provided if patch_paths are provided")

        if patch_paths:
            self.patch_paths = patch_paths
            self.patch_classes: list[int] = patch_classes
        else:
            self.classes: list[int] = DAGMDataset.all_classes if not classes else [*classes]
            self.img_dir = img_dir
            self.patch_paths: list[str] = []
            for cls in self.classes:
                self.patch_paths.extend(glob.glob(os.path.join(img_dir, f"Class{cls}", "Train", "*.png")))

        self.cache_patches: bool = cache_patches
        self.patches_cache: dict[str, tuple[Image.Image, str]] = {}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        patch_path = self.patch_paths[idx]
        if self.cache_patches and patch_path in self.patches_cache:
            image, key = self.patches_cache[patch_path]
        else:
            image = Image.open(patch_path)
            key = dagm_get_patch_key(patch_path)
            if self.cache_patches:
                self.patches_cache[patch_path] = (image, key)
        if self.transform:
            image = self.transform(image)

        return image, key
