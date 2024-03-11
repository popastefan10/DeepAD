import glob
import os
import re
from typing import Callable, Literal
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image

DAGM_dataset_type = Literal["Original", "Defect-free", "Defect-only"]


# Dataset class for DAGM 2007 dataset
class DAGMDataset(Dataset):
    # Returns image and label paths for given classes and type
    @staticmethod
    def __get_images_and_labels_paths(
        img_dir: str, classes: list[int], type: DAGM_dataset_type
    ) -> tuple[list[str], list[str]]:
        image_paths: list[str] = []
        label_paths: list[str] = []
        for cls in classes:
            image_paths.extend(glob.glob(os.path.join(img_dir, f"Class{cls}", "Train", "*.png")))
            label_paths.extend(glob.glob(os.path.join(img_dir, f"Class{cls}", "Train", "Label", "*_label.png")))

        if type == "Defect-free":
            # Remove images with labels
            label_images_paths: list[str] = [
                dagm_get_image_path(img_dir, dagm_get_class(label_path), dagm_get_label_name(label_path))
                for label_path in label_paths
            ]
            image_paths = list(set(image_paths) - set(label_images_paths))
            label_paths = []
        elif type == "Defect-only":
            # Keep only images with labels
            label_images_paths: list[str] = [
                dagm_get_image_path(img_dir, dagm_get_class(label_path), dagm_get_label_name(label_path))
                for label_path in label_paths
            ]
            image_paths = list(set(label_images_paths))

        # Sort paths by class and image name
        sort_fn: Callable[[str], tuple[str, str]] = lambda path: (dagm_get_class(path), dagm_get_image_name(path))
        image_paths.sort(key=sort_fn)
        label_paths.sort(key=sort_fn)

        return image_paths, label_paths

    def __init__(
        self,
        img_dir: str,
        transform=None,
        target_transform=None,
        classes: list[int] = None,
        type: DAGM_dataset_type = "Original",
    ):
        self.classes: list[int] = list(range(1, 11)) if not classes else [*classes]
        self.img_dir = img_dir
        self.type = type

        image_paths, label_paths = DAGMDataset.__get_images_and_labels_paths(img_dir, self.classes, type)
        self.image_paths: list[str] = image_paths
        self.label_paths: dict[str, str] = dict(
            [(dagm_get_label_key(label_path), label_path) for label_path in label_paths]
        )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        image = read_image(image_path)
        label_key = dagm_get_image_key(image_path)
        label = read_image(self.label_paths[label_key]) if label_key in self.label_paths else torch.zeros(image.shape)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_index_of_image(self, cls: int, image_name: str) -> int:
        return self.image_paths.index(dagm_get_image_path(self.img_dir, cls, image_name))


# TODO - move these functions to a separate file, dagm.py


# Returns class number from path
def dagm_get_class(path: str) -> str:
    return re.search(r"Class(\d+)", path).group(1)


# Returns image name from path
def dagm_get_image_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# Returns label name from path
def dagm_get_label_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].strip("_label")


# Returns image key from path in format <class>_<image_name>
def dagm_get_image_key(path: str) -> str:
    return dagm_get_class(path) + "_" + dagm_get_image_name(path)


# Returns label key from path in format <class>_<label_name>
def dagm_get_label_key(path: str) -> str:
    return dagm_get_class(path) + "_" + dagm_get_label_name(path)


# Returns image path from class and image name (which consists of 4 digits, e.g. 0587, 1183)
def dagm_get_image_path(dir: str, cls: int, image_name: str) -> str:
    return os.path.join(dir, f"Class{cls}", "Train", f"{image_name}.PNG")
