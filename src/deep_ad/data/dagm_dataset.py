import glob
import os
import re
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image


# Dataset class for DAGM 2007 dataset
class DAGMDataset(Dataset):
    def __init__(self, img_dir: str, transform=None, target_transform=None, classes: list[int] = None):
        self.classes: list[int] = list(range(1, 11)) if not classes else [*classes]
        self.img_dir = img_dir
        self.image_paths: list[str] = []
        for cls in self.classes:
            self.image_paths.extend(sorted(glob.glob(os.path.join(img_dir, f"Class{cls}", "Train", "*.png"))))

        label_paths: list[str] = []
        for cls in self.classes:
            label_paths.extend(glob.glob(os.path.join(img_dir, f"Class{cls}", "Train", "Label", "*_label.png")))
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
        return self.image_paths.index(os.path.join(self.img_dir, f"Class{cls}", "Train", f"{image_name}.PNG"))


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
