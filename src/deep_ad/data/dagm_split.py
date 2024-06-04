import glob
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, random_split
from torchvision.transforms.v2._transform import Transform
from typing import TypeVar

from src.deep_ad.config import Config
from src.deep_ad.data.dagm_dataset import DAGMDataset, DAGMDatasetDev, DAGMPatchDataset
from src.deep_ad.data.dagm_utils import dagm_get_patches_dir


DS = TypeVar("DS", DAGMDataset, DAGMDatasetDev)


# Returns train, val and test datasets for DAGM
# The datasets contain only defect-free images
# Images from each class are split as follows: 80% train, 10% val, 10% test
# Use constructor to specify which dataset class to use
def dagm_get_splits(config: Config, constructor: type[DS]) -> tuple[DS, DS, DS]:
    # Create a dataset for each class
    class_datasets = [
        constructor(img_dir=config.DAGM_raw_dir, classes=[cls], type="Defect-free") for cls in DAGMDataset.all_classes
    ]

    # Split the datasets into training, validation, and test sets
    split_datasets = [
        random_split(ds, lengths=config.dagm_lengths, generator=config.generator) for ds in class_datasets
    ]

    # Concatenate class datasets
    split_datasets = list(zip(*split_datasets))
    train_dataset = ConcatDataset(split_datasets[0])
    val_dataset = ConcatDataset(split_datasets[1])
    test_dataset = ConcatDataset(split_datasets[2])

    return train_dataset, val_dataset, test_dataset


def dagm_patch_get_splits(
    config: Config,
    train_transform: Transform | None = None,
    val_transform: Transform | None = None,
    classes: list[int] | None = None,
    cache_patches: bool = False,
) -> tuple[DAGMPatchDataset, DAGMPatchDataset]:
    """
    Splits the patches into train and validation datasets. \\
    Uses `train_test_split` from `sklearn.model_selection` to split the patches. \\
    Train dataset will be obtained first, by splitting according to the first ratio from `config.dagm_lengths`. \\
    Then, the val and test datasets will be obtained by splitting the remaining paths according to the second ratio.
    """
    # First, get all the paths and classes
    classes = classes or list(range(1, 11))
    patches_dir = dagm_get_patches_dir(
        config,
        ppi=config.ppi,
        patch_size=config.raw_patch_size,
        pad=config.patches_pad,
        name=config.patches_dataset_name,
    )
    patches_cls_paths = [glob.glob(os.path.join(patches_dir, f"Class{cls}\\Train\\*.png")) for cls in classes]
    patches_paths: list[str] = []
    classes: list[int] = []
    for cls, cls_paths in enumerate(patches_cls_paths):
        patches_paths += cls_paths
        classes += [cls + 1] * len(cls_paths)

    # Split the paths and classes
    train_split, _ = config.dagm_lengths
    train_paths, val_paths, train_classes, val_classes = train_test_split(
        patches_paths, classes, test_size=1 - train_split, random_state=config.seed, stratify=classes
    )

    # Use transforms for training only
    train_dataset = DAGMPatchDataset(
        patch_paths=train_paths,
        patch_classes=train_classes,
        transform=train_transform,
        cache_patches=cache_patches,
    )
    val_dataset = DAGMPatchDataset(
        patch_paths=val_paths, patch_classes=val_classes, transform=val_transform, cache_patches=cache_patches
    )

    return train_dataset, val_dataset
