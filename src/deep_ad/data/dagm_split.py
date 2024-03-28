from torch.utils.data import ConcatDataset, random_split
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


# Returns train, val and test datasets for DAGM patches dataset
# The datasets contain patches obtained from defect-free images
# Images from each class are split as follows: 80% train, 10% val, 10% test
def dagm_patch_get_splits(config: Config) -> tuple[DAGMPatchDataset, DAGMPatchDataset, DAGMPatchDataset]:
    # Create a dataset for each class
    patches_dir = dagm_get_patches_dir(config, ppi=config.patches_per_image, patch_size=config.raw_patch_size)
    class_datasets = [DAGMPatchDataset(img_dir=patches_dir, classes=[cls]) for cls in DAGMDataset.all_classes]

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
