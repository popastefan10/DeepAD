from torch.utils.data import ConcatDataset, random_split
from typing import Callable, TypeVar

from src.deep_ad.config import Config
from src.deep_ad.data.dagm_dataset import DAGMDataset, DAGMDatasetDev


DS = TypeVar("DS", DAGMDataset, DAGMDatasetDev)


# Returns train, val and test datasets for DAGM
# The datasets contain only defect-free images
# Images from each class are split as follows: 80% train, 10% val, 10% test
# Use constructor to specify which dataset class to use
def dagm_get_datasets(config: Config, constructor: type[DS]) -> tuple[DS, DS, DS]:
    # Create a dataset for each class
    class_datasets = [
        constructor(img_dir=config.DAGM_dir, classes=[cls], type="Defect-free") for cls in DAGMDataset.all_classes
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
