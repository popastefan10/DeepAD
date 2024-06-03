import os
import re

from src.deep_ad.config import Config


# Returns class number from path
def dagm_get_class(path: str) -> int:
    return int(re.search(r"Class(\d+)", path).group(1))


# Returns image name from path
def dagm_get_image_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# Returns label name from path
def dagm_get_label_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].split("_")[0]


def dagm_build_key(image_class: str, image_name: str) -> str:
    return image_class + "_" + image_name


# Returns image key from path in format <class>_<image_name>
def dagm_get_image_key(path: str) -> str:
    return dagm_build_key(str(dagm_get_class(path)), dagm_get_image_name(path))


# Returns label key from path in format <class>_<label_name>
def dagm_get_label_key(path: str) -> str:
    return str(dagm_get_class(path)) + "_" + dagm_get_label_name(path)


# Returns patch key from path in format <class>_<image_name>_<patch_id>
def dagm_get_patch_key(path: str) -> str:
    return str(dagm_get_class(path)) + "_" + "_".join(os.path.basename(path).split("_")[:2])


# Returns image path from class and image name (which consists of 4 digits, e.g. 0587, 1183)
def dagm_get_image_path(dir: str, cls: int, image_name: str, train: bool = True) -> str:
    subset = "Train" if train else "Test"
    return os.path.join(dir, f"Class{cls}", subset, f"{image_name}.PNG")


# Returns patches directory path based on ppi and patch size
def dagm_get_patches_dir(config: Config, ppi: int, patch_size: int, pad: bool = False, name: str | None = None) -> str:
    return os.path.join(
        config.DAGM_processed_dir, f"{ppi}ppi_{patch_size}px{'_pad' if pad else ''}{f'_{name}' if name else ''}"
    )
