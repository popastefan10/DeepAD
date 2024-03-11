import os
import re


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
