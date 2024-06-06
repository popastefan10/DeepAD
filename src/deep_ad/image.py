import matplotlib
from src.deep_ad.config import is_running_on_runpod

try:
    backend = "Agg" if is_running_on_runpod() else "TkAgg"
    matplotlib.use(backend)
except ImportError:
    print(f"Could not set matplotlib backend to '{backend}'!")

if not is_running_on_runpod():
    try:
        from mpl_interactions import panhandler, zoom_factory
    except ImportError:
        print("Could not import mpl_interactions!")

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from typing import Literal


# Type alias for bounding boxes in the format (x1, y1, x2, y2)
TBBox = tuple[int, int, int, int]


# Shows an image using OpenCV
def show_image(title: str, image: np.ndarray):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def plot_images(
    images: list[np.ndarray],
    titles: list[str],
    rows: int = 1,
    cols: int = 1,
    figsize: tuple[int, int] = (5, 5),
    title: str | None = None,
    range: Literal["auto", "01", "255", "minmax"] = "auto",
    cmap: str = 'gray',
    show: bool = True,
) -> None:
    """
    Plots a list of images with their respective titles in a grid layout.
    Images must be in a grayscale (shape HxW) or RGB (shape HxWx3) format.
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
    for ax, image, img_title in zip(axes, images, titles):
        vmin, vmax = None, None
        if range == "auto":
            vmin, vmax = (0, 1) if image.max() - image.min() > 0.1 else (image.min(), image.max())
        elif range == "01":
            vmin, vmax = 0, 1
        elif range == "255":
            vmin, vmax = 0, 255
        elif range == "minmax":
            vmin, vmax = image.min(), image.max()
        else:
            raise ValueError(f"Invalid range: {range}")

        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax) if len(image.shape) == 2 else ax.imshow(image)
        ax.set_title(img_title)
        ax.axis("off")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()


# https://mpl-interactions.readthedocs.io/en/stable/examples/zoom-factory.html
# Plots three images side by side: the original image, the label, and the original image with the label boundary
# The image and label must be RGB or gray
def show_image_with_label(image: np.ndarray, label: np.ndarray, title: str = "Image with label") -> None:
    if image.shape[-1] != 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    if label.shape[-1] != 3:
        label = cv.cvtColor(label, cv.COLOR_GRAY2RGB)

    label_dilated = cv.dilate(label, np.ones((3, 3), np.uint8), iterations=1)
    out_boundary = np.add(label_dilated, -label)
    out_boundary: np.ndarray = cv.cvtColor(out_boundary, cv.COLOR_RGB2GRAY)

    # create copy of image with out_boundary as red color
    image_with_out_boundary = image.copy()
    image_with_out_boundary[out_boundary > 0] = [255, 0, 0]

    # Plot setup
    with plt.ioff():
        fig = plt.figure(figsize=(10, 4))
        ax0 = plt.subplot(1, 3, 1)
        ax1 = plt.subplot(1, 3, 2, sharex=ax0, sharey=ax0)
        ax2 = plt.subplot(1, 3, 3, sharex=ax0, sharey=ax0)
        fig.suptitle(title)
        ax0.set_title("Original image")
        ax1.set_title("Label")
        ax2.set_title("Image with label boundary")

    # Plot the images
    ax0.imshow(image)
    ax1.imshow(label)
    ax2.imshow(image_with_out_boundary)

    # Add zooming and panning
    panhandler(fig)
    zoom_factory(ax0)
    zoom_factory(ax1)
    zoom_factory(ax2)
    plt.tight_layout()
    plt.show()


# Plots an image with its bounding boxes
def show_image_with_bboxes(image: np.ndarray, bboxes: list[TBBox], title: str = "Image with bounding boxes") -> None:
    if image.shape[-1] != 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    # Plot setup
    with plt.ioff():
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(2, 1, 1)
        ax.imshow(image)
        ax.set_title(title)

    # Plot the bounding boxes
    for i, bbox in enumerate(bboxes):
        rect = plt.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], f"{i}", color="r")

        ax_bbox = plt.subplot(4, len(bboxes), i + 2 * len(bboxes) + 1)
        ax_bbox.imshow(image[bbox[1] : bbox[3], bbox[0] : bbox[2]])
        ax_bbox.set_title(f"Box {i}")

    # Add zooming and panning
    panhandler(fig)
    zoom_factory(ax)
    plt.tight_layout()
    plt.show()


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    title: str,
    save_path: str | None = None,
    show_plot: bool = True,
) -> None:
    """
    Plots the training and validation losses over epochs.
    """
    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1000)
    if show_plot:
        plt.show()


# Computes intersection over union between two bounding boxes
# Bounding boxes are represented as tuples (x1, y1, x2, y2)
def intersection_over_union(bbox_a: TBBox, bbox_b: TBBox, epsilon: float = 1e-5) -> float:
    inter_tl_x = max(bbox_a[0], bbox_b[0])
    inter_tl_y = max(bbox_a[1], bbox_b[1])
    inter_br_x = min(bbox_a[2], bbox_b[2])
    inter_br_y = min(bbox_a[3], bbox_b[3])
    inter_area = max(0, inter_br_x - inter_tl_x + 1) * max(0, inter_br_y - inter_tl_y + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
    union_area = box_a_area + box_b_area - inter_area

    iou = inter_area / (union_area + epsilon)

    return iou


def create_center_mask(image_size: int = 128, center_size: int = 32) -> np.ndarray:
    """
    Creates a binary mask for the center patch of an image. The mask will contain 1 in the center square and 0 in the
    surrounding pixels.

    Args:
        `image_size` - The size of the image.
        `center_size` - The size of the center patch.

    Returns:
        A binary mask with shape (`image_size`, `image_size`). The returned mask will have type `np.float32`.
    """
    mask: np.ndarray = np.zeros((image_size, image_size), dtype=np.float32)
    offset = (image_size - center_size) // 2
    mask[offset : offset + center_size, offset : offset + center_size] = 1.0

    return mask

# https://www.sergilehkyi.com/uk/2019/10/image-segmentation-with-python/
def heatmap_overlay(
    image: np.ndarray, mask: np.ndarray, alpha: float, colormap: int = cv.COLORMAP_CIVIDIS
) -> np.ndarray:
    """
    Args:
    * `image`: `(H, W, C)` image
    * `mask`: `(H, W)` mask
    * `alpha`: float in [0, 1] for the transparency of the mask
    """
    overlay = cv.applyColorMap((mask * 255).astype(np.uint8), colormap=colormap)
    overlay[mask == 0] = [0, 0, 0]
    overlay = cv.cvtColor(overlay, cv.COLOR_BGR2RGB)
    overlay = cv.resize(overlay, (image.shape[1], image.shape[0]))
    return cv.addWeighted(image, 1 - alpha, overlay, alpha, 0)
