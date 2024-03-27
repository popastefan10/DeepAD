import matplotlib
matplotlib.use("TkAgg")  # Faster rendering

from mpl_interactions import panhandler, zoom_factory

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# Type alias for bounding boxes in the format (x1, y1, x2, y2)
TBBox = tuple[int, int, int, int]


# Shows an image using OpenCV
def show_image(title: str, image: np.ndarray):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


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
