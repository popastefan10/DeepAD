import matplotlib
matplotlib.use("TkAgg")  # Faster rendering

from mpl_interactions import panhandler, zoom_factory

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# Shows an image using OpenCV
def show_image(title: str, image: np.ndarray):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# https://mpl-interactions.readthedocs.io/en/stable/examples/zoom-factory.html
# Plots three RGB images side by side: the original image, the label, and the original image with the label boundary
def show_image_with_label(image: np.ndarray, label: np.ndarray, title: str = "Image with label") -> None:
    label_dilated = cv.dilate(label, np.ones((3, 3), np.uint8), iterations=1)
    out_boundary = np.add(label_dilated, -label)
    out_boundary = cv.cvtColor(out_boundary, cv.COLOR_RGB2GRAY)

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
