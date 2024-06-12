import numpy as np
import torch

from scipy.ndimage import label as scipy_label
from torch.nn.functional import pad, sigmoid
from torcheval.metrics.functional import binary_auprc
from typing import Callable

from src.deep_ad.config import Config
from src.deep_ad.image import create_center_mask
from src.deep_ad.model import DeepCNN
from src.deep_ad.save_manager import SaveManager
from src.deep_ad.trainer import create_optimizer
from src.deep_ad.transforms import normalize_to_mean_std


def load_pretrained(config: Config, save_manager: SaveManager, run_name: str, checkpoint_name: str) -> DeepCNN:
    """Returns a pretrained model loaded from a checkpoint."""
    model = DeepCNN(config)
    optimizer = create_optimizer(model, config)
    model, _, _, _, _ = save_manager.load_checkpoint(
        model, optimizer, path=save_manager.get_checkpoint_path(config, run_name, checkpoint_name)
    )

    return model


def pad_image(config: Config, image: torch.Tensor, patch_size: int | None = None) -> tuple[torch.Tensor, int, int, int]:
    """
    Args:
    * `image`: a tensor with shape `[... x] C x H x W`
    * `patch_size`: the size of the patch that will be extracted from the image. If not provided, `config.patch_size`

    Returns:
        * `padded_image`: a tensor with shape `[... x] C x (H + 2 * pad_size) x (W + 2 * pad_size)`
        * `min_pad`: minimum required padding such that the center region of size `config.content_size` of the first
        patch of size `config.patch_size` will be positioned on `(0, 0)` coordinates of the original, unpadded image
        * `margin`: additional margin required for the rightmost patches
        * `pad_size`: total padding size (most probably equal to `min_pad + margin`)
    """
    min_pad = ((patch_size or config.patch_size) - config.content_size) // 2
    margin = (config.stride - ((image.shape[-1] - config.content_size) % config.stride)) % config.stride
    pad_size = min_pad + margin
    padded_image = pad(image, (pad_size, pad_size, pad_size, pad_size), mode="reflect").squeeze()

    return padded_image, min_pad, margin, pad_size


def cut_margins(image: torch.Tensor, margin: int) -> torch.Tensor:
    """
    Args:
    * `image`: a tensor with shape `[H x W]`
    * `width`: width of the margin to be removed

    Returns:
    * `image`: the input tensor modified in-place by cahnging all pixels within a margin of width=`width` to `0`.
    """
    if margin > 0:
        image[:margin, :], image[-margin:, :], image[:, :margin], image[:, -margin:] = 0, 0, 0, 0
    return image


def reconstruct_by_inpainting(config: Config, image: torch.Tensor, model: DeepCNN) -> torch.Tensor:
    """
    Args:
    * `image`: a tensor with shape `[... x] C x H x W`
    * `model`: the model used for inpainting

    Returns:
    * `inpainted_image`: a tensor with shape `H x W` representing the image reconstructed from content patches cropped from patches inpainted by a model
    """
    padded_image, _, margin, pad_size = pad_image(config, image.unsqueeze(0))
    image_length = image.shape[-1]
    num_windows = (image_length + 2 * pad_size - config.patch_size) // config.stride + 1

    inpainted_contents = []
    mask = torch.asarray(create_center_mask(image_size=config.patch_size, center_size=config.center_size))
    with torch.no_grad():
        for i in range(num_windows):
            for j in range(num_windows):
                # Crop input patch
                tli, bri = margin + i * config.stride, margin + i * config.stride + config.patch_size  # top-left
                tlj, brj = margin + j * config.stride, margin + j * config.stride + config.patch_size  # bottom-right
                patch = padded_image[tli:bri, tlj:brj]
                # Feed to model
                input = normalize_to_mean_std(
                    patch.unsqueeze(0).unsqueeze(0), mean=0.5267019737681685, std=0.19957033073362934
                )
                input = input * (1 - mask)
                input = input.to(config.device)
                output = model(input)
                output = normalize_to_mean_std(output, mean=image.mean(), std=image.std())
                # Crop content region
                tl, br = (config.patch_size - config.content_size) // 2, (config.patch_size + config.content_size) // 2
                inpainted_contents.append(output.detach().squeeze().cpu().numpy()[tl:br, tl:br])
    inpainted_contents = torch.asarray(np.array(inpainted_contents))

    image_size = image.shape[-1]
    inpainted_image = torch.zeros((image_size + margin, image_size + margin))
    weights = torch.zeros((image_size + margin, image_size + margin))
    for i in range(num_windows):
        for j in range(num_windows):
            tli, bri = i * config.stride, i * config.stride + config.content_size
            tlj, brj = j * config.stride, j * config.stride + config.content_size
            inpainted_image[tli:bri, tlj:brj] += torch.asarray(inpainted_contents[i * num_windows + j])
            weights[tli:bri, tlj:brj] += 1
    inpainted_image /= weights
    inpainted_image = inpainted_image[:image_size, :image_size]

    return inpainted_image


from src.deep_ad.image import plot_image_pixels


def compute_diff(image: torch.Tensor, inpainted_image: torch.Tensor) -> torch.Tensor:
    """
    Args:
    * `diff_image`: the difference between the original image and the inpainted image

    Processes the image such that the final pixels values will lie in the range `[0, 1]`. It also increases the gap
    between dark and bright pixels by moving brighter pixels closer `1` and darker pixels closer to `0`.
    """
    image_normed = normalize_to_mean_std(
        image.unsqueeze(0).unsqueeze(0), mean=0.5267019737681685, std=0.19957033073362934
    ).squeeze()
    inpainted_normed = normalize_to_mean_std(
        inpainted_image.unsqueeze(0).unsqueeze(0), mean=0.5267019737681685, std=0.19957033073362934
    ).squeeze()
    diff_image = torch.abs(image_normed - inpainted_normed)
    diff_01 = (diff_image - diff_image.min()) / (diff_image.max() - diff_image.min())
    weight = 1
    wdiff = weight * diff_01
    postproc = sigmoid(wdiff)
    diff_image = postproc

    # wdiff = weight * (diff_01 - diff_01.median())
    # wdiff = weight * (diff_01 - diff_01.mean())
    # wdiff = weight * (diff_01 - 0.4)
    # wdiff = weight * diff_01
    # postproc = sigmoid(wdiff)
    # postproc = torch.exp(wdiff)
    # postproc = wdiff
    # postproc = (postproc - postproc.min()) / (postproc.max() - postproc.min())
    # postproc = 1 - postproc
    # postproc = (postproc - postproc.min()) / (postproc.max() - postproc.min())

    return postproc


def compute_heatmap(
    image: torch.Tensor,
    hm_patch_size: int,
    hm_num_windows: int,
    patch_metric: Callable[[torch.Tensor], float],
) -> torch.Tensor:
    heatmap = torch.zeros_like(image)
    for row in range(hm_num_windows):
        for col in range(hm_num_windows):
            tlr, brr, tlc, brc = (
                row * hm_patch_size,
                (row + 1) * hm_patch_size,
                col * hm_patch_size,
                (col + 1) * hm_patch_size,
            )
            patch = image[tlr:brr, tlc:brc]
            heatmap[tlr:brr, tlc:brc] = patch_metric(patch)
    return heatmap


def binarize_heatmap(heatmap: torch.Tensor, threshold: float) -> torch.Tensor:
    hmc = heatmap.clone()
    hmc[hmc < threshold] = 0
    hmc[hmc >= threshold] = 1
    return hmc


def remove_islands(heatmap: torch.Tensor, heatmap_bin: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """
    Args:
    * `heatmap`: image with the heatmap
    * `heatmap_bin`: binarized `heatmap`

    Breaks the binary image into islands of pixels adjacent on row or column and keeps only the largest islands of
    pixels. The idea is that reconstruction noise is likely to be found isolated in the image, thus making up the
    smaller islands.

    Returns:
    * `image`: binary image with islands removed
    * `island_sizes`: sizes of remaining islands
    """
    structure = torch.ones((3, 3))
    labeled_image, num_features = scipy_label(heatmap_bin.cpu().numpy(), structure=structure)
    if num_features == 0:
        return heatmap_bin, [0], [0], [0], [0]
    sizes = np.bincount(labeled_image.flatten())
    sizes[0] = 0  # Discard background denoted by 0
    sorted_islands = np.argsort(sizes)

    max_values = np.zeros_like(sizes, dtype=np.float32)
    for i in range(1, num_features + 1):
        max_values[i] = heatmap[labeled_image == i].max()
    min_values = np.zeros_like(sizes, dtype=np.float32)
    for i in range(1, num_features + 1):
        min_values[i] = heatmap[labeled_image == i].min()
    mean_values = np.zeros_like(sizes, dtype=np.float32)
    for i in range(1, num_features + 1):
        mean_values[i] = heatmap[labeled_image == i].mean()

    max_pixel = heatmap.max().item()
    second_max_pixel = heatmap[heatmap < max_pixel].max().item()
    third_max_pixel = heatmap[heatmap < second_max_pixel].max().item()
    keep_labels = sorted_islands[max_values[sorted_islands] > third_max_pixel]

    mask = torch.zeros_like(heatmap_bin)
    for label in keep_labels:
        mask = torch.logical_or(mask, torch.asarray(labeled_image == label))
    return (
        heatmap_bin * mask,
        sizes[keep_labels],
        max_values[keep_labels],
        min_values[keep_labels],
        mean_values[keep_labels],
    )


def compute_anomaly_heatmap(
    config: Config, image: torch.Tensor, inpainted_image: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Args:
    * `image`: a tensor with shape `H x W`
    * `inpainted_image`: a tensor with shape `H x W`

    Returns:
    * `heatmap`: a tensor with shape `H x W` representing the anomaly heatmap. This should be ready for calculating the AUPRC directly on it
    """
    diff_image = compute_diff(image, inpainted_image, norm=True)

    # Compute the initial heatmap
    patch_surface = config.hm_patch_size**2
    hm_num_windows = (diff_image.shape[-1] - config.hm_patch_size) // config.hm_patch_size + 1
    diff_cut = cut_margins(diff_image.clone(), margin=2)
    patch_metric = lambda patch: patch.max()
    heatmap = compute_heatmap(diff_cut, config.hm_patch_size, hm_num_windows, patch_metric)

    # Binarize heatmap
    hmb = binarize_heatmap(heatmap, threshold=threshold)
    hmr, _, _, _, _ = remove_islands(heatmap_bin=hmb, heatmap=heatmap)

    return diff_image * hmr


def compute_anomaly_heatmap_adaptive(
    config: Config, image: torch.Tensor, inpainted_image: torch.Tensor
) -> tuple[torch.Tensor, float, list[float]]:
    """
    Args:
    * `image`: a tensor with shape `H x W`
    * `inpainted_image`: a tensor with shape `H x W`

    Returns:
    * `heatmap`: a tensor with shape `H x W` representing the anomaly heatmap. This should be ready for calculating the AUPRC directly on it
    * `recommended_threshold`: the threshold that should be used for binarizing the heatmap
    * `recommended_thresholds`: a list of thresholds that could be used for binarizing the heatmap, including the recommended one
    """
    diff_image = compute_diff(image, inpainted_image)

    # Compute the initial heatmap
    patch_surface = config.hm_patch_size**2
    hm_num_windows = (diff_image.shape[-1] - config.hm_patch_size) // config.hm_patch_size + 1
    diff_cut = cut_margins(diff_image.clone(), margin=2)
    patch_metric = lambda patch: patch.max()
    heatmap = compute_heatmap(diff_cut, config.hm_patch_size, hm_num_windows, patch_metric)

    # Try different thresholds
    thresholds = []
    sizes_mat = []
    max_values_mat = []
    min_values_mat = []
    mean_values_mat = []
    for threshold in np.arange(0.0, 1.0, 0.01):
        hmb = binarize_heatmap(heatmap, threshold=threshold)
        hmr, sizes, max_values, min_values, mean_values = remove_islands(heatmap_bin=hmb, heatmap=heatmap)
        sizes_mat.append(np.array(sizes))
        max_values_mat.append(np.array(max_values))
        min_values_mat.append(np.array(min_values))
        mean_values_mat.append(np.array(mean_values))
        thresholds.append(threshold)

    max_values = np.array([max(values) for values in max_values_mat])
    min_values = np.array([min(values) for values in min_values_mat])
    mean_values = np.array([np.mean(sizes * mean_values) for sizes, mean_values in zip(sizes_mat, mean_values_mat)])

    # Threshold recommendation
    max_sizes = np.array([max(sizes) for sizes in sizes_mat])
    max_sizes_diff = -np.diff(max_sizes)
    diff_over_max = max_sizes_diff / max_sizes[:-1]
    thresholds = np.array(thresholds)

    # Remove threholds that have max island greater than 50% of the image
    keep_thresholds = max_sizes / patch_surface / 4096 < 0.5
    # Remove thresholds that don't have islands
    keep_thresholds = np.logical_and(keep_thresholds, max_sizes > 0)
    # Remove thresholds that have a high difference in the max size
    keep_thresholds[:-1] = np.logical_and(keep_thresholds[:-1], diff_over_max < 0.5)
    # Now, recommend the first threshold
    recommended_threshold = thresholds[keep_thresholds][0]
    recommended_size = max_sizes[keep_thresholds][0]
    # But, keep all the thresholds which have the max size at least half of the size corresponding to the recommended threshold
    keep_thresholds = np.logical_and(keep_thresholds, max_sizes >= recommended_size * 0.5)
    recommended_thresholds = thresholds[keep_thresholds]

    hmb = binarize_heatmap(heatmap, threshold=recommended_threshold)
    hmr, sizes, max_values, min_values, mean_values = remove_islands(heatmap_bin=hmb, heatmap=heatmap)

    return diff_image * hmr, recommended_threshold, recommended_thresholds
