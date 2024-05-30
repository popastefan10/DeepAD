import numpy as np
import torch

from scipy.ndimage import label as scipy_label
from torch.nn.functional import pad, sigmoid
from typing import Callable

from src.deep_ad.config import Config
from src.deep_ad.image import create_center_mask
from src.deep_ad.model import DeepCNN
from src.deep_ad.save_manager import SaveManager
from src.deep_ad.trainer import create_optimizer, normalize_to_mean_std


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


def diff_postprocessing(diff_image: torch.Tensor) -> torch.Tensor:
    """
    Args:
    * `diff_image`: the difference between the original image and the inpainted image

    Processes the image such that the final pixels values will lie in the range `[0, 1]`. It also increases the gap
    between dark and bright pixels by moving brighter pixels closer `1` and darker pixels closer to `0`.
    """
    weight = 1
    postproc = sigmoid(weight * ((diff_image - diff_image.min()) / (diff_image.max() - diff_image.min()) - 0.5))
    postproc = (postproc - postproc.min()) / (postproc.max() - postproc.min())
    postproc = 1 - postproc
    postproc = (postproc - postproc.min()) / (postproc.max() - postproc.min())

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


def remove_islands(image: torch.Tensor) -> torch.Tensor:
    """
    Args:
    * `image`: binary image

    Breaks the binary image into islands of pixels adjacent on row or column and keeps only the largest islands of
    pixels. The idea is that reconstruction noise is likely to be found isolated in the image, thus making up the
    smaller islands.
    """
    labeled_image, num_features = scipy_label(image.cpu().numpy())
    if num_features == 0:
        return image
    sizes = np.bincount(labeled_image.flatten())
    sizes[0] = 0  # Discard background denoted by 0
    sorted_islands = np.argsort(sizes)
    biggest_island = sizes[sorted_islands[-1]]
    keep_labels = sorted_islands[sizes[sorted_islands] >= int(biggest_island * 0.7)]
    mask = torch.zeros_like(image)
    for label in keep_labels:
        mask = torch.logical_or(mask, torch.asarray(labeled_image == label))
    return image * mask
