import torch

from torch.nn.functional import pad

from src.deep_ad.config import Config
from src.deep_ad.model import DeepCNN
from src.deep_ad.save_manager import SaveManager
from src.deep_ad.trainer import create_optimizer


def load_pretrained(config: Config, save_manager: SaveManager, run_name: str, checkpoint_name: str) -> DeepCNN:
    """Returns a pretrained model loaded from a checkpoint."""
    model = DeepCNN(config)
    optimizer = create_optimizer(model, config)
    model, _, _, _, _ = save_manager.load_checkpoint(
        model, optimizer, path=save_manager.get_checkpoint_path(config, run_name, checkpoint_name)
    )

    return model


def pad_image(config: Config, image: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    """
    `image`: a tensor with shape `[... x] C x H x W`

    Returns:
        * `padded_image`: a tensor with shape `[... x] C x (H + 2 * pad_size) x (W + 2 * pad_size)`
        * `min_pad`: minimum required padding such that the center region of size `config.content_size` of the first
        patch of size `config.patch_size` will be positioned on `(0, 0)` coordinates of the original, unpadded image
        * `margin`: additional margin required for the rightmost patches
        * `pad_size`: total padding size (most probably equal to `min_pad + margin`)
    """
    min_pad = (config.patch_size - config.content_size) // 2
    margin = (config.stride - ((image.shape[-1] - config.content_size) % config.stride)) % config.stride
    pad_size = min_pad + margin
    padded_image = pad(image, (pad_size, pad_size, pad_size, pad_size), mode="reflect").squeeze()

    return padded_image, min_pad, margin, pad_size
