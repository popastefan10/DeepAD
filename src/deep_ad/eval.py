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
