import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

from src.deep_ad.config import Config


class SaveManager:
    def __init__(self, config: Config) -> None:
        self.save_dir = config.save_dir
        self.checkpoints_dir = self._get_checkpoints_dir(self.save_dir)
        self.plots_dir = self._get_plots_dir(self.save_dir)
        self.inpaintings_dir = self._get_inpaintings_dir(self.save_dir)

    @staticmethod
    def _get_checkpoints_dir(save_dir: str) -> str:
        return os.path.join(save_dir, "checkpoints")

    @staticmethod
    def _get_plots_dir(save_dir: str) -> str:
        return os.path.join(save_dir, "plots")

    def get_checkpoint_path(self, run_name: str, checkpoint_name: str) -> str:
        return os.path.join(self.checkpoints_dir, "checkpoints", run_name, f"{checkpoint_name}.pt")

    @staticmethod
    def get_checkpoint_path(config: Config, run_name: str, checkpoint_name: str) -> str:
        return os.path.join(SaveManager._get_checkpoints_dir(config.save_dir), run_name, f"{checkpoint_name}.pt")

    @staticmethod
    def get_config_path(save_dir: str, run_name: str) -> str:
        return os.path.join(SaveManager._get_checkpoints_dir(save_dir), run_name, "config.yml")

    @staticmethod
    def _get_inpaintings_dir(save_dir: str) -> str:
        return os.path.join(save_dir, "inpaintings")

    @staticmethod
    def get_detections_dir(save_dir: str, run_name: str, checkpoint_name: str) -> str:
        return os.path.join(save_dir, "detections", run_name, checkpoint_name)

    def get_inpaintings_dir(self, run_name: str, checkpoint_name: str) -> str:
        return os.path.join(self.inpaintings_dir, run_name, checkpoint_name)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_losses: list[float],
        val_losses: list[float],
        epoch: int,
        run_name: str,
        name: str,
    ) -> None:
        """Saves model and optimizer state dicts along with training and validation losses in a checkpoint file."""
        save_dir = os.path.join(self.checkpoints_dir, run_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{name}.pt")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "epoch": epoch,
            },
            save_path,
        )
        print(f"Checkpoint saved at '{save_path}'")

    @staticmethod
    def load_checkpoint(
        model: nn.Module, optimizer: torch.optim.Optimizer, path: str
    ) -> tuple[nn.Module, torch.optim.Optimizer, list[float], list[float], int]:
        """
        Loads model and optimizer state dicts along with training and validation losses from a checkpoint file.
        Returns:
            - pretrained model,
            - loaded optimizer
            - training losses
            - validation losses
            - epoch
        """
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        print(f"Checkpoint loaded from '{path}'.")

        return model, optimizer, train_losses, val_losses, epoch

    def save_plot(self, run_name: str, plot_name: str) -> None:
        plot_dir = os.path.join(self.plots_dir, run_name)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir, plot_name))

    def save_config(self, config: Config, run_name: str) -> None:
        save_dir = os.path.join(self.checkpoints_dir, run_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "config.yml")
        config.save(save_path)

    def save_inpainting(
        self, inpainted_image: torch.Tensor, image_key: str, run_name: str, checkpoint_name: str
    ) -> None:
        save_dir = self.get_inpaintings_dir(run_name, checkpoint_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{image_key}.pt")
        torch.save(inpainted_image, save_path)

    def try_load_inpainting(self, image_key: str, run_name: str, checkpoint_name: str) -> torch.Tensor | None:
        """Image name should not contain any extensions"""
        load_path = os.path.join(self.get_inpaintings_dir(run_name, checkpoint_name), f"{image_key}.pt")
        inpainted_image = None
        if os.path.exists(load_path):
            inpainted_image = torch.load(load_path)

        return inpainted_image

    def save_detections(self, detections: list[torch.Tensor], run_name: str, checkpoint_name: str) -> None:
        save_dir = self.get_detections_dir(run_name, checkpoint_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        detections = torch.cat(detections, dim=0)
        save_path = os.path.join(save_dir, "detections.pt")
        torch.save(detections, save_path)
