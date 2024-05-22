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

    @staticmethod
    def _get_checkpoints_dir(save_dir: str) -> str:
        return os.path.join(save_dir, "checkpoints")

    @staticmethod
    def _get_plots_dir(save_dir: str) -> str:
        return os.path.join(save_dir, "plots")

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

    def get_checkpoint_path(self, name: str) -> str:
        return os.path.join(self.checkpoints_dir, "checkpoints", name)

    @staticmethod
    def get_checkpoint_path(config: Config, run_name: str, name: str) -> str:
        return os.path.join(SaveManager._get_checkpoints_dir(config.save_dir), run_name, f"{name}.pt")

    @staticmethod
    def load_checkpoint(
        model: nn.Module, optimizer: torch.optim.Optimizer, path: str
    ) -> tuple[nn.Module, torch.optim.Optimizer, list[float], list[float], int]:
        """
        Loads model and optimizer state dicts along with training and validation losses from a checkpoint file. \\
        Returns the loaded model, optimizer, training losses, validation losses, and the epoch.
        """
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]

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
