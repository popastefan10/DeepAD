import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2

from collections import OrderedDict
from torch.utils.data import DataLoader
from typing import Callable, Literal

from src.deep_ad.config import Config
from src.deep_ad.image import create_center_mask, plot_images
from src.deep_ad.measurements import GPUWatch, Stopwatch
from src.deep_ad.save_manager import SaveManager


def define_loss_function(
    Lambda: float, mask: torch.Tensor, N: float, loss_type: Literal["l1_norm", "l1_loss"] = "l1_norm"
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Define the loss function for the model.

    Args:
        * `Lambda` - The weight of the center patch.
        * `mask` - Binary mask corresponding to the center patch (1 - center, 0 - surrounding pixels).
        * `N` - Normalization factor. Defined in the paper as the number of pixels in the image.
    """

    def loss_function_l1_norm(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = output - target
        loss = torch.mean(
            Lambda * torch.linalg.norm(mask * diff, ord=1, dim=(-2, -1)) / N
            + (1 - Lambda) * torch.linalg.norm((1 - mask) * diff, dim=(-2, -1)) / N
        )

        return loss

    def loss_function_l1_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(
            Lambda * torch.nn.functional.l1_loss(mask * output, mask * target, reduction="sum") / N
            + (1 - Lambda) * torch.nn.functional.l1_loss((1 - mask) * output, (1 - mask) * target, reduction="sum") / N
        )

        return loss

    return loss_function_l1_norm if loss_type == "l1_norm" else loss_function_l1_loss


def create_optimizer(model: nn.Module, config: Config) -> torch.optim.Adam:
    return torch.optim.Adam(
        model.parameters(), lr=config.optim_lr, betas=config.optim_adam_betas, eps=config.optim_adam_eps
    )


class Trainer:
    """
    A Trainer object should receive a model and some configuration parameters and train the model for a specified number
    of epochs. It does not handle loading pretrained models. However, it must save data during training in order to be
    able to resume training from best checkpoints.
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        run_name: str,
        train_epochs: int | None = None,
        limit_batches: int | None = None,
        save_epochs: list[int] | None = None,
    ) -> None:
        """
        Initializes the Trainer object.

        Args:
            * `run_name` - The name of the run. This will be used to save checkpoints and other data.
            * `limit_batches` - If this value is not `None`, this number of batches will be used for training and validation.
            The batches will be extracted prior to iterating through the dataset, because otherwise the DataLoader will
            shuffle the data and we won't use the same batches for training and validation.
            * `save_epochs` - A list of epochs at which the model should be saved.
        """
        self.config = config
        self.device = config.device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.save_manager = SaveManager(config)
        self.run_name = run_name

        self.mask = torch.asarray(create_center_mask()).to(self.device)
        self.loss_function = define_loss_function(
            Lambda=config.loss_Lambda, mask=self.mask, N=config.loss_N, loss_type=config.loss_type
        )
        self.optimizer = create_optimizer(model, config)
        self.train_epochs = train_epochs or config.train_epochs
        self.save_epochs = save_epochs or []

        # If batches are limited, we need to create new DataLoaders that won't shuffle the data
        self.limit_batches = limit_batches
        if limit_batches:
            self.train_dataloader = DataLoader(
                train_dataloader.dataset,
                batch_size=train_dataloader.batch_size,
                num_workers=train_dataloader.num_workers,
                shuffle=False,
            )
            self.val_dataloader = DataLoader(
                val_dataloader.dataset,
                batch_size=val_dataloader.batch_size,
                num_workers=val_dataloader.num_workers,
                shuffle=False,
            )

        self.train_num_batches = len(self.train_dataloader)
        self.val_num_batches = len(self.val_dataloader)

    def _save_checkpoint(self, epoch_num: int, train_losses: list[float], val_losses: list[float], name: str) -> None:
        self.save_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            train_losses=train_losses,
            val_losses=val_losses,
            epoch=epoch_num,
            run_name=self.run_name,
            name=name,
        )

    def _save_config(self) -> None:
        """Saves the configuration passed in the constructor."""
        self.save_manager.save_config(config=self.config, run_name=self.run_name)

    def _normalize_to_mean_std(self, tensor: torch.Tensor, mean: torch.tensor, std: torch.tensor) -> torch.Tensor:
        output = tensor - tensor.mean(dim=(0, 2, 3), keepdim=True)
        output = output / output.std(dim=(0, 2, 3), keepdim=True)
        output = output * std + mean

        return output

    def train_epoch(self, epoch_num: int) -> float:
        """
        Computes the average loss for the training dataset and updates the model's weights.

        Args:
            `epoch_num` - The current epoch number, 0-indexed.

        Returns:
            The average loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        for batch_num, (images, _) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            # Remove the center from each image
            images = images.to(self.device)
            inputs = images * (1 - self.mask)
            output = self.model(inputs)
            output = self._normalize_to_mean_std(
                output, images.mean(dim=(0, 2, 3), keepdim=True), images.std(dim=(0, 2, 3), keepdim=True)
            )
            if epoch_num == 0 or (epoch_num + 1) % 50 == 0:
            # if epoch_num == 99:
                p_images = [images[0], inputs[0], output[0]]
                p_images = [im.cpu().detach().numpy().squeeze() for im in p_images]
                plot_images(images=p_images, titles=["Original", "Input", "Output"], cols=3)
                for im in p_images:
                    print(im.shape, im.min(), im.max())

            loss = self.loss_function(output, images)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if batch_num % 100 == 0:
                print(f"\tBatch {batch_num + 1:3d}/{self.train_num_batches}: Loss {loss.item():.6f}")
            if self.limit_batches and batch_num + 1 >= self.limit_batches:
                break

        epoch_loss /= batch_num + 1

        return epoch_loss

    def eval_epoch(self) -> float:
        """
        Computes the average loss for the validation dataset.

        Returns:
            The average loss for the epoch.
        """
        self.model.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batch_num, (images, _) in enumerate(self.val_dataloader):
                # Remove the center from each image
                images = images.to(self.device)
                inputs = images * (1 - self.mask)
                output = self.model(inputs)
                output = self._normalize_to_mean_std(
                    output, images.mean(dim=(0, 2, 3), keepdim=True), images.std(dim=(0, 2, 3), keepdim=True)
                )

                loss = self.loss_function(output, images)
                epoch_loss += loss.item()

                if self.limit_batches and batch_num + 1 >= self.limit_batches:
                    break

            epoch_loss /= batch_num + 1

        return epoch_loss

    def train(self) -> tuple[list[float], list[float]]:
        """
        Returns:
            A tuple containing the training and validation losses for each epoch.
        """
        self._save_config()
        stopwatch = Stopwatch()
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        for epoch_num in range(self.train_epochs):
            stopwatch.start()
            train_loss = self.train_epoch(epoch_num)
            val_loss = self.eval_epoch()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(
                f"Epoch {epoch_num + 1:3d}/{self.train_epochs}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}, time {stopwatch.elapsed_since_beginning():.3f} s"
            )

            # Check if the model should be saved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch_num, train_losses, val_losses, name="best")
            if epoch_num + 1 in self.save_epochs:
                self._save_checkpoint(epoch_num, train_losses, val_losses, name=f"epoch_{epoch_num + 1}")

        print("Training finished.")

        return train_losses, val_losses


class TrainerWatch(Trainer):
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        train_epochs: int | None = None,
        limit_batches: int | None = None,
    ) -> None:
        super().__init__(config, model, train_dataloader, val_dataloader, train_epochs, limit_batches)
        self.stopwatch = Stopwatch()
        self.gpuwatch = GPUWatch()
        self.time_deltas: OrderedDict[str, list[float]] = OrderedDict()
        self.gpu_deltas: OrderedDict[str, list[int]] = OrderedDict()

    def _checkpoint(
        self,
        instruction: str,
        should_print: bool = True,
        last_time_checkpoint: float | None = None,
        last_gpu_checkpoint: float | None = None,
    ) -> None:
        if last_time_checkpoint is not None:
            time_delta = self.stopwatch.elapsed_since_last_checkpoint(last_checkpoint=last_time_checkpoint)
            self.stopwatch.checkpoint()
        else:
            time_delta = self.stopwatch.checkpoint()
        if instruction not in self.time_deltas:
            self.time_deltas[instruction] = [time_delta]
        else:
            self.time_deltas.get(instruction).append(time_delta)

        if last_gpu_checkpoint is not None:
            gpu_delta = self.gpuwatch.gain_since_last_checkpoint(last_checkpoint=last_gpu_checkpoint)
            self.gpuwatch.checkpoint()
        else:
            gpu_delta = self.gpuwatch.checkpoint()
        if instruction not in self.gpu_deltas:
            self.gpu_deltas[instruction] = [gpu_delta]
        else:
            self.gpu_deltas.get(instruction).append(gpu_delta)

        if should_print:
            print(f"{time_delta:.3f} s, {gpu_delta:+5d} MiB | {instruction}")

    def get_overview_df(self) -> pd.DataFrame:
        def format_time(time: float) -> str:
            minutes = int(time // 60)
            seconds = time % 60
            if minutes:
                return f"{minutes}m {int(seconds)}s"
            return f"{seconds:.3f}s"

        df = pd.DataFrame(columns=["Instruction", "Total T", "Avg T", "Total GPU", "Avg GPU"])
        for instruction, time_deltas in self.time_deltas.items():
            gpu_deltas = self.gpu_deltas[instruction]
            df.loc[len(df)] = {
                "Total T": format_time(sum(time_deltas)),
                "Avg T": format_time(sum(time_deltas) / len(time_deltas)),
                "Total GPU": f"{sum(gpu_deltas):+5d} MiB",
                "Avg GPU": f"{sum(gpu_deltas) / len(gpu_deltas):+5.0f} MiB",
                "Instruction": instruction,
            }

        return df

    def train_epoch(self, epoch_num: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        self.stopwatch.checkpoint()
        self.gpuwatch.checkpoint()
        for batch_num, (images, _) in enumerate(self.train_dataloader):
            self._checkpoint(instruction="enumerate(self.train_dataloader)", should_print=epoch_num == 0)
            self.optimizer.zero_grad()
            self._checkpoint(instruction="optimizer.zero_grad()", should_print=epoch_num == 0)

            # Remove the center from each image
            images = images.to(self.device)
            self._checkpoint(instruction="images.to(device)", should_print=epoch_num == 0)
            inputs = images * (1 - self.mask)
            self._checkpoint(instruction="inputs = images * (1 - self.mask)", should_print=epoch_num == 0)
            output = self.model(inputs)
            self._checkpoint(instruction="output = self.model(inputs)", should_print=epoch_num == 0)

            loss = self.loss_function(output, images)
            self._checkpoint(instruction="loss = self.loss_function(output, images)", should_print=epoch_num == 0)
            loss.backward()
            self._checkpoint(instruction="loss.backward()", should_print=epoch_num == 0)
            self.optimizer.step()
            self._checkpoint(instruction="optimizer.step()", should_print=epoch_num == 0)
            epoch_loss += loss.item()
            self._checkpoint(instruction="epoch_loss += loss.item()", should_print=epoch_num == 0)

            if self.limit_batches and batch_num + 1 >= self.limit_batches:
                break

        epoch_loss /= batch_num + 1

        return epoch_loss

    def train(self) -> tuple[list[float], list[float]]:
        self.stopwatch.start()
        self.gpuwatch.start()
        train_losses: list[float] = []
        val_losses: list[float] = []
        for epoch_num in range(self.train_epochs):
            epoch_stopwatch, epoch_gpuwatch = Stopwatch(), GPUWatch()
            epoch_t0, epoch_gpu0 = epoch_stopwatch.start(), epoch_gpuwatch.start()

            time_checkpoint, gpu_checkpoint = self.stopwatch.checkpoint(return_delta=False), self.gpuwatch.checkpoint(
                return_delta=False
            )
            train_loss = self.train_epoch(epoch_num)
            self._checkpoint(
                instruction="train_epoch()",
                should_print=epoch_num == 0,
                last_time_checkpoint=time_checkpoint,
                last_gpu_checkpoint=gpu_checkpoint,
            )

            time_checkpoint, gpu_checkpoint = self.stopwatch.checkpoint(return_delta=False), self.gpuwatch.checkpoint(
                return_delta=False
            )
            val_loss = self.eval_epoch()
            self._checkpoint(
                instruction="eval_epoch()",
                should_print=epoch_num == 0,
                last_time_checkpoint=time_checkpoint,
                last_gpu_checkpoint=gpu_checkpoint,
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self._checkpoint(
                instruction="epoch", should_print=False, last_time_checkpoint=epoch_t0, last_gpu_checkpoint=epoch_gpu0
            )
            print(
                f"Epoch {epoch_num + 1:3d}/{self.train_epochs}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}, {epoch_stopwatch.elapsed_since_beginning():.3f} s, {epoch_gpuwatch.gain_since_beginning():+5d} MiB"
            )

        print(
            f"Training finished: {self.gpuwatch.gain_since_beginning():+5d} MiB, {self.stopwatch.elapsed_since_beginning():.3f} s"
        )

        return train_losses, val_losses
