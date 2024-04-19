import torch

from torch import nn
from typing import Callable


class DeepCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            self._create_conv(5, 1, 1, 1, 32, 2),
            self._create_activation(),
            self._create_conv(3, 1, 1, 32, 64, 1),
            self._create_activation(),
            self._create_conv(3, 1, 1, 64, 64, 1),
            self._create_activation(),
            self._create_conv(3, 1, 2, 64, 128, 1),
            self._create_activation(),
            self._create_conv(3, 1, 1, 128, 128, 1),
            self._create_activation(),
            self._create_conv(3, 1, 1, 128, 128, 1),
            self._create_activation(),
            self._create_conv(3, 2, 1, 128, 128, 2),
            self._create_activation(),
            self._create_conv(3, 4, 1, 128, 128, 4),
            self._create_activation(),
            self._create_conv(3, 8, 1, 128, 128, 8),
            self._create_activation(),
            self._create_conv(3, 16, 1, 128, 128, 16),
            self._create_activation(),
            self._create_conv(3, 1, 1, 128, 128, 1),
            self._create_activation(),
            self._create_conv(3, 1, 1, 128, 128, 1),
            self._create_activation(),
            self._create_upscale(),
            self._create_conv(3, 1, 1, 128, 64, 1),
            self._create_activation(),
            self._create_conv(3, 1, 1, 64, 64, 1),
            self._create_activation(),
            self._create_conv(3, 1, 1, 64, 32, 1),
            self._create_activation(),
            self._create_conv(3, 1, 1, 32, 16, 1),
            self._create_activation(),
            self._create_conv(3, 1, 1, 16, 1, 1),
        )

    def _create_conv(
        self, kernel_size: int, dilation: int, stride: int, in_channels: int, out_channels: int, padding: int
    ) -> nn.Conv2d:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
            padding_mode="reflect",
        )

    def _create_activation(self) -> nn.ELU:
        return nn.ELU()

    def _create_upscale(self) -> nn.Upsample:
        return nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        # out = torch.nn.functional.sigmoid(out)

        return out


def define_loss_function(
    Lambda: float, mask: torch.Tensor, N: float
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Define the loss function for the model.

    Args:
        `Lambda` - The weight of the center patch.
        `mask` - Binary mask corresponding to the center patch (1 - center, 0 - surrounding pixels).
        `N` - Normalization factor. Defined in the paper as the number of pixels in the image.
    """

    def loss_function(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = output - target
        loss = (
            Lambda * torch.linalg.norm(mask * diff, ord=1, dim=(-2, -1)) / N
            + (1 - Lambda) * torch.linalg.norm((1 - mask) * diff, dim=(-2, -1)) / N
        )

        return loss

    return loss_function
