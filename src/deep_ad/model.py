import torch

from torch import nn

from src.deep_ad.config import Config


class DeepCNN(nn.Module):
    def __init__(self, config: Config, old_architecture: bool = False) -> None:
        super().__init__()
        self.config = config
        if old_architecture:
            self.network = nn.Sequential(
                self._create_conv(5, 1, 1, 1, 32, 2),
                self._create_batch_norm(32),
                self._create_activation(),
                self._create_conv(3, 1, 1, 32, 64, 1),
                self._create_batch_norm(64),
                self._create_activation(),
                self._create_conv(3, 1, 1, 64, 64, 1),
                self._create_batch_norm(64),
                self._create_activation(),
                self._create_conv(3, 1, 2, 64, 128, 1),
                self._create_batch_norm(128),
                self._create_activation(),
                self._create_conv(3, 1, 1, 128, 128, 1),
                self._create_batch_norm(128),
                self._create_activation(),
                self._create_conv(3, 1, 1, 128, 128, 1),
                self._create_batch_norm(128),
                self._create_activation(),
                self._create_conv(3, 2, 1, 128, 128, 2),
                self._create_batch_norm(128),
                self._create_activation(),
                self._create_conv(3, 4, 1, 128, 128, 4),
                self._create_batch_norm(128),
                self._create_activation(),
                self._create_conv(3, 8, 1, 128, 128, 8),
                self._create_batch_norm(128),
                self._create_activation(),
                self._create_conv(3, 16, 1, 128, 128, 16),
                self._create_batch_norm(128),
                self._create_activation(),
                self._create_conv(3, 1, 1, 128, 128, 1),
                self._create_batch_norm(128),
                self._create_activation(),
                self._create_conv(3, 1, 1, 128, 128, 1),
                self._create_batch_norm(128),
                self._create_activation(),
                self._create_upscale(),
                self._create_conv(3, 1, 1, 128, 64, 1),
                self._create_batch_norm(64),
                self._create_activation(),
                self._create_conv(3, 1, 1, 64, 64, 1),
                self._create_batch_norm(64),
                self._create_activation(),
                self._create_conv(3, 1, 1, 64, 32, 1),
                self._create_batch_norm(32),
                self._create_activation(),
                self._create_conv(3, 1, 1, 32, 16, 1),
                self._create_batch_norm(16),
                self._create_activation(),
                self._create_conv(3, 1, 1, 16, 1, 1),
            )
        else:
            use_bn = self.config.batch_norm
            self.network = nn.Sequential(
                self._create_conv_module(5, 1, 1, 1, 32, 2, use_bn),
                self._create_conv_module(3, 1, 1, 32, 64, 1, use_bn),
                self._create_conv_module(3, 1, 1, 64, 64, 1, use_bn),
                self._create_conv_module(3, 1, 2, 64, 128, 1, use_bn),
                self._create_conv_module(3, 1, 1, 128, 128, 1, use_bn),
                self._create_conv_module(3, 1, 1, 128, 128, 1, use_bn),
                self._create_conv_module(3, 2, 1, 128, 128, 2, use_bn),
                self._create_conv_module(3, 4, 1, 128, 128, 4, use_bn),
                self._create_conv_module(3, 8, 1, 128, 128, 8, use_bn),
                self._create_conv_module(3, 16, 1, 128, 128, 16, use_bn),
                self._create_conv_module(3, 1, 1, 128, 128, 1, use_bn),
                self._create_conv_module(3, 1, 1, 128, 128, 1, use_bn),
                self._create_upscale(),
                self._create_conv_module(3, 1, 1, 128, 64, 1, use_bn),
                self._create_conv_module(3, 1, 1, 64, 64, 1, use_bn),
                self._create_conv_module(3, 1, 1, 64, 32, 1, use_bn),
                self._create_conv_module(3, 1, 1, 32, 16, 1, use_bn),
                self._create_conv(3, 1, 1, 16, 1, 1),
            )

    def _create_conv_module(
        self,
        kernel_size: int,
        dilation: int,
        stride: int,
        in_channels: int,
        out_channels: int,
        padding: int,
        batch_norm: bool,
    ) -> nn.Module:
        modules = [self._create_conv(kernel_size, dilation, stride, in_channels, out_channels, padding)]
        if batch_norm:
            modules.append(nn.BatchNorm2d(num_features=out_channels, track_running_stats=True))
        modules.append(nn.ELU())

        return nn.Sequential(*modules)

    def _create_conv(
        self, kernel_size: int, dilation: int, stride: int, in_channels: int, out_channels: int, padding: int
    ) -> nn.Conv2d:
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
            padding_mode="reflect",
        )
        if self.config.init_weights:
            nn.init.trunc_normal_(conv.weight, mean=0.0, std=1.0, a=-5e-2, b=5e-2)
            nn.init.constant_(conv.bias, 0.0)

        return conv

    def _create_batch_norm(self, num_features: int) -> nn.BatchNorm2d:
        return nn.BatchNorm2d(num_features=num_features, track_running_stats=True)

    def _create_activation(self) -> nn.ELU:
        return nn.ELU()

    def _create_upscale(self) -> nn.Upsample:
        return nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        # out = torch.clip(out, -1.0, 1.0)
        # out = torch.nn.functional.sigmoid(out)
        out = torch.nn.functional.tanh(out)

        return out
