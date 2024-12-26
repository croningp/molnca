import torch
import torch.nn as nn


class PerceptionBlock(nn.Module):
    def __init__(
        self,
        num_channels,
        normal_std=0.02,
        use_normal_init=True,
        zero_bias=True,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.normal_std = normal_std
        self.conv1 = torch.nn.Conv3d(
            self.num_channels,
            self.num_channels * 3,
            3,
            stride=1,
            padding=1,
            # groups=self.num_channels,
            bias=False,
        )

        def init_weights(m):
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.normal_(m.weight, std=normal_std)
                if getattr(m, "bias", None) is not None:
                    if zero_bias:
                        torch.nn.init.zeros_(m.bias)
                    else:
                        torch.nn.init.normal_(m.bias, std=normal_std)

        if use_normal_init:
            with torch.no_grad():
                self.apply(init_weights)

    def forward(self, x):
        return self.conv1(x)
