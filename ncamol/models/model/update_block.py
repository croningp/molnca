import torch
import torch.nn as nn


def make_sequental(num_channels, channel_dims):
    conv3d = torch.nn.Conv3d(num_channels * 3, channel_dims[0], kernel_size=1)
    relu = torch.nn.ReLU()
    layer_list = [conv3d, relu]
    for i in range(1, len(channel_dims)):
        layer_list.append(
            torch.nn.Conv3d(channel_dims[i - 1],
                            channel_dims[i], kernel_size=1)
        )
        layer_list.append(torch.nn.ReLU())
    layer_list.append(
        torch.nn.Conv3d(channel_dims[-1],
                        num_channels, kernel_size=1, bias=False)
    )
    return torch.nn.Sequential(*layer_list)

class UpdateBlock(nn.Module):
    def __init__(
        self,
        num_channels: int = 7,
        channel_dims=[42, 42],
        normal_std=0.02,
        use_normal_init=True,
        zero_bias=True,
    ):
        super().__init__()
        self.out = make_sequental(num_channels, channel_dims)

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
        return self.out(x)