import torch
from torch import nn
import torch.nn.functional as F
import math


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.downsample = nn.Sequential(
            # Blur(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2),
        ) if downsample else nn.Sequential()

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = self.downsample(x)
        x = (x + res) / math.sqrt(2)
        return x


class Discriminator(nn.Module):
    def __init__(self, hyper_paras):
        super().__init__()

        self.conv = nn.Sequential(
            DiscriminatorBlock(3, 64, downsample=True),  # 64
            DiscriminatorBlock(64, 128, downsample=True),  # 32
            DiscriminatorBlock(128, 256, downsample=True),  # 16
            DiscriminatorBlock(256, 512, downsample=True),  # 8
            DiscriminatorBlock(512, 512, downsample=True),  # 4
            DiscriminatorBlock(512, 512, downsample=True),  # 2
            )

        self.fc = nn.Sequential(
            nn.Linear(2 * 2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input_dict):
        x = input_dict['img']
        out = self.conv(x)
        return self.fc(out.view(out.shape[0], -1)).squeeze()
