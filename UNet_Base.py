import torch.nn.init as init
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
from torch import nn


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


# CNN Architecture 1
# https://github.com/bnsreenu/python_for_microscopists/blob/master/208-simple_multi_unet_model.py

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.enc4 = DoubleConv(64, 128)
        self.center = DoubleConv(128, 256)
        self.dec4 = DoubleConv(256, 128)
        self.dec3 = DoubleConv(128, 64)
        self.dec2 = DoubleConv(64, 32)
        self.dec1 = DoubleConv(32, 16)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        center = self.center(F.max_pool2d(enc4, 2))
        dec4 = self.dec4(F.interpolate(center, scale_factor=2))
        dec3 = self.dec3(F.interpolate(dec4, scale_factor=2))
        dec2 = self.dec2(F.interpolate(dec3, scale_factor=2))
        dec1 = self.dec1(F.interpolate(dec2, scale_factor=2))
        final = self.final(dec1)
        return final
