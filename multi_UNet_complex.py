import torch.nn as nn
import torch.nn.functional as F

def initialize_conv_layer(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out')
        if layer.bias is not None:
            layer.bias.data.zero_()

def initialize_batchnorm_layer(layer):
    if isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()

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

        # Initialize weights for the conv layers and batchnorm layers
        for layer in self.double_conv:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                initialize_conv_layer(layer)
            elif isinstance(layer, nn.BatchNorm2d):
                initialize_batchnorm_layer(layer)

    def forward(self, x):
        return self.double_conv(x)

class ComplexUNet(nn.Module):
    def __init__(self, num_classes):
        super(ComplexUNet, self).__init__()
        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.center = DoubleConv(256, 512)
        self.dec4 = DoubleConv(512, 256)
        self.dec3 = DoubleConv(256, 128)
        self.dec2 = DoubleConv(128, 64)
        self.dec1 = DoubleConv(64, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

        # Initialize weights for the conv layers and batchnorm layers in each block
        for block in [self.enc1, self.enc2, self.enc3, self.enc4, self.center, self.dec4, self.dec3, self.dec2, self.dec1]:
            for layer in block.children():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    initialize_conv_layer(layer)
                elif isinstance(layer, nn.BatchNorm2d):
                    initialize_batchnorm_layer(layer)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        center = self.center(F.max_pool2d(enc4, 2))
        dec4 = self.dec4(F.interpolate(center, scale_factor=2, mode='bilinear', align_corners=False))
        dec3 = self.dec3(F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=False))
        dec2 = self.dec2(F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=False))
        dec1 = self.dec1(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False))
        final = self.final(dec1)
        return final
