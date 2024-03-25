from torch import nn
import torch


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1)

    def forward(self, x, residual):
        x = self.convT(x)
        combined = torch.cat([x, residual], dim=1)
        out = self.conv1(combined)
        return out


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.conv1 = DownConv(1, 32)  # 512 to 256
        self.conv2 = DownConv(32, 64)  # 256 to 128
        self.conv3 = DownConv(64, 128)  # 128 to 64
        self.conv4 = DownConv(128, 256)  # 64 to 32
        self.conv5 = DownConv(256, 512)  # 32 to 16
        self.convT1 = UpConv(512, 256)  # 16 to 32
        self.convT2 = UpConv(256, 128)  # 32 to 64
        self.convT3 = UpConv(128, 64)  # 64 to 128
        self.convT4 = UpConv(64, 32)  # 128 to 256
        self.convout = nn.ConvTranspose2d(32, 2, 2, 2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        y1 = self.convT1(x5, x4)
        y2 = self.convT2(y1, x3)
        y3 = self.convT3(y2, x2)
        y4 = self.convT4(y3, x1)
        out = self.convout(y4)
        return out
