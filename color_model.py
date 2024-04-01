from torch import nn
import torch


def split_data(data):
    """
    splits a data batch that consists of 1 luminance channel and 2 color channels to an input consisting of the luminance channel and an expected output consisting of the 2 color channels
    """
    return data[:, 0:1, :, :], data[:, 1:, :, :]


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        down = self.pool(x)
        out = self.conv_block(down)
        return out


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.convT = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.conv_block = ConvBlock(in_channels * 2, out_channels)

    def forward(self, x, residual):
        x = self.convT(x)
        combined = torch.cat([x, residual], dim=1)
        out = self.conv_block(combined)
        return out


class ConvMiniBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvMiniBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, (3, 3), padding=1, padding_mode="reflect"
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv(x)
        # make sure stays 4d even for single input
        size = out.size()
        out = out.view(-1, size[-3], size[-2], size[-1])
        out = self.norm(out)
        out = self.relu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.mini_conv1 = ConvMiniBlock(in_channels, out_channels)
        self.mini_conv2 = ConvMiniBlock(out_channels, out_channels)
        self.mini_conv3 = ConvMiniBlock(out_channels, out_channels)

    def forward(self, x):
        out = self.mini_conv1(x)
        out = self.mini_conv2(out)
        out = self.mini_conv3(out)
        return out


class Generator(nn.Module):
    def __init__(self, color: str):
        super(Generator, self).__init__()
        self.conv1 = ConvBlock(1, 32)  # 256
        self.conv2 = DownConv(32, 64)  # 256 to 128
        self.conv3 = DownConv(64, 128)  # 128 to 64
        self.conv4 = DownConv(128, 256)  # 64 to 32
        self.conv5 = DownConv(256, 256)  # 32 to 16
        self.convT1 = UpConv(256, 128)  # 16 to 32
        self.convT2 = UpConv(128, 64)  # 32 to 64
        self.convT3 = UpConv(64, 32)  # 64 to 128
        self.convT4 = UpConv(32, 32)  # 128 to 256
        self.conv_out = nn.Conv2d(32, 3 if color == "rgb" else 2, 1)  # 256
        self.sig = nn.Sigmoid()

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
        out = self.conv_out(y4)
        out = self.sig(out) * 255
        return out


class Discriminator(nn.Module):
    def __init__(self, color: str):
        super(Discriminator, self).__init__()
        self.conv1 = DownConv(3 if color == "rgb" else 2, 64)
        self.conv2 = DownConv(64, 128)
        self.conv3 = DownConv(128, 256)
        self.fc = nn.LazyLinear(512)
        self.relu = nn.LeakyReLU(0.2)
        self.fc_out = nn.Linear(512, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc_out(x)
        x = self.activation(x)
        return x


class FullModel(nn.Module):
    def __init__(self, color: str, lr) -> None:
        super(FullModel, self).__init__()
        self.gen = Generator(color)
        self.disc = Discriminator(color)
        self.criterion = nn.BCELoss()
        self.genOptim = torch.optim.Adam(self.gen.parameters(), lr)
        self.discOptim = torch.optim.Adam(self.disc.parameters(), lr)
        self.label_real = torch.full((1024,), 1)
        self.label_fake = torch.full((1024,), 0)

    def step(self, data: torch.Tensor):
        # setup
        gray, colors = split_data(data)
        size = data.size()[0]

        # real step
        self.disc.zero_grad()
        out_real = self.disc(colors)
        loss_real = self.criterion(out_real, self.label_real[:size])
        loss_real.backward()

        # fake step
        out_gen = self.gen(gray)
        # discriminator
        out_fake = self.disc(out_gen)
        loss_fake = self.criterion(out_fake, self.label_fake[:size])
        loss_fake.backward()
        loss_disc = loss_real + loss_fake
        self.discOptim.step()
        # generator
        self.gen.zero_grad()
        # again because we did a step on the discriminator
        out_fake = self.disc(out_gen)
        loss_gen = self.criterion(out_fake, self.label_real[:size])
        loss_gen.backward()
        self.genOptim.step()

        # finale
        loss_disc = loss_disc.mean().item()
        loss_gen = loss_gen.mean().item()
        return loss_disc, loss_gen
