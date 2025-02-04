from torch import nn
import torch
from torcheval.metrics import FrechetInceptionDistance
import torchvision.transforms.v2 as transforms

to_tensor = transforms.Compose(
    (transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True))
)


def combine_data(gray, data, color):
    match color:
        case "RGB":
            return data
        case "HSV":
            return torch.cat((data, gray), 1)
        case "YCbCr":
            return torch.cat((gray, data), 1)
    return data


def split_data(data, color):
    """
    splits a data batch that consists of 1 luminance channel and 2/3 color channels to an input consisting of the luminance channel and an expected output consisting of the 2 color channels
    """
    if color == "HSV":
        return data[:, 2:], data[:, :2]
    else:  # RGB/YCbCr
        return data[:, 0:1], data[:, 1:]


def DownConv(in_channels, leaky=True):
    out_channels = in_channels * 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, True),
        ConvBlock(out_channels, out_channels, leaky),
    )


def UpConv(in_channels, leaky=False):
    out_channels = in_channels // 4
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        ConvBlock(out_channels, out_channels, leaky),
    )


def ConvBlock(in_channels, out_channels, leaky=True, norm=True, relu=True):
    model = [
        nn.Conv2d(
            in_channels,
            out_channels,
            (3, 3),
            padding=1,
            padding_mode="reflect",
            bias=not norm,
        )
    ]
    if norm:
        model += [nn.BatchNorm2d(out_channels)]
    if relu:
        model += [nn.LeakyReLU(0.2, True) if leaky else nn.ReLU(True)]
    return nn.Sequential(*model)


class View(nn.Module):
    def forward(self, x):
        size = x.size()
        return x.view(-1, size[-3], size[-2], size[-1])


class Generator(nn.Module):
    def __init__(self, color: str):
        super(Generator, self).__init__()
        self.conv1 = ConvBlock(1, 32, True)
        self.conv2 = DownConv(32)
        self.conv3 = DownConv(64)
        self.conv4 = DownConv(128)
        self.conv5 = DownConv(256)
        self.conv6 = ConvBlock(512, 512, True)
        self.convT1 = UpConv(1024)
        self.convT2 = UpConv(512)
        self.convT3 = UpConv(256)
        self.convT4 = UpConv(128)
        self.conv_out = nn.Conv2d(64, 3 if color == "RGB" else 2, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        y1 = self.convT1(torch.cat((x6, x5), 1))
        y2 = self.convT2(torch.cat((y1, x4), 1))
        y3 = self.convT3(torch.cat((y2, x3), 1))
        y4 = self.convT4(torch.cat((y3, x2), 1))
        out = self.conv_out(torch.cat((y4, x1), 1))
        out = self.sig(out)
        # out.clamp(0, 255)
        return out


class ResidualDiscriminator(nn.Module):
    def __init__(self, color):
        super(ResidualDiscriminator, self).__init__()
        self.conv1 = ConvBlock(4 if color == "RGB" else 3, 32, True)  # 256
        self.conv2 = DownConv(32)
        self.conv3 = DownConv(64)
        self.conv4 = DownConv(128)
        self.conv5 = DownConv(256)
        self.conv6 = ConvBlock(512, 512, True)
        self.convT1 = UpConv(1024)
        self.convT2 = UpConv(512)
        self.convT3 = UpConv(256)
        self.convT4 = UpConv(128)
        self.conv_out = nn.Conv2d(64, 1, 1)
        self.flat = nn.Flatten()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        y1 = self.convT1(torch.cat((x6, x5), 1))
        y2 = self.convT2(torch.cat((y1, x4), 1))
        y3 = self.convT3(torch.cat((y2, x3), 1))
        y4 = self.convT4(torch.cat((y3, x2), 1))
        out = self.conv_out(torch.cat((y4, x1), 1))
        out = self.flat(out)
        return out


def SeqDiscriminator(color, patch: bool = False):
    disc = nn.Sequential(
        ConvBlock(4 if color == "RGB" else 3, 64, True),
        DownConv(64),
        DownConv(128),
        DownConv(256),
        nn.Conv2d(512, 1, 1),
        nn.Flatten(),
    )
    if not patch:
        disc = nn.Sequential(
            disc,
            nn.LazyLinear(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )
    return disc


class FullModel(nn.Module):
    def __init__(
        self,
        color: str,
        device,
        lr: float = 5e-4,
        patch: bool = False,
        unet_gan: bool = True,
        wasserstein: bool = False,
        gen_loss_weight: int | float = 512,
    ) -> None:
        super(FullModel, self).__init__()
        self.color = color
        self.to_image = transforms.ToPILImage(color)
        self.gen = Generator(color)
        if unet_gan:
            self.disc = ResidualDiscriminator(color)
        else:
            self.disc = SeqDiscriminator(color, patch)
        self.wasserstein = wasserstein
        self.criterion_disc = nn.BCEWithLogitsLoss()
        self.criterion_gen = nn.L1Loss()
        self.genOptim = torch.optim.Adam(self.gen.parameters(), lr, betas=(0.5, 0.999))
        self.discOptim = torch.optim.Adam(
            self.disc.parameters(), lr, betas=(0.5, 0.999)
        )
        self.l1_loss_weight = gen_loss_weight
        self.device = device
        self.label_real: torch.Tensor = None
        self.label_fake: torch.Tensor = None
        self.fid = FrechetInceptionDistance()

    def check_buffers(self, single_size):
        """
        Makes sure the BCE target tensors exist and are in the right size before using them
        """
        if self.label_real is None:
            # label smoothing
            self.label_real = torch.full(
                (1024, single_size), 0.9, dtype=torch.float16, device=self.device
            )
            self.register_buffer("real", self.label_real, False)
            self.label_fake = torch.full(
                (1024, single_size), 0.1, dtype=torch.float16, device=self.device
            )
            self.register_buffer("fake", self.label_fake, False)

    def calc_disc_loss(self, output, real, size, single_size):
        """
        Calculates the loss for the discriminator
        """
        self.check_buffers(single_size)
        if self.wasserstein:
            return -output.mean() if real else output.mean()
        else:
            return self.criterion_disc(
                output, self.label_real[:size] if real else self.label_fake[:size]
            )

    def combine_gen_output(self, gen_input, gen_output):
        """
        Combined the given generator input and output to the same format as the input dataset
        """
        match self.color:
            case "YCbCr" | "RGB":
                return torch.cat((gen_input, gen_output), 1)
            case "HSV":
                return torch.cat((gen_output, gen_input), 1)
        return gen_output

    def step(self, data: torch.Tensor):
        """
        Perform a single training step
        """
        self.train()
        # setup
        gray, colors = split_data(data, self.color)
        size = data.size()[0]

        # real step
        self.disc.zero_grad()
        with torch.autocast(device_type="cuda"):
            out_disc_real = self.disc(data)
            single_size = out_disc_real.size()[1]
            loss_disc_real = self.calc_disc_loss(out_disc_real, True, size, single_size)
            del out_disc_real
            # fake step
            out_gen = self.gen(gray)
            # discriminator
            in_disc_fake = self.combine_gen_output(gray, out_gen.detach())
            out_disc_fake = self.disc(in_disc_fake)
            loss_disc_fake = self.calc_disc_loss(
                out_disc_fake, False, size, single_size
            )
            del out_disc_fake
            loss_disc = loss_disc_real + loss_disc_fake
            loss_disc.backward()
        self.discOptim.step()
        # generator
        self.gen.zero_grad()
        with torch.autocast(device_type="cuda"):
            # again because we did a step on the discriminator
            out_disc_fake = self.disc(self.combine_gen_output(gray, out_gen))
            # True because generator wants disc to be wrong
            loss_gen_disc = self.calc_disc_loss(out_disc_fake, True, size, single_size)
            loss_gen_l1 = self.criterion_gen(out_gen, colors) * self.l1_loss_weight
            del out_disc_fake
            # test other methods like multiplication
            loss_gen = loss_gen_disc + loss_gen_l1
            loss_gen.backward()
        self.genOptim.step()
        return (
            loss_gen_disc.mean().item(),
            loss_gen_l1.mean().item(),
            loss_disc.mean().item(),
        )

    def test(self, data, samples_to_return: int):
        """
        Calculate FID and return samples generated from the given data using the generator
        """
        self.eval()
        with torch.no_grad(), torch.autocast("cuda"):
            # setup
            gray, colors = split_data(data, self.color)
            size = data.size()[0]
            # real step
            out_real = self.disc(data)
            single_size = out_real.size()[1]
            loss_disc_real = self.calc_disc_loss(out_real, True, size, single_size)
            accuracy = (out_real >= 0).sum()
            del out_real
            # fake step
            out_gen = self.gen(gray)
            loss_gen_l1 = (
                self.criterion_gen(out_gen, colors).mean().item() * self.l1_loss_weight
            )
            combined = self.combine_gen_output(gray, out_gen)
            out_disc = self.disc(combined)
            loss_disc_fake = self.calc_disc_loss(out_disc, False, size, single_size)
            loss_disc = loss_disc_real + loss_disc_fake
            loss_disc = loss_disc.mean().item()
            accuracy += (out_disc <= 0).sum()

            # FID calculation and samples creation
            loss_gen_discriminated = self.criterion_disc(
                out_disc, self.label_real[:size]
            )
            fake_images_tensor = combine_data(gray, out_gen, self.color)
            fake_images = [
                self.to_image(image).convert("RGB") for image in fake_images_tensor
            ]
            fake_images_tensor = torch.stack(to_tensor(fake_images))
            real_images_tensor = data if self.color != "RGB" else data[:, 1:]
            real_images = [
                self.to_image(image).convert("RGB") for image in real_images_tensor
            ]
            real_images_tensor = torch.stack(to_tensor(real_images))
            self.fid.update(fake_images_tensor, False)
            self.fid.update(real_images_tensor, True)
            loss_gen_fid = self.fid.compute()
            self.fid.reset()
            # finale
            loss_gen_discriminated = loss_gen_discriminated.mean().item()
            loss_gen_fid = loss_gen_fid.mean().item()
            return (
                loss_gen_discriminated,
                loss_gen_fid,
                loss_disc,
                (accuracy / (2 * size * single_size)).item(),
                fake_images[:samples_to_return],
                loss_gen_l1,
            )
