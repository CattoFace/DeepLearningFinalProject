#!/usr/bin/env python
# coding: utf-8

# # Task
# Image colorization using autoencoders is a task that involves training a neural network to add color information to grayscale images.
#
# The autoencoder architecture consists of an encoder that encodes the grayscale image into a lower-dimensional representation and a decoder that reconstructs the colored image from this representation. During training, the model learns to minimize the reconstruction error between the original colored image and the reconstructed image. Autoencoders can capture the underlying structure and dependencies in images, enabling them to generate visually pleasing colorizations. This unsupervised learning approach offers a promising solution for adding color to grayscale images.
#
# For this task, we will attempt to colorize grayscale images of birds. This notebook can also serve as a tutorial for PyTorch and model building.

# # Importing libraries

# In[ ]:


import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


# # Model Architecture
#
# 1. The model has a encoder block for encoding the grayscale image into a lower representation and then a decoder block to generate the colorized image.
# 2. The architecture is inspired from the Unet architecture used for image segmentation.
#
# ![image.png](attachment:877377f3-a986-46b9-9d1c-675f5332819f.png)!
#

# ### Encoder Block

# In[ ]:


class Encoder(nn.Module):
    def __init__(self, do_bn=True):
        super().__init__()
        self.block1 = self.inner_block(1, 32)
        self.block2 = self.inner_block(32, 64)
        self.block3 = self.inner_block(64, 128)
        self.block4 = self.inner_block(128, 256)
        self.block5 = self.inner_block(256, 384)

    def inner_block(self, in_c, out_c, maxpool=2):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # 3, 150, 150
        h1 = self.block1(x)  # 32, 75, 75
        h2 = self.block2(h1)  # 64, 37, 37
        h3 = self.block3(h2)  # 128, 18, 18
        h4 = self.block4(h3)  # 256, 9, 9
        h5 = self.block5(h4)  # 384, 4, 4

        return [h1, h2, h3, h4, h5]


# In[ ]:


x = torch.randn(1, 1, 150, 150)
x_out = Encoder()(x)
len(x_out)


# ### Decoder Block

# In[ ]:


class Decoder(nn.Module):
    def __init__(self, do_bn=True):
        super().__init__()
        self.inner1 = self.inner_block(384, 256, 3, 0)
        self.inner2 = self.inner_block(256, 128, 4, 1)
        self.inner3 = self.inner_block(128, 64, 3, 0)
        self.inner4 = self.inner_block(64, 32, 3, 0)
        self.inner5 = self.inner_block(32, 3, 4, 1, out=True)

        self.cb1 = self.conv_block(512, 256)
        self.cb2 = self.conv_block(256, 128)
        self.cb3 = self.conv_block(128, 64)
        self.cb4 = self.conv_block(64, 32)

    def inner_block(
        self,
        in_c,
        out_c,
        kernel_size,
        padding,
        out=False,
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_c,
                out_c,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU() if not out else nn.Sigmoid(),
            nn.Dropout(0.2) if not out else nn.Identity(),
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, h):
        # 384, 5, 5
        breakpoint()
        x = h[-1]
        x = self.inner1(x)  # 256, 9, 9

        x = torch.concat([x, h[-2]], dim=1)
        x = self.cb1(x)
        x = self.inner2(x)  # 128, 20, 20

        x = torch.concat([x, h[-3]], dim=1)
        x = self.cb2(x)
        x = self.inner3(x)  # 64, 40, 40

        x = torch.concat([x, h[-4]], dim=1)
        x = self.cb3(x)
        x = self.inner4(x)  # 32, 80, 80

        x = torch.concat([x, h[-5]], dim=1)
        x = self.cb4(x)
        x = self.inner5(x)  # 3, 160, 160

        return x


# ### Complete Model

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = 5e-4
        self.encoder = Encoder(do_bn=True)
        self.decoder = Decoder(do_bn=True)
        self.loss_fxn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        h = self.encoder(x)
        h = self.decoder(h)
        return h

    def training_step(self, X):
        pred = self.forward(X)
        loss = self.loss_fxn(pred, X)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def validation_step(self, X):
        with torch.no_grad():
            pred = self.forward(X)
            loss = self.loss_fxn(pred, X)
        return loss
