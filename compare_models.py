import numpy as np
import torch
from tqdm import tqdm
from color_model import FullModel
from pathlib import Path
import matplotlib.pyplot as plt


def load_data(train_ratio, color):
    arr = torch.load(f"dataset_{color}.pt")
    permutation = torch.randperm(arr.size()[0])
    arr = arr[permutation]
    train_len = int(len(arr) * train_ratio)
    return arr[:train_len], arr[train_len:]

def save_samples(samples):
    fig = plt.figure(figsize=(10, 10))
    plt.title(f"Epoch {epoch}")
    for i, sample in enumerate(samples):
        fig.add_subplot(4, 5, i + 1)
        plt.imshow(sample)
    plt.savefig("samples.png")
    plt.close()

def count_parameters(model, sample):
    model.test(sample, 0)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# parameters:
train_ratio = 0.98
# setup
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_data, test_data = load_data(train_ratio, color)
train_data, test_data = (
    train_data.to(device),
    test_data.to(device),
)
print("Loaded data")
resume = False
resume_path = "model"
model = FullModel(
    color,
    device,
    5e-4,
    patch=patch,
    unet_gan=unet_gan,
    wasserstein=west,
    gen_loss_weight=gen_weight,
).to(device)
# model: torch.nn.Module = torch.compile(model)
print("Parameters:", count_parameters(model, train_data[:2]))
    model.load_state_dict(torch.load("model"))
    model.to(device)
    (
        loss_gen_disc_test,
        fid,
        loss_disc_test,
        acc_disc,
        samples,
        loss_gen_l1_test,
    ) = model.test(test_data, 20)
    save_samples(samples)

    print(
        f"test loss: gen- (disc:{loss_gen_disc_test:.2f},l1:{loss_gen_l1_test}, FID:{fid}), disc- (BCE:{loss_disc_test:.2f}, accuracy:{acc_disc})"
        )
