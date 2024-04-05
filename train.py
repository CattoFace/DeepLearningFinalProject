import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from color_model import Generator, split_data
from pathlib import Path

from torcheval.metrics import FrechetInceptionDistance
import torchvision.transforms.v2 as transforms

to_image = transforms.ToPILImage()


def load_data(train_ratio, color):
    arr = torch.load(f"dataset_{color}.pt")
    permutation = torch.randperm(arr.size()[0])
    arr = arr[permutation]
    train_len = int(len(arr) * train_ratio)
    return arr[:train_len], arr[train_len:]


def train(epochs, batch_size, checkpoint_interval):
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.5, 0.999))
    train_loss = 0
    test_loss = 0
    with tqdm(total=epochs, unit="Epoch") as pbar:
        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(len(train_data))
            train_data_permuted = train_data[permutation]
            batches = torch.split(train_data_permuted, batch_size)
            model.train()
            optimizer.zero_grad()
            for batch in tqdm(batches, leave=False, desc="Learning", unit="Batch"):
                model_input, expected_output = split_data(batch, color)
                with torch.autocast("cuda"):
                    output = model(model_input)
                    loss = criterion(output, expected_output)
                loss.backward()
                train_loss = loss.item()
                optimizer.step()
                pbar.set_description(
                    f"train loss- {train_loss:.2f}, test fid- {test_loss:.2f}"
                )
            test_loss = evaluate(model, test_data, epoch)
            pbar.update()
            if checkpoint_interval and epoch % checkpoint_interval == 0:
                torch.save(
                    model.state_dict(), f"results/no_gan/checkpoints/model{epoch}"
                )


fid = FrechetInceptionDistance()


def evaluate(model, data, epoch):
    model.eval()
    with torch.no_grad(), torch.autocast("cuda"):
        inp, expected_output = split_data(data, color)
        expected_output = expected_output.type(torch.float32)
        output = model(inp).type(torch.float32)
        fid.update(output, False)
        fid.update(expected_output, True)
        loss_fid = fid.compute()
        fid.reset()
        fig = plt.figure(figsize=(10, 10))
        samples = [to_image(image) for image in output[:20]]
        plt.title(f"Epoch {epoch}")
        for i, sample in enumerate(samples):
            fig.add_subplot(4, 5, i + 1)
            plt.imshow(sample)
        plt.savefig(f"results/no_gan/samples{epoch}.png")
        plt.close()
        return loss_fid


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # parameters:
    color = "RGB"  # rgb/lab/ycbcr
    train_ratio, epochs, batch_size, checkpoint_interval = 0.98, 30, 32, 1
    # setup
    Path("results/no_gan/checkpoints").mkdir(
        exist_ok=True, parents=True
    )  # make sure checkpoints folder exists
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_data, test_data = load_data(train_ratio, color)
    train_data, test_data = (
        train_data.to(device).type(torch.float16),
        test_data.to(device).type(torch.float16),
    )
    print("Loaded data")
    to_train = True
    resume = False
    resume_path = "model"
    # model = NewGenerator(color).to(device)
    model = Generator(color).to(device)
    # model: nn.Module = torch.compile(model)
    print("Parameters:", count_parameters(model))
    if to_train:
        print("Starting training")
        model.to(device)
        if resume:
            print(f"Resuming saved {resume_path}")
            model.load_state_dict(torch.load(resume_path))
        train(epochs, batch_size, checkpoint_interval)
        torch.save(model.state_dict(), "results/no_gan/model")
    else:
        model.load_state_dict(torch.load("model"))
        model.to(device)
        loss = evaluate(model, test_data, 0)
        print(f"test {loss=}")
