import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from color_model import Generator, split_data
from pathlib import Path


def load_data(train_ratio, color):
    arr = torch.load(f"dataset_{color}.pt")
    train_len = int(len(arr) * train_ratio)
    return arr[:train_len], arr[train_len:]


def train(epochs, batch_size, checkpoint_interval):
    train_loss_list = []
    test_loss_list = []
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
                model_input, expected_output = split_data(batch)
                with torch.autocast("cuda"):
                    output = model(model_input)
                loss = criterion(output, expected_output)
                loss.backward()
                train_loss = loss.item()
                optimizer.step()
                pbar.set_description(
                    f"loss: train- {train_loss:.2f}, test- {test_loss:.2f}"
                )
            test_loss = evaluate(model, test_data)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            pbar.update()
            if checkpoint_interval and epoch % checkpoint_interval == 0:
                torch.save(model.state_dict(), f"checkpoints/model{epoch}")
    plt.title("Model Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(np.arange(epochs), train_loss_list, color="red", label="Train")
    plt.plot(np.arange(epochs), test_loss_list, color="blue", label="Test")
    plt.legend(loc="lower right")
    plt.savefig("graph.png")


def evaluate(model, data):
    model.eval()
    with torch.no_grad(), torch.autocast("cuda"):
        criterion = nn.L1Loss()
        inp, expected_output = split_data(data)
        output = model(inp)
        return criterion(output, expected_output).item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # parameters:
    color = "ycbcr"  # rgb/lab/ycbcr
    train_ratio, epochs, batch_size, checkpoint_interval = 0.99, 10, 32, 1
    # setup
    Path("checkpoints").mkdir(exist_ok=True)  # make sure checkpoints folder exists
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
        torch.save(model.state_dict(), "model")
    else:
        model.load_state_dict(torch.load("model"))
        model.to(device)
        loss = evaluate(model, test_data)
        print(f"test {loss=}")