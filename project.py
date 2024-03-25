import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from color_model import ColorNet
from pathlib import Path


def load_data(train_ratio, lab):
    arr = torch.load("dataset_lab.pt" if lab else "dataset_ycbcr.pt")
    train_len = int(len(arr) * train_ratio)
    return arr[:train_len], arr[train_len:]


def split_data(data):
    """
    splits a data batch that consists of 1 luminance channel and 2 color channels to an input consisting of the luminance channel and an expected output consisting of the 2 color channels
    """
    return data[:, 0:1, :, :], data[:, 1:, :, :]


def train(epochs, batch_size, checkpoint_interval):
    batches = torch.split(train_data, batch_size)
    train_loss_list = []
    test_loss_list = []
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-3, weight_decay=10e-4)
    with tqdm(total=epochs, unit="Epoch") as pbar:
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            for batch in tqdm(batches, leave=False, desc="Learning"):
                batch = batch.to(device).type(torch.float32)
                model_input, expected_output = split_data(batch)
                output = model(model_input)
                loss = criterion(output, expected_output)
                loss.backward()
                optimizer.step()
            train_loss = evaluate(model, train_data, batch_size)
            test_loss = evaluate(model, test_data, batch_size)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            pbar.set_description(f"loss: train:{train_loss:.2f}, test:{test_loss:.2f}")
            pbar.update()
            if checkpoint_interval is not None and epoch % checkpoint_interval == 0:
                torch.save(model.state_dict(), f"checkpoints/model{epoch}")
    plt.title("Model Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(np.arange(epochs), train_loss_list, color="red", label="Train")
    plt.plot(np.arange(epochs), test_loss_list, color="blue", label="Test")
    plt.legend(loc="lower right")
    plt.savefig("graph.png")


def evaluate(model, data, batch_size):
    model.eval()
    batches = torch.split(data, batch_size)
    with torch.no_grad():
        criterion = nn.MSELoss()
        loss = 0
        for batch in tqdm(batches, leave=False, desc="Evaluating"):
            batch = batch.to(device).type(torch.float32)
            inp, expected_output = split_data(batch)
            output = model(inp)
            loss += criterion(output, expected_output).item()
        return loss / len(data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # parameters:
    train_ratio, epochs, batch_size, lab, checkpoint_interval = 0.9, 20, 32, True, 1
    # setup
    Path("checkpoints").mkdir(exist_ok=True)  # make sure checkpoints folder exists
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_data, test_data = load_data(train_ratio, lab)
    print("Loaded data")
    to_train = True
    model = ColorNet().to(device)
    # model: nn.Module = torch.compile(model)
    print("Parameters:", count_parameters(model))
    if to_train:
        print("Starting training")
        model.to(device)
        train(epochs, batch_size, checkpoint_interval)
        torch.save(model.state_dict(), "model")
    else:
        model.load_state_dict(torch.load("model"))
        model.to(device)
        loss = evaluate(model, test_data, batch_size)
        print(f"test {loss=}")
