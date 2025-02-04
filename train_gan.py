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


def save_samples(samples, epoch, path):
    fig = plt.figure(figsize=(10, 10))
    plt.title(f"Epoch {epoch}")
    for i, sample in enumerate(samples):
        fig.add_subplot(4, 5, i + 1)
        plt.imshow(sample)
    plt.savefig(f"{path}/samples{epoch}.png")
    plt.close()


def train(epochs, batch_size, model, train_data, test_data, path):
    results = []  # gen_discriminated,gen_fid, disc, acc_disc
    data_len = len(train_data)
    loss_disc_train = 0
    with tqdm(total=epochs, unit=" Epoch") as pbar:
        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(data_len)
            train_data_permuted = train_data[permutation]
            batches = torch.split(train_data_permuted, batch_size)
            loss_gen_disc_train = 0
            loss_disc_train = 0
            loss_gen_l1_train = 0
            with tqdm(
                total=data_len, unit=" Image", desc="Learning", leave=False
            ) as pbar_inner:
                for batch in batches:
                    loss_gen_disc_train, loss_gen_l1_train, loss_disc_train = (
                        model.step(batch)
                    )
                    pbar_inner.update(len(batch))
                    pbar_inner.set_description(
                        f"train loss - gen: (binary:{loss_gen_disc_train:.2f}, l1:{loss_gen_l1_train:.2f}), disc: {loss_disc_train:.2f})"
                    )
            (
                loss_gen_disc_test,
                fid,
                loss_disc_test,
                acc_disc,
                samples,
                loss_gen_l1_test,
            ) = model.test(test_data, 20)
            results.append(
                (
                    loss_gen_disc_train,
                    loss_gen_disc_test,
                    loss_disc_train,
                    loss_disc_test,
                    loss_gen_l1_train,
                    loss_gen_l1_test,
                    acc_disc,
                    fid,
                )
            )
            pbar.set_description(
                f"test gen loss: (binary:{loss_gen_disc_test:.2f}, FID:{fid:.2f}), disc acc: {acc_disc:.2f}"
            )
            torch.save(model.state_dict(), f"{path}/checkpoints/model{epoch}")

            save_samples(samples, epoch, path)
            pbar.update()
    np.savetxt(path + "/test_results.txt", results, delimiter=",", fmt="%1.2f")


def count_parameters(model, sample):
    model.test(sample, 0)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    batch_size: int,
    epochs: int,
    patch: bool,
    unet_gan: bool,
    west: bool,
    gen_weight: int | float,
    color: str,
    path: str,
):
    path = "results/" + path
    # parameters:
    train_ratio = 0.98
    # setup
    Path(path, "checkpoints").mkdir(
        exist_ok=True, parents=True
    )  # make sure checkpoints folder exists
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
    print("Starting training")
    model.to(device)
    if resume:
        print(f"Resuming saved {resume_path}")
        model.load_state_dict(torch.load(resume_path))
    train(
        epochs,
        batch_size,
        model,
        train_data,
        test_data,
        path,
    )
    torch.save(model.state_dict(), f"{path}/model")


if __name__ == "__main__":
    train_model(64, 100, False, True, False, 512, "RGB", "final_unet_RGB")
