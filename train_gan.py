import numpy as np
import torch
from tqdm import tqdm
from color_model import FullModel
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import ToPILImage


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


def combine_output(model_input, model_output, color):
    match color:
        case "RGB":
            return model_output
        case "YCbCr":
            return torch.cat((model_input, model_output), 1)
        case "HSV":
            return torch.cat((model_output, model_input), 1)
    return model_output


def train(epochs, batch_size, model, train_data, test_data, path, color):
    to_image = ToPILImage(color)
    results = []  # gen_discriminated,gen_fid, disc, acc_disc
    data_len = len(train_data)
    loss_disc_train = 0
    with tqdm(total=epochs, unit=" Epoch") as pbar:
        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(data_len)
            train_data_permuted = train_data[permutation]
            batches = torch.split(train_data_permuted, batch_size)
            with tqdm(
                total=data_len, unit=" Image", desc="Learning", leave=False
            ) as pbar_inner:
                for batch in batches:
                    loss_gen_binary_train, loss_gen_l1_train, loss_disc_train = (
                        model.step(batch)
                    )
                    pbar_inner.update(len(batch))
                    pbar_inner.set_description(
                        f"train loss - gen: (binary:{loss_gen_binary_train:.2f}, l1:{loss_gen_l1_train:.2f}), disc: {loss_disc_train:.2f})"
                    )
            gen_discriminated, gen_fid, disc, acc_disc, samples = model.test(
                test_data, 20
            )
            results.append((gen_discriminated, gen_fid, disc, acc_disc))
            pbar.set_description(
                f"test gen loss: (binary:{gen_discriminated:.2f}, FID:{gen_fid:.2f}), disc acc: {acc_disc:.2f}"
            )
            torch.save(model.state_dict(), f"{path}/checkpoints/model{epoch}")

            samples = combine_output(test_data[:20, 0], samples, color)
            samples = [to_image(img).convert(("RGB")) for img in samples]
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
    to_train = True
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
    if to_train:
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
            color,
        )
        torch.save(model.state_dict(), f"{path}/model")
    else:
        model.load_state_dict(torch.load(f"{path}/model"))
        model.to(device)
        loss_gen_disc, loss_gen_fid, loss_disc, loss_disc, _ = model.test(test_data, 0)
        print(
            f"test loss: gen- (disc:{loss_gen_disc:.2f},l1:{loss_gen_fid}), disc- {loss_disc:.2f}"
        )


if __name__ == "__main__":
    train_model(64, 5, True, True, True, 100, "RGB", "unet_west_rgb_10")