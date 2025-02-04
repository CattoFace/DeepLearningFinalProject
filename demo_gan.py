import torch
from color_model import FullModel
from train import load_data
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms

# setup
torch.manual_seed(0)
device = torch.device("cuda")
print(device)
print("Loaded data")
to_train = True
color = "HSV"  # YCbCr/HSV
data, _ = load_data(1, color)
data = data.to(device)
model = FullModel(
    color, device, patch=False, unet_gan=False, wasserstein=False, gen_loss_weight=100
).to(device)
model.load_state_dict(
    torch.load("results/basic_hsv/checkpoints/model1", map_location=device)
)
to_image = transforms.ToPILImage(color)
# model: torch.nn.Module = torch.compile(model)
model.eval()
for i in range(1, 1000):
    model.load_state_dict(
        torch.load(f"results/basic_hsv/checkpoints/model{i}", map_location=device)
    )
    index = 3
    # index = int(input(f"select image 1-{len(data)}: ")) + 1
    image_in = data[index : index + 2]
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 3, 1)
    image_in_rgb = to_image(image_in[0]).convert("RGB")
    plt.imshow(image_in_rgb)
    plt.title("Original")
    fig.add_subplot(1, 3, 2)
    plt.imshow(image_in[0, 0 if color == "YCbCr" else 2].cpu(), cmap="gray")
    plt.title("Grayscale")
    fig.add_subplot(1, 3, 3)
    model_out = model.color_images(image_in[:, 0:1].type(torch.float16))
    breakpoint()
    if color == "YCvCr":
        combined = torch.cat((image_in[0, 0:1], model_out))
    else:
        combined = torch.cat((model_out[0], image_in[0, 2:]))
    image_out = to_image(combined).convert("RGB")
    plt.title("Model Colored")
    plt.imshow(image_out)
    plt.show()
