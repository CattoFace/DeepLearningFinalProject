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
data, _ = load_data(1, "RGB")
data = data.to(device)
to_image = transforms.ToPILImage("RGB")
breakpoint()
model = FullModel("RGB", device, patch=True).to(device)
model.load_state_dict(torch.load("model", map_location=device))
# model.load_state_dict(torch.load("checkpoints/model7", map_location=device))
# model: torch.nn.Module = torch.compile(model)
model.eval()
for i in range(1, 1000):
    # model.load_state_dict(torch.load(f"checkpoints/model{i}", map_location=device))
    print("loaded ", i)
    index = int(input(f"select image 1-{len(data)-2}: ")) + 1
    # index = 500
    image_in = data[index : index + 2]
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 3, 1)
    image_in_rgb = to_image(image_in[0, 1:])
    breakpoint()
    plt.imshow(image_in_rgb)
    plt.title("Original")
    fig.add_subplot(1, 3, 2)
    plt.imshow(image_in[0, 0].cpu(), cmap="gray")
    plt.title("Grayscale")
    fig.add_subplot(1, 3, 3)
    model_out = model.color_images(image_in[:, 0:1].type(torch.float16))
    # breakpoit()
    image_out = to_image(model_out[0])
    plt.title("Model Colored")
    plt.imshow(image_out)
    plt.show()
