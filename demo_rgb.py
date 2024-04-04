import torch
from color_model import Generator
from train import load_data
import matplotlib.pyplot as plt
from PIL import Image

# setup
torch.manual_seed(0)
device = torch.device("cuda")
print(device)
print("Loaded data")
to_train = True
_, data = load_data(0.99, "rgb")
data = data.to(device)
model = Generator("rgb").to(device)
# model: torch.nn.Module = torch.compile(model)
# model.load_state_dict(torch.load("model", map_location=device))
with torch.no_grad(), torch.autocast(device_type="cuda"):
    model.eval()
    for i in range(1, 1000):
        model.load_state_dict(torch.load(f"checkpoints/model{i}", map_location=device))
        index = 10
        image_in = data
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(1, 3, 1)
        image_in_rgb = Image.fromarray(image_in[0, 1:].permute((1, 2, 0)).cpu().numpy())
        plt.imshow(image_in_rgb)
        plt.title("Original")
        fig.add_subplot(1, 3, 2)
        plt.imshow(image_in[0, 0].cpu(), cmap="gray")
        plt.title("Grayscale")
        fig.add_subplot(1, 3, 3)
        model_out = model(image_in[:, 0:1].type(torch.float16)).type(torch.uint8)[0]
        breakpoint()
        image_out = Image.fromarray(model_out.permute((1, 2, 0)).cpu().numpy())
        plt.title("Model Colored")
        plt.imshow(image_out)
        plt.show()
