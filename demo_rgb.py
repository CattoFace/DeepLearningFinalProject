import torch
from color_model import Generator
from train import load_data
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# setup
torch.manual_seed(0)
device = torch.device("cpu")
print(device)
print("Loaded data")
to_train = True
data, _ = load_data(1, "rgb")
model = Generator("rgb").to(device)
# model.load_state_dict(torch.load("checkpoints/model1", map_location=device))
model.load_state_dict(torch.load("model", map_location=device))
model.eval()
# model = torch.compile(model)
while True:
    index = int(input(f"select image 1-{len(data)}: ")) + 1
    image_in = data[index : index + 1]
    fig = plt.figure(figsize=(1, 3))
    fig.add_subplot(1, 3, 1)
    image_in_rgb = Image.fromarray(np.asarray(image_in[0, 1:].permute((1, 2, 0))))
    plt.imshow(image_in_rgb)
    plt.title("Original")
    fig.add_subplot(1, 3, 2)
    plt.imshow(image_in[0, 0], cmap="gray")
    plt.title("Grayscale")
    fig.add_subplot(1, 3, 3)
    model_out = model(image_in[:, 0].type(torch.float32)).type(torch.uint8)[0]
    breakpoint()
    image_out = Image.fromarray(np.asarray(model_out.permute((1, 2, 0))))
    plt.title("Model Colored")
    plt.imshow(image_out)
    plt.show()
