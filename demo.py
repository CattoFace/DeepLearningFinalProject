import torch
from color_model import ColorNet
from project import load_data
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# parameters:
train_ratio, epochs, batch_size, lab, checkpoint_interval = 0.9, 20, 32, True, 1
# setup
torch.manual_seed(0)
device = torch.device("cpu")
print(device)
data, _ = load_data(1, lab)
print("Loaded data")
to_train = True
model = ColorNet().to(device)
model.load_state_dict(torch.load("model", map_location=device))
model.eval()
# model = torch.compile(model)
fig = plt.figure(figsize=(1, 3))
while True:
    index = int(input(f"select image 1-{len(data)}: ")) + 1
    image = data[index : index + 1]
    fig.add_subplot(1, 3, 1)
    image_in = Image.fromarray(np.asarray(image[0].permute(1, 2, 0)), "LAB").convert(
        "RGB"
    )
    plt.imshow(image_in)
    plt.title("Original")
    fig.add_subplot(1, 3, 2)
    plt.imshow(image[0, 0], cmap="gray")
    plt.title("Grayscale")
    fig.add_subplot(1, 3, 3)
    model_out = model(image[:, 0].type(torch.float32)).type(torch.uint8)[0]
    breakpoint()
    image_out = Image.fromarray(
        np.asarray(torch.cat([image[:, 0], model_out]).permute((1, 2, 0))), "LAB"
    ).convert("RGB")
    plt.title("Model Colored")
    plt.imshow(image_out)
    plt.show()
