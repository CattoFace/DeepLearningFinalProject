from glob import glob
import torch
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
from PIL import Image

color = "YCbCr"
size = 128
images = glob("dataset/*.jpg")
t = transforms.Compose(
    (transforms.ToImage(), transforms.ToDtype(torch.float16, scale=True))
)

array = torch.empty((len(images), 3, size, size), dtype=torch.float16)
for i, img_path in enumerate(tqdm(images, unit="Images", desc="Loading Images")):
    array[i] = t(Image.open(img_path).resize((size, size)).convert(color))
print("Saving Dataset to pt file")
torch.save(array, f"dataset_{color}.pt")
