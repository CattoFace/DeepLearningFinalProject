from glob import glob
import torch
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
from PIL import Image

images = glob("dataset/*.jpg")
t = transforms.ToImage()
size = 128
array = torch.empty((len(images), 4, size, size), dtype=torch.uint8)
array_gray = torch.empty((len(images), 1, size, size), dtype=torch.uint8)
for i, img_path in enumerate(tqdm(images, unit="Images", desc="Loading Images")):
    image = Image.open(img_path).resize((size, size))
    array[i, 1:] = t(image)
    array[i, 0] = t(image.convert("L"))
print("Saving Dataset to pt file")
torch.save(array, "dataset_rgb.pt")
