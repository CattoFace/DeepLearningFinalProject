from glob import glob
import torch
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
from PIL import Image

lab = True
size = 128
images = glob("dataset/*.jpg")
t = transforms.ToImage()
array = torch.empty((len(images), 3, size, size), dtype=torch.uint8)
for i, img_path in enumerate(tqdm(images, unit="Images", desc="Loading Images")):
    array[i] = t(
        Image.open(img_path).resize((size, size)).convert("LAB" if lab else "YCbCr")
    )
print("Saving Dataset to pt file")
torch.save(array, "dataset_lab.pt" if lab else "dataset_ycbcr.pt")
