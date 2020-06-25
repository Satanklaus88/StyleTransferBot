from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def Load_image(imagepath, imsize=128, return_size=False):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor()])
    image = Image.open(imagepath)
    size = min(image.size)
    image = loader(image).unsqueeze(0)
    device = torch.device("cpu")
    if return_size:
        return image.to(device, torch.float), size
    else:
        return image.to(device, torch.float)


def Save_image(tensor, filename):
    out = tensor.squeeze(0)
    out = out.cpu().detach().numpy()
    out = np.moveaxis(out, 0, 2)
    plt.imsave(filename, out)
