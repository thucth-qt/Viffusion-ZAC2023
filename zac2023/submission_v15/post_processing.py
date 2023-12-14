from PIL import Image
from torchvision import  transforms
import torch
import torchvision.transforms.functional as F
import pad_color
import numpy as np

RGB_mean = [0.5240151030192457, 0.5826905714889826, 0.6114445483337769][::-1]
RGB_std = [0.2555236373419503, 0.2457990873387373, 0.24959702246625548][::-1]

normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * (torch.Tensor([RGB_std[i] for i in range(3)]).clone() / x.reshape((3, -1)).std(dim=-1)).unsqueeze(-1).unsqueeze(-1)),
        transforms.Lambda(lambda x: x + (torch.Tensor([RGB_mean[i] for i in range(3)]).clone() - x.reshape((3, -1)).mean(dim=-1)).unsqueeze(-1).unsqueeze(-1)),
        transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        transforms.ToPILImage()
    ])

illumination_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + (torch.Tensor([RGB_mean[i] for i in range(3)]).clone() - x.reshape((3, -1)).mean(dim=-1)).unsqueeze(-1).unsqueeze(-1)),
        transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        transforms.ToPILImage()
    ])

def normalize(image, only_illumination=False):
    global normalize_transform, illumination_transform
    
    transform = illumination_transform if only_illumination else normalize_transform
    image = transform(image)
    return image

def scale(image, scale_ratio=1.0):
    new_size = [int(image.size[1]*scale_ratio), int(image.size[0]*scale_ratio)]
    image = F.resize(image, new_size)
    return image

def scale_rectangle(image, h_scale_ratio=0.8, w_scale_ratio=1.0):
    new_size = [int(image.size[1]*h_scale_ratio), int(image.size[0]*w_scale_ratio)]
    image = F.resize(image, new_size)
    return image

def padding_const(image, padding_value=None, desired_width=1024, desired_height=533):
    if padding_value is None:
        padding_value = tuple(int(v*255) for v in RGB_mean)
    
    image = pad_color.pad(image, desired_width, desired_height, color=padding_value)
    return image

def padding_gauss_noise(image, desired_width=1024, desired_height=533, gauss_mean=RGB_mean, gauss_std=RGB_std):
    noisy_image = np.random.normal(np.array(gauss_mean)*255, np.array(gauss_std)*255, size=(desired_height, desired_width, 3))
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image)
    
    x = (noisy_image.size[0] - image.size[0]) // 2
    y = (noisy_image.size[1] - image.size[1]) // 2
    
    noisy_image.paste(image, (x,y))
    
    return noisy_image

def padding(image, padding_stype='const', padding_value=None, desired_width=1024, desired_height=533, gauss_mean=RGB_mean, gauss_std=RGB_std):
    assert padding_stype in ['const', 'gauss_noise']
    
    if padding_stype == "const":
        image = padding_const(image, padding_value, desired_width, desired_height)
    elif padding_stype == "gauss_noise":
        image = padding_gauss_noise(image, desired_width, desired_height, gauss_mean, gauss_std)
        pass
    
    return image

def blend(image, ratio, bg_color=RGB_mean):
    bg = Image.new('RGB', (image.width, image.height), color=tuple(int(v*255) for v in bg_color))
    image = Image.blend(bg, image, alpha=ratio)
    return image

def concat_with_flip(image):
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    concat_image = Image.new("RGB", (image.width + flipped_image.width, image.height))
    concat_image.paste(image, (0, 0))
    concat_image.paste(flipped_image, (image.width, 0))
    
    return concat_image

def concat_images(image1, image2, middle_space=50, bg_color=RGB_mean):
    concat_image = Image.new("RGB", (image1.width + middle_space + image2.width, image1.height), color=tuple(int(v*255) for v in bg_color))
    concat_image.paste(image1, (0, 0))
    concat_image.paste(image2, (image1.width + middle_space, 0))
    
    return concat_image
