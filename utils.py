import os

import imageio
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import cv2
import nibabel as nib
import einops
import numpy
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def keep_image_size_open_rgba(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGBA', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('P', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask
def keep_image_size_open_L(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('L', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

def keep_image_size_open_rgb(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images_double(images, path, lable):
    imggrid = []
    lablegrid = []

    lable = lable.to("cpu")
    images = images.to("cpu")
    for i in range(images.shape[0]):
        img = einops.rearrange(images[i, ...], "c w h -> w (c h) ")
        imggrid.append(img)
        label_z = einops.rearrange(lable[i, ...], "c w h -> w (c h) ")
        lablegrid.append(label_z)

    img = torch.stack(imggrid, dim=0)
    img = einops.rearrange(img, 'c w h -> (c w) h')
    img = img.to('cpu').numpy()
    lable = torch.stack(lablegrid, dim=0)
    lable = einops.rearrange(lable, 'c w h -> (c w) h')
    lable = lable.to('cpu').numpy()

    lable = lable * 255
    img = np.concatenate((lable, img), axis=1)
    cv2.imwrite(path, img)

def save_images_three(images, path, lable, label_):
    imggrid = []
    lablegrid = []
    lablegrid_ = []
    lable = lable.to("cpu")
    label_ = label_.to("cpu")
    images = images.to("cpu")
    for i in range(images.shape[0]):
        img = einops.rearrange(images[i, ...], "c w h -> w (c h) ")
        imggrid.append(img)
        label_z = einops.rearrange(lable[i, ...], "c w h -> w (c h) ")
        lablegrid.append(label_z)
        label_y = einops.rearrange(label_[i, ...], "c w h -> w (c h) ")
        lablegrid_.append(label_y)

    img = torch.stack(imggrid, dim=0)
    img = einops.rearrange(img, 'c w h -> (c w) h')
    img = img.to('cpu').numpy()

    lable = torch.stack(lablegrid, dim=0)
    lable = einops.rearrange(lable, 'c w h -> (c w) h')
    lable = lable.to('cpu').numpy()

    lable_ = torch.stack(lablegrid_, dim=0)
    lable_ = einops.rearrange(lable_, 'c w h -> (c w) h')
    lable_ = lable_.to('cpu').numpy()
    # print(np.unique(lable_))
    # print(np.unique(lable))
    lable = lable * 255
    lable_ = lable_
    img = np.concatenate((img,  lable, lable_), axis=1)
    cv2.imwrite(path, img)

def save_images(images, path):
    imggrid = []
    images = images.to("cpu")
    for i in range(images.shape[0]):
        img = einops.rearrange(images[i, ...], "c w h -> w (c h) ")
        imggrid.append(img)

    img = torch.stack(imggrid, dim=0)
    img = einops.rearrange(img, 'c w h -> (c w) h')
    img = img.to('cpu').numpy()
    cv2.imwrite(path, img)

def save_images_single_channel(images, path, epoch):
    images = images.to("cpu").numpy()
    for i in range(images.shape[0]):
        path_new = path
        img = einops.rearrange(images[i, ...], "c w h -> w h c")
        path_new = os.path.join(path_new, f'{epoch}_{i}.jpg')
        cv2.imwrite(path_new, img)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

    class Get_tager_sample(Dataset):
        def __init__(self):
            self.img_path = os.listdir("/media/ybxy/code/U2net/train_1k")

        def __getitem__(self, idx):
            img_name = self.img_path[idx]
            radar = numpy.load(os.path.join("/media/ybxy/code/U2net/train_1k", img_name))
            # Mytrainsform(radar)
            radar = (torch.from_numpy(radar))

            # radar = torch.squeeze(radar)
            # print(radar.shape)

            tagert = radar[0:4, :, :]

            sample = radar[4:8, :, :]
            tagert = einops.rearrange(tagert, " t c w h ->  c t w h")
            sample = einops.rearrange(sample, " t c w h ->  c t w h")
            return tagert, sample

        def __len__(self):
            return len(self.img_path)
    train = Get_tager_sample()

    dataloader = DataLoader(train, batch_size=4, shuffle=True,num_workers=8,drop_last=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def nii_to_image(filepath, imgfile):
    filenames = os.listdir(filepath)

    for f in filenames:
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)  
        img_fdata = img.get_fdata()
        fname = f.replace('.nii', '')
        img_f_path = os.path.join(imgfile, fname)

        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)

        (x, y, z) = img.shape
        for i in range(z):
            silce = img_fdata[:, :, i]
            pill_image = Image.fromarray(silce)
            rotated_img = pill_image.rotate(90, expand=True)
            slice_ = np.array(rotated_img)
            imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), slice_)


def smooth_segmentation_mask(segmentation_mask):
    segmentation_mask[segmentation_mask == 128] = 0

    kernel_size = 3

    erode_weight = torch.ones((1, 1, kernel_size, kernel_size), device=segmentation_mask.device)
    dilate_weight = torch.ones((1, 1, kernel_size, kernel_size), device=segmentation_mask.device)

    eroded_mask = F.conv2d(1 - segmentation_mask.float(), erode_weight, padding=kernel_size // 2)

    dilated_mask = F.conv2d(eroded_mask, dilate_weight, padding=kernel_size // 2)

    dilated_mask = torch.clamp(dilated_mask, 0, 1)

    return dilated_mask


if __name__ == '__main__':
    segmentation_mask = torch.randn((13, 1, 255, 255)).cuda()

    smoothed_mask = smooth_segmentation_mask(segmentation_mask)

    smoothed_mask = smoothed_mask
