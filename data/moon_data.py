import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import random


def get_patch(img_in, img_tar, img_color, patch_size, scale, ix=-1, iy=-1):
    # print(type(img_in), type(img_tar), type(img_color))
    (ih, iw) = img_in.size
    tp = scale * patch_size
    ip = patch_size
    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    img_color = img_color.crop((ty, tx, ty + tp, tx + tp))
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}
    return img_in, img_tar, img_color, info_patch


class MoonData(Dataset):
    def __init__(self,
                 lr_dir=None,
                 hr_dir=None,
                 img_dir=None,
                 hr_size=(512, 512),
                 scale_factor=4,
                 transform=ToTensor(),
                 target_transform=ToTensor(),
                 guided_transform=ToTensor(),
                 patched=True,
                 patch_size=32):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.img_dir = img_dir
        self.hr_size = hr_size
        self.scale = scale_factor
        self.patched = patched
        self.patch_size = patch_size
        self.files = sorted(os.listdir(hr_dir))
        self.transform = transform
        self.target_transform = target_transform
        self.guided_transform = guided_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, f'{str(self.files[idx])}')
        img_path = os.path.join(self.img_dir, f'{str(self.files[idx])}')
        hr = Image.open(hr_path)
        img = Image.open(img_path)
        lr_path = os.path.join(self.lr_dir, f'{str(self.files[idx])}')
        lr = Image.open(lr_path)
        lr = lr.resize((hr.size[0] // self.scale, hr.size[1] // self.scale))

        if self.patched:
            lr, hr, img, info = get_patch(lr, hr, img, self.patch_size, self.scale)

        if self.transform:
            lr = self.transform(lr)
        if self.target_transform:
            hr = self.target_transform(hr)
        if self.guided_transform:
            img = self.guided_transform(img)
        return lr, hr, img, self.files[idx]
