import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomCrop
import numpy as np 
import scipy.ndimage as image

from tqdm import tqdm

import h5py

class FoveonMaskDataset(Dataset):

    def __init__(self, paths, target_transform=None):

        self.imgs = []
        for path in tqdm(paths):
            file = h5py.File(path, 'r')
            n_frames, w, h, d = file['Data']['Data'].shape
            [self.imgs.append((path, i, w, h, d)) for i in range(n_frames)]
        self.flip = RandomHorizontalFlip()

        self.mask_offset_0 = torch.zeros((896, 1344), dtype=torch.bool)
        for r, row in enumerate(self.mask_offset_0):
            for c, pos in enumerate(row):
                if (r+c)%2 == 0: 
                    self.mask_offset_0[r][c] = True

        self.mask_offset_1 = torch.clone(~self.mask_offset_0)
        self.mask_offset_0 = torch.cat(3*[self.mask_offset_0.unsqueeze(0)])
        self.mask_offset_1 = torch.cat(3*[self.mask_offset_1.unsqueeze(0)])
        
        self.vit_crop = RandomCrop(32*3)
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        
        path, i, w, h, d = self.imgs[idx]

        file = h5py.File(path, 'r')
        y = file['Data']['Data'][i].astype(np.int32)
        
        y = image.convolve1d(image, filter = np.array[1.9,-0.9] , axis = 1, mode = 'constant', origin = -1)
        y = torch.Tensor(y)).float()/2**16
        y = y.permute([2,0,1]) # was h,w,c now c,h,w
        
        # reshape
        y = self.flip(y)
        # random crop for ViT
        y = self.vit_crop(y)
        
        X = torch.clone(y)
        
        # checkerboard
        X = self._checkerboard(X, bool(np.random.rand()>.5))
        
        # resize y 
        y = y[:, 32:32*2, 32:32*2]
        return X, y
    

    def _checkerboard(self, image, is_start_one: bool):
        return image * self.vit_crop(self.mask_offset_0)
