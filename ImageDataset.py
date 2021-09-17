import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, RandomHorizontalFlip
import numpy as np 
import scipy.ndimage as im

import h5py

class FoveonMaskDataset(Dataset):

    def __init__(self, paths, target_transform=None):

        self.imgs = []
        for path in paths:
            file = h5py.File(path, 'r')
            n_frames, w, h, d = file['Data']['Data'].shape
            [self.images.append((path, i, w, h, d)) for i in range(n_frames)]
        self.resize = Resize((896, 1344))
        self.flip = RandomHorizontalFlip()

        self.mask_offset_0 = torch.zeros((896,1344), dtype=torch.bool)
        for r, row in enumerate(self.mask_offset_0):
            for c, pos in enumerate(row):
                if (r+c)%2 == 0: 
                    mask_offset_0[r][c] = True

        self.mask_offset_1 = torch.clone(~self.mask_offset_0)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        
        path, i, w, h, d = self.imgs[idx]

        file = h5py.File(path, 'r')
        y = file['Data']['Data'][i]

        #acceptable place to put the blurring convolution?
        im.convolve1d(y, filter = np.array[1.9,-0.9] , axis = 1, mode = 'constant', origin = -1)
        
        # reshape
        y = self.resize(y)
        y = self.flip(y)

        X = torch.clone(y)
        
        # checkerboard
        X = self._checkerboard(X, bool(np.random.rand()>.5))

        return X, y
    

    def _checkerboard(image, is_start_one: bool):
        if is_start_one:
            return torch.masked_select(image, self.mask_offset_0)
        else:
            return torch.masked_select(image, self.mask_offset_1)        
