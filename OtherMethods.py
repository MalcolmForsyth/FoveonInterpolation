import numpy as np
from ImageDataset import FoveonMaskDataset
import pathlib
import glob 
import torch
import numpy
from torch.nn import functional as F
from metrics import PSNR, ssim
from torch.utils.data import DataLoader
# root directory

root = r'/home/william/fov_data/'# root of stuff

paths = []
for folder in glob.glob(root + "*"):
    for sub_folder in glob.glob(folder+"/*"):
        print(sub_folder)
        for sub_sub_folder in glob.glob(sub_folder+"/*"):
            paths.append(sub_sub_folder)

torch.manual_seed(42)
np.random.seed(42)
print(paths[:5])

train_size = int(.8*len(paths))  # 80:20 split
np.random.shuffle(paths)

train_paths = paths[:train_size]
test_paths = paths[train_size:]

training_dataset = FoveonMaskDataset(train_paths)
training_dataset = FoveonMaskDataset(test_paths)

training_dataloader = DataLoader(dataset=training_dataset,
                                 batch_size=4,
                                 shuffle=True,
                                 pin_memory=True)


def Checkered_conv(image, filter):
    #assumes the top-left entry in the image is 0

    #outer checker layers
    image = F.conv2d(image, filter, stride=2)

    #inner checker layers
    image[1:][1:] = F.conv2d(image[1:][1:], filter, stride=2)

    return image

#verification
filter = torch.tensor([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]])
print(1)
print(Checkered_conv(torch.tensor([[[[0.1,1,0,1,0], [1,0,1,0,1], [0,1,0,1,0], [1,0,1,0,1], [0,1,0,1,0]]]]), filter))
#print(Checkered_conv(torch.tensor([[[[0.1,1,0], [1,0,1], [0,1,0]]]]), filter))

bic_mse, l_mse, bic_psnr, bic_ssim, l_psnr, l_ssim = np.zeros(len(training_dataloader))

#(to-do) load model
for i, X, y in enumerate(training_dataloader):
    #define the filters, repeating three times since its RGB
    bic = [[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]
    bic_conv_filter = [bic, bic, bic]
    l_bic = [[0, 1, 0, 1, 0], [1, 0, 8, 0, 1], [0, 8, 0, 8, 0], [1, 0, 8, 0, 1], [0, 1, 0, 1, 0]] / 40
    l_bic_conv_filter = [l_bic, l_bic, l_bic]

    #run the convolutions
    bicubic = Checkered_conv(X, bic_conv_filter)
    l_bicubic = Checkered_conv(X, l_bic_conv_filter)
    #nn_interpolate = model(X)

    bic_mse[i], bic_psnr[i], bic_ssim[i] = mse(y, bicubic), PSNR(y, bicubic), ssim(y, bicubic)
    l_mse[i], l_psnr[i], l_ssim[i] = mse(y, l_bicubic), PSNR(y, l_bicubic), ssim(y, l_bicubic)
    #nn_err[i], nn_psnr[i], l_ssim[i] = mse(y, nn_interpolate), PSNR(y, l_bicubic), ssim(y, l_bicubic)

bic_means = [np.mean(bic_mse), np.mean(bic_ssim), np.mean(bic_psnr)]
l_means = [np.mean(l_mse), np.mean(l_ssim), np.mean(l_psnr)]
#nn_means = [np.mean(nn_mse), np.mean(nn_ssim), np.mean(nn_psnr)]


