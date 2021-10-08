from torch.nn import functional as F
import numpy as np
import torch

def Checkered_conv(image, filter):
    mask_offset_1 = torch.zeros((896, 1344), dtype=torch.bool)
    for r, row in enumerate(mask_offset_1):
        for c, pos in enumerate(row):
            if (r+c)%2 == 1: 
                mask_offset_1[r][c] = True
    mask_offset_1 = torch.cat(3*[mask_offset_1.unsqueeze(0)])
    #outer checker layers
    print("abc")
    mask_offset_1 = torch.cat(image.shape[0]*[mask_offset_1.unsqueeze(0)])
    checkerboard = image * mask_offset_1

    filter = torch.tensor(filter)
    checkerboard = checkerboard.unsqueeze(1)
    filter = filter.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    output = F.conv3d(checkerboard, filter, padding=(0,1,1))
    output = output.squeeze(1)
    image[~mask_offset_1] = output[~mask_offset_1]
    return image

#verification
filter = torch.tensor([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
print(1)
print(filter.shape)
image = torch.rand(2,3,896,1344)
output = Checkered_conv(image, filter)
print(output)
print(output.shape)