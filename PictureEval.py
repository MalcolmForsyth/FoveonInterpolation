import numpy as np
from ImageDataset import FoveonMaskDataset
import glob 
import torch
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
# root directory

root = r'/home/william/fov_data/'# root of stuff

paths = []
for folder in glob.glob(root + "*"):
    for sub_folder in glob.glob(folder+"/*"):
        for sub_sub_folder in glob.glob(sub_folder+"/*"):
            paths.append(sub_sub_folder)


np.random.seed(42)
# split paths into test/train 
train_size = int(.8*len(paths))  # 80:20 split
np.random.shuffle(paths)

train_paths = paths[:train_size]
test_paths = paths[train_size:]

print(train_paths)

training_dataset = FoveonMaskDataset(train_paths)
testing_dataset = FoveonMaskDataset(test_paths)

# random seed
random_seed = 42
num_epochs = 12

torch.manual_seed(random_seed)

# split dataset into training set and validation set


training_dataloader = DataLoader(dataset=training_dataset,
                                 batch_size=4,
                                 shuffle=True)


testing_dataloader = DataLoader(dataset=testing_dataset,
                                   batch_size=2,
                                   shuffle=True)

plt.imshow(training_dataloader[0])
