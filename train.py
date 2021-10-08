from Trainer import Trainer
from UNet import UNet

import numpy as np
from ImageDataset import FoveonMaskDataset
from Trainer import Trainer
import pathlib
import glob 
import torch
import numpy
import wandb
from vit import ViT

from ConvModel import ThreeConvModel

wandb.init("fov-interp")
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


training_dataset = FoveonMaskDataset(train_paths)
testing_dataset = FoveonMaskDataset(test_paths)

# random seed
random_seed = 42
num_epochs = 2

torch.manual_seed(random_seed)

# split dataset into training set and validation set


training_dataloader = DataLoader(dataset=training_dataset,
                                 batch_size=4,
                                 shuffle=True,
                                 num_workers=4,
                                 prefetch_factor=4,
                                 pin_memory=True)


testing_dataloader = DataLoader(dataset=testing_dataset,
                                   batch_size=4,
                                   shuffle=False)
model = ThreeConvModel(1)# UNet(in_channels=3, out_channels=3, n_blocks=5, start_filters=24)
'''
model = ViT(
    image_size = 32*3, # pass three patches in 
    patch_size = 16,
    num_classes = 32**2, # predict every other pixel
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
'''
model = model.to('cuda')
print("N Params:", sum(p.numel() for p in model.parameters()))

'''     def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False
                 ):
'''
def wandb_log(d):
    wandb.log(d)

wandb.watch(model)

crit = torch.nn.MSELoss() # RMSE
opt = torch.optim.SGD(model.parameters(), lr = .001, momentum=.9)

trainer = Trainer(model, torch.device('cuda'), crit, opt,  
                  training_dataloader, testing_dataloader, 
                  lr_scheduler=torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.5),
                  epochs=num_epochs, wandb_log=wandb_log)
print(trainer.run_trainer())
