import numpy as np
from ImageDataset import FoveonMaskDataset
from Trainer import Trainer
import pathlib
import glob 

# root directory

root = # root of stuff

paths = []
for folder in glob.glob(root + "*"):
    for sub_folder in glob.glob(folder+"/*"):
        for sub_sub_folder in glob.glob(sub_folder+"/*"):
            paths.append(sub_sub_folder)
            
# split paths into test/train 
train_size = 0.8  # 80:20 split


training_dataset = FoveonMaskDataset(paths)
testing_dataset = FoveonMaskDataset(paths)

# random seed
random_seed = 42
num_epochs = 30

torch.manual_seed(random_seed)

# split dataset into training set and validation set


training_dataloader = DataLoader(dataset=training_dataset,
                                 batch_size=2,
                                 shuffle=True)


testing_dataloader = DataLoader(dataset=testing_dataset,
                                   batch_size=2,
                                   shuffle=True)


model = UNet(in_channels=3, out_channels=3)

print("N Params:", sum(p.numel() for p in model.parameters()))

crit = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr = .01, momentum=.9)

trainer = Trainer(model, torch.device('cuda'), crit, opt,  
                  training_dataloader, testing_dataloader, 
                  lr_scheduler=torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=.1),
                  epochs=num_epochs
                  )