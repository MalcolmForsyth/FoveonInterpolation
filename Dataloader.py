import numpy as np
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pathlib

# root directory
root = pathlib.Path.cwd() / 'Foveon Videos'


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


# Get picture files
targets = get_filenames_of_path(root / 'Input')

def Checkerboard(image: np.ndarray, is_start_zero: bool):
    """Turns an image into a checkerboard pattern by alternating zeroes, starting with a zero in the [0,0] element base on the boolean"""
    x_switch = int(is_start_zero)
    for x in range(len(image)):
        #this is to ensure that horizontally/vertically adjacent elements are not both zero or both activated
        x_switch += 1
        y_switch = x_switch - 1
        for y in range(len(image[x])):
            y_switch += 1
            if y_switch % 2 == 0:
                image[x][y] = np.zeros(image[x][y].shape)
    return

# create the input dataset
inputs = [Checkerboard(img, True) for img in targets] + [Checkerboard(img, False) for img in targets]

# random seed
random_seed = 42

# split dataset into training set and validation set
train_size = 0.8  # 80:20 split

inputs_train, inputs_valid = train_test_split(
    inputs,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

targets_train, targets_valid = train_test_split(
    targets,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

# dataset training
dataset_train = ImageDataset(inputs=inputs_train,
                                    targets=targets_train)

# dataset validation
dataset_valid = ImageDataset(inputs=inputs_valid,
                                    targets=targets_valid)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=2,
                                 shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=2,
                                   shuffle=True)