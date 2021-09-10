import numpy as np


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



