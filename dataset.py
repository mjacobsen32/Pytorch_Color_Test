import random
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch

'''
    This is our custom Red, Green, Blue dataset
    it is created with all zeros in a matrix of shape (3,3,3)
    then the middle pixel is manually changed to be the max (255)
    value for the color channel, creating a simple dot in the middle
    of the 3x3 image
'''
class CustomImageDataset(Dataset):
    def __init__(self, len):
        self.len = len
        red_dot = np.zeros(shape=(3,3,3),dtype=np.uint8)
        red_dot[1][1][0]=255
        green_dot = np.zeros(shape=(3,3,3),dtype=np.uint8)
        green_dot[1][1][1]=255
        blue_dot = np.zeros(shape=(3,3,3),dtype=np.uint8)
        blue_dot[1][1][2]=255
        
        d = {0:red_dot, 1:green_dot, 2:blue_dot}
        
        # for testing purposes if we only want one red, green, and blue
        # to use we just intialize the length of the dataset as 3
        # otherwise it creates a completely random rgb dataset
        if len != 3:
            self.labels = []
            self.images = []
            for _ in range(0, len):
                n = random.randrange(3)
                self.labels.append(n)
                self.images.append(d[n])
        elif len == 3:
            self.labels = [0,1,2]
            self.images = [red_dot,green_dot,blue_dot]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = np.array(self.images[idx], dtype=np.float32)
        return image, self.labels[idx]