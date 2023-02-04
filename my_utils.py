from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
import torchvision

'''
    utility functions
'''


'''
    Originally used to print a matrix of different models testing with 
    different learnign rates and batch size. Not used anymore because
    I found the most accurate learning rate to start at 0.01 and the most
    accurate batch size was consistently 8
'''
def print_mat(mat, lr, bs):
    print("batch_size:\t", end='')
    for b in bs: print("{}\t".format(b), end="")
    print("\n\t\t---------------------------------------------------------")
    print('LR', end='')
    for idx, row in enumerate(mat):
        print("\t\t\n{}\t\t".format(lr[idx]), end='')
        for acc in row:
            print("{:.2f}\t".format(acc), end="")
        print("")


'''
    Print the model parameters
'''
def print_model_weights(model):
    layer = model['fc1']
    print(layer.weight.data[0])
    print(layer.bias.data[0])
    
    
'''
    This function prints out the learned kernel weights
    in the form of a heat map. Pass in the file name to save too
'''
def visTensor(tensor, ch=0, nrow=3, padding=1, fname=None):
    images = []
    for filter in tensor:
        filter = filter.transpose(0, 2)
        for i in filter:
            images.append(torch.stack([i,i,i], 0))
    n = torch.stack(images, 0)
    
    grid = utils.make_grid(n, nrow=nrow, normalize=True, padding=1, pad_value=0.90)
    plt.figure(figsize=(3,3), facecolor='white',edgecolor='red')
    plt.tick_params(left=False)
    plt.xticks(ticks=[])
    plt.yticks(ticks=[], labels=[])
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.savefig(str("./kernels/" + fname + ".png"))