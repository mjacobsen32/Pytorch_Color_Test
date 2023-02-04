import torch
import os
from NN_1_3 import NeuralNetwork_1_3
from NN_3_3 import NeuralNetwork_3_3
from my_utils import visTensor

'''
    Messy file...
    
    This file is used to load in models of either 1 kernel (3x3)
    or 3 kernel (3x3) and save their visualized learned parameters
    after they have been saved.
'''

fnames_one = ["One_Kernel_(3X3)_001_8.pt"]
fnames = ["Three_Kernel_(3X3)_001_8.pt",
          "Three_Kernel_(3X3)_1_8.pt",
          "Three_Kernel_(3X3)_01_8.pt"]

for f in fnames_one:
    model = NeuralNetwork_1_3()
    model.load_state_dict(torch.load(os.path.join('./models', f)))
    filter = model.features[0].weight.data.clone()
    visTensor(filter, ch=0, allkernels=True, fname=f)
    
for f in fnames:
    model = NeuralNetwork_3_3()
    model.load_state_dict(torch.load(os.path.join('./models', f)))
    filter = model.features[0].weight.data.clone()
    visTensor(filter, ch=0, allkernels=True, fname=f)