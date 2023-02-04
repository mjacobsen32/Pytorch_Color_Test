import torch
import numpy as np
import torch.nn as nn
from dataset import CustomImageDataset as Dataset
from NN_1_3 import NeuralNetwork_1_3
from NN_3_3 import NeuralNetwork_3_3
from perfect_3_3 import getNN_3_3_Perfect
from perfect_1_3 import getNN_1_3_Perfect


'''
    Show model structure
'''
def show_model(model):
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print(model.state_dict()[param_tensor])
        print('\n\n')


'''
    Predict all 3 classes with specified model
    Used to visualize what the mdoel is doing and
    if it is accurate with predictions
    (primarily a sanity check function)
'''
def predict(model):
    classes = ['red','green','blue']
    ds = Dataset(3) # PRE-SPECIFIED DATA SET WITH R, G, B
                    # WHEN CALLED WITH LEN=3
                    # SEE DATASET CLASS FOR EXPLANATION

    data = torch.utils.data.DataLoader(ds, 3)

    R_X, R_Y = ds[0:1][0], ds[0][1]
    G_X, G_Y = ds[1:2][0], ds[1][1]
    B_X, B_Y = ds[2:3][0], ds[2][1]

    C_L = [(R_X, R_Y), (G_X, G_Y), (B_X, B_Y)]
    for X, Y in C_L:
        print(X)
        pred = model(torch.from_numpy(X))
        print(pred)
        predicted, actual = classes[pred[0].argmax(0)], classes[Y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

'''
    used to load in a saved model with extension .pt
    show the model
    predict each of the classes
    save the model at the end (really only if you import
    the newly created perfect model)
'''
def main():
    l = ["RBGperfect_1_3.pt","BGRperfect_1_3.pt"]
    
    m2 = getNN_3_3_Perfect() # LOADS IN A NEWLY CREATED PERFECT 3_3 model
    m1 = getNN_1_3_Perfect() # LOADS IN A NEWLY CREATED PERFECT 1_3 model
    show_model(m1)
    show_model(m2)
    #predict(m2)
    #torch.save(m2.state_dict(), "./models/perfect_3_3.pt")
    
main()
