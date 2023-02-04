import torch
import numpy as np
import torch.nn as nn
from dataset import CustomImageDataset as Dataset
from NN_1_3 import NeuralNetwork_1_3
from NN_3_3 import NeuralNetwork_3_3
from perfect_3_3 import getNN_3_3_Perfect

'''
for m in l:
    model.load_state_dict(torch.load("./models/"+m))
    with torch.no_grad():
        for X, Y in C_L:
            pred = model(torch.from_numpy(X))
            print(pred)
            predicted, actual = classes[pred[0].argmax(0)], classes[Y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print(model.state_dict()[param_tensor])
'''
def show_model(model):
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print(model.state_dict()[param_tensor])
        print('\n\n')

#for X, Y in C_L:
#    print(X)
#    pred = model(torch.from_numpy(X))
#    print(pred)
#    predicted, actual = classes[pred[0].argmax(0)], classes[Y]
#    print(f'Predicted: "{predicted}", Actual: "{actual}"')

def predict(model):
    classes = ['red','green','blue']
    ds = Dataset(3)
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

def main():
    l = ["RBGperfect_1_3.pt","BGRperfect_1_3.pt"]
    
    m2 = getNN_3_3_Perfect()
    #m2 = NeuralNetwork_3_3()
    show_model(m2)
    predict(m2)
    torch.save(m2.state_dict(), "./models/perfect_3_3.pt")
    
main()
