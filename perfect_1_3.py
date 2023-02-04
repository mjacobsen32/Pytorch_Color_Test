import torch
from NN_1_3 import NeuralNetwork_1_3

'''
    Manually creating a model and manaully assigning weights and biases
    These weights and biases create a 100% accurate model of 1 kernel with
    shape (3x3). It leverages the biases to weight each channel independently
    100% accuracy is not possible without the biases designed this way
'''
def getNN_1_3_Perfect() -> NeuralNetwork_1_3:
    model = NeuralNetwork_1_3()
    features_0_weights = torch.zeros([1,3,3,3])
    features_0_weights[0][1][1][0] = 0.1
    features_0_weights[0][1][1][1] = 0.2
    features_0_weights[0][1][1][2] = 0.3

    features_0_bias = torch.zeros([1])

    fc1_weight = torch.zeros([3, 1])
    fc1_weight[0][0] = -0.1
    fc1_weight[1][0] = 0.1
    fc1_weight[2][0] = 0.2

    fc1_bias = torch.zeros([3])
    fc1_bias[0] = 14.0
    fc1_bias[1] = 6.0
    fc1_bias[2] = 0.0

    model.state_dict()['features.0.weight'].data.copy_(features_0_weights)
    model.state_dict()['features.0.bias'].data.copy_(features_0_bias)
    model.state_dict()['fc1.weight'].data.copy_(fc1_weight)
    model.state_dict()['fc1.bias'].data.copy_(fc1_bias)
    
    return model
