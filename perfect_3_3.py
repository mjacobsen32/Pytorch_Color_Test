import torch
from NN_3_3 import NeuralNetwork_3_3

def getNN_3_3_Perfect() -> NeuralNetwork_3_3:
    model = NeuralNetwork_3_3()
    features_0_weights = torch.zeros([3,3,3,3])
    features_0_weights[0][1][1][0] = 1.0
    features_0_weights[1][1][1][1] = 1.0
    features_0_weights[2][1][1][2] = 1.0

    features_0_bias = torch.zeros([3])

    fc1_weight = torch.zeros([3, 3])
    fc1_weight[0][0] = 1.0
    fc1_weight[1][1] = 1.0
    fc1_weight[2][2] = 1.0

    fc1_bias = torch.zeros([3])
    fc1_bias[0] = 0.0
    fc1_bias[1] = 0.0
    fc1_bias[2] = 0.0

    model.state_dict()['features.0.weight'].data.copy_(features_0_weights)
    model.state_dict()['features.0.bias'].data.copy_(features_0_bias)
    model.state_dict()['classifier.0.weight'].data.copy_(fc1_weight)
    model.state_dict()['classifier.0.bias'].data.copy_(fc1_bias)
    
    return model
