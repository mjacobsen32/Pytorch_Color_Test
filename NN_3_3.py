import torch


class NeuralNetwork_3_3(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork_3_3, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=3,
                            kernel_size=3,
                            stride=1,
                            groups=1,
                            bias=True),
            torch.nn.ReLU(inplace=True)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=3, out_features=3, bias=True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return(x)
        