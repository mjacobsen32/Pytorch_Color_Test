import numpy as np
import torch

from dataset import CustomImageDataset as Dataset
from my_utils import visTensor
from NN_1_3 import NeuralNetwork_1_3
from NN_3_3 import NeuralNetwork_3_3

batch = 8   
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cpu"), y.to("cpu")
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct, test_loss

def run(model, ds, m):
    _, test_set, _ = torch.utils.data.random_split(dataset=ds, lengths=[0, 3000, 0]) 
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    test_acc, test_loss = test(test_loader, model, loss_fn)
    print("test_acc: {}, val_loss: {}".format(test_acc, test_loss))
    filter = model.features[0].weight.data.clone()
    visTensor(filter, ch=0, nrow=3, padding=1, fname=m)

ds = Dataset(3000)
l = ["perfect_1_3.pt","One_Kernel_(3X3)_0.pt","One_Kernel_(3X3)_1.pt","One_Kernel_(3X3)_2.pt","One_Kernel_(3X3)_3.pt","One_Kernel_(3X3)_4.pt","One_Kernel_(3X3)_5.pt","One_Kernel_(3X3)_6.pt"]
mod = NeuralNetwork_1_3()
for idx, m in enumerate(l):
    print("\n\n{} ACCURACY:\n".format(m))
    mod.load_state_dict(torch.load("./models/"+m))
    run(mod, ds, m)
    
l = ["perfect_3_3.pt","Three_Kernel_(3X3)_0.pt","Three_Kernel_(3X3)_1.pt","Three_Kernel_(3X3)_2.pt","Three_Kernel_(3X3)_3.pt","Three_Kernel_(3X3)_4.pt","Three_Kernel_(3X3)_5.pt","Three_Kernel_(3X3)_6.pt"]
mod = NeuralNetwork_3_3()
for idx, m in enumerate(l):
    print("\n\n{} ACCURACY:\n".format(m))
    mod.load_state_dict(torch.load("./models/"+m))
    run(mod, ds, m)


