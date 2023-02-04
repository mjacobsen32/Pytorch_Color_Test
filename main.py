import numpy as np
import torch
from torchsummary import summary

from dataset import CustomImageDataset as Dataset
from my_utils import print_mat, print_model_weights, visTensor
from NN_1_3 import NeuralNetwork_1_3
from NN_3_3 import NeuralNetwork_3_3
from NN_3_1 import NeuralNetwork_3_1
from NN_1 import NeuralNetwork_1_1

lr = 0.1
batch = 8
epochs = 12
COUNTER = 0

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to("cpu"), y.to("cpu")

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / size
            
            
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


def run(lr, batch, mod_str, model, loss_fn, optimizer, sched, train_loader, val_loader, epochs):
    print("model: {}\nlr: {}\nbatch_size: {}\nloss_func: {}\noptimization: {}\n".format(mod_str, lr, batch, loss_fn, "SGD"))
    for t in range(epochs):
        train_loss = train(train_loader, model, loss_fn, optimizer)
        print("{}: {:.2f}".format(t+1, train_loss), end=' -> ')
        _, val_loss = test(val_loader, model, loss_fn)
        sched.step()
    test_acc, test_loss = test(val_loader, model, loss_fn)
    print("test_acc: {}, val_loss: {}".format(test_acc, test_loss))
    return test_acc

lr = [0.01]
batch = [8]

def multi(mod_str, net, f_string):
    matrix = np.zeros((len(lr), len(batch)))
    counter = 0
    for i in range(100):
        train_set, val_set, _ = torch.utils.data.random_split(dataset=ds, lengths=[8500, 1500, 0]) 
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch[0])
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch[0])
        
        torch.seed()
        model = net().to("cpu")
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr[0])
        sched = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
        matrix[0][0] = run(lr=lr[0], batch=batch[0], mod_str=mod_str, model=model, loss_fn=loss_fn, optimizer=optimizer, 
                                    sched=sched, train_loader=train_loader, val_loader=val_loader, epochs=2)
        if matrix[0][0] == 1.0:
            fname = mod_str + "_" + str(counter)
            torch.save(model.state_dict(), "./models/"+fname+"_"+".pt")
            filter = model.features[0].weight.data.clone()
            visTensor(filter, ch=0, allkernels=True, fname=fname)
            counter+=1
            exit(0)
    return matrix
        

ds = Dataset(10000)

#print("\n\nONE KERNEL (1X1) TEST ACCURACY MATRIX\n")
#acc_mat = multi("One_Kernel_(1X1)", lr, batch, NeuralNetwork_1_1, "one_kernel_1X1")
#print_mat(acc_mat, lr, batch)
#print("AVERAGE ACC: {}".format(acc_mat.sum() / (len(lr) * len(batch))))

#print("\n\nTHREE KERNELs (1X1) TEST ACCURACY MATRIX\n")
#acc_mat = multi("Three_Kernel_(1X1)", lr, batch, NeuralNetwork_3_1, "three_kernel_1X1")
#print_mat(acc_mat, lr, batch)
#print("AVERAGE ACC: {}".format(acc_mat.sum() / (len(lr) * len(batch))))
'''
print("\n\nONE KERNEL (3X3) TEST ACCURACY MATRIX\n")
summary(NeuralNetwork_1_3().to("cpu"), (3,3,3))
acc_mat = multi("One_Kernel_(3X3)", NeuralNetwork_1_3, "one_kernel_3X3")
#print_mat(acc_mat, lr, batch)
print("AVERAGE ACC: {}".format(acc_mat.sum() / (len(lr) * len(batch))))
'''

print("\n\nTHREE KERNEL (3X3) TEST ACCURACY MATRIX\n")
summary(NeuralNetwork_3_3().to("cpu"), (3,3,3))
acc_mat = multi("Three_Kernel_(3X3)", NeuralNetwork_3_3, "three_kernel_3X3")
#print_mat(acc_mat, lr, batch)
print("AVERAGE ACC: {}".format(acc_mat.sum() / (len(lr) * len(batch))))


