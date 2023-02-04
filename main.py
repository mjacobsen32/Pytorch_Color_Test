from torchsummary import summary

from run import run, multi
from dataset import CustomImageDataset as Dataset
from NN_1_3 import NeuralNetwork_1_3
from NN_3_3 import NeuralNetwork_3_3
from NN_3_1 import NeuralNetwork_3_1
from NN_1_1 import NeuralNetwork_1_1


'''
    Primary driving file.
    multi runs multiple passes through a newly initalized model and 
    saves all the accuracies to acc_list which prints them out
'''
        
### Hyper parameters for the model creation
### scheduler, optimization and loss function can be specified
### in the run.py file if desired

ds = Dataset(10000) 
lr = 0.01
batch = 8
epochs = 10
runs = 100
save_perfect_acc_models = False

print("\n\nTHREE KERNEL (3X3) TEST ACCURACY\n")
summary(NeuralNetwork_3_3().to("cpu"), (3,3,3))
acc_list = multi("Three_Kernel_(3X3)", NeuralNetwork_3_3, "three_kernel_3X3", runs,
                save_perfect_acc_models, batch, lr, epochs, ds)
print("AVERAGE ACC: {}".format(acc_mat.sum() / runs))
      
print("\n\nONE KERNEL (3X3) TEST ACCURACY\n")
summary(NeuralNetwork_1_3().to("cpu"), (3,3,3))
acc_list = multi("One_Kernel_(3X3)", NeuralNetwork_1_3, "one_kernel_3X3", runs,
                save_perfect_acc_models, batch, lr, epochs, ds)
print("AVERAGE ACC: {}".format(acc_mat.sum() / runs))


