import torch
from my_utils import print_mat, print_model_weights, visTensor
from model_funcs import train, test


'''
    Run individual training and testing pass on the model
'''
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


'''
    multi:
    mod_str: string name of model
    net: model object
    f_string: file string to save the kernel png
    runs: number of times to run the exact same test with new initalizations
    save_kernels: bool to save or not save 100% accurate kernels
'''
def multi(mod_str, net, f_string, runs, save_kernels, batch, lr, epochs, ds):
    stored_accs= []
    counter = 0
    for i in range(runs):
        train_set, val_set, _ = torch.utils.data.random_split(dataset=ds, lengths=[8500, 1500, 0]) 
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch[0])
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch[0])
        
        torch.seed()
        model = net().to("cpu")
        
        ###
        ### More hyper parameters for the model, not specified in main.py
        ###
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr[0])
        sched = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
        
        stored_accs.append(run(lr=lr, batch=batch, mod_str=mod_str, model=model, loss_fn=loss_fn, optimizer=optimizer, 
                                    sched=sched, train_loader=train_loader, val_loader=val_loader, epochs=epochs))
        if save_kernels and stored_accs[-1] == 1.0:
            fname = mod_str + "_" + str(counter)
            torch.save(model.state_dict(), "./models/"+fname+"_"+".pt")
            filter = model.features[0].weight.data.clone()
            visTensor(filter, ch=0, allkernels=True, fname=fname)
            counter+=1
    return stored_accs