
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from itertools import permutations as permute
from models import MLP, SimpleCNN
from torch.utils.data import DataLoader
from math import factorial
from time import time as process_time
import argparse
import os, sys
import numpy
import random
#torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser(description = "modifies task parameters")
parser.add_argument("--inc", dest = "increment", default = 30, type = float)
parser.add_argument("--tasks", dest = "num_tasks", default = 4, type = int)
parser.add_argument("--index", dest = "index", default = 0, type = int)

#parser.add_argument("--samples", dest = "samples", default=  1, type = int)
parser.add_argument("--mod", dest = "transform", default = "rot")
parser.add_argument("--gpu", dest = "gpuid", default = "0", type = str)
parser.add_argument("--seed", dest = "seed", default = 0, type = int)
parser.add_argument("--print", dest = "print", default = "term", type = str)
parser.add_argument("--data", dest = "dataset", default = "cifar", type = str)
#parser.add_argument("--epochs", dest = "num_epochs", default = 35, type = int)

args = parser.parse_args()
numpy.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(True)
#torch.manual_seed(args.seed)
#torch.use_deterministic_algorithms(True)
# args.increment = int(args.increment)
# args.num_tasks = int(args.num_tasks)
# args.start = int(args.start)
# args.end = int(args.end)
if args.print == "file":
    sys.stdout = open(f"cifar_outputs/{args.dataset}_{args.transform}_{args.num_tasks}_{int(args.increment)}_{args.index}.txt", "w")

class Noise:
    def __init__(self, noise):
        
        self.noise = torch.zeros(size = (3,32,32))+noise
        
        
    def __call__(self, X):

        
        #noisy_X = torch.uniform(X-noise, X+noise)
        #if torch.sum(noisy_X - X)>0.1: print("x")
        return torch.normal(X, self.noise)
class Rotation:
    def __init__(self, angle):
        self.angle = angle
        
    def __call__(self, X):
        return transforms.functional.rotate(X, self.angle)
class Brightness:
    def __init__(self, bright):
        self.bright = bright
    def __call__(self, X):
        return transforms.functional.adjust_brightness(X, self.bright+0.4)

if args.transform == "noise":
    trans = Noise
    task_range = range(args.num_tasks)
elif args.transform == "bright":
    task_range = range(args.num_tasks)
    trans = Brightness
else:
    trans = Rotation
    task_range = range(args.num_tasks)
if args.dataset == "cifar":
    data_class = CIFAR10
    num_channels = 3
    
else: 
    data_class = MNIST
    num_channels = 1


   

train_data = [((i*args.increment), DataLoader(data_class(root = "data", train = True, download = True, 
        transform = transforms.Compose([transforms.ToTensor(), trans(i*args.increment)])), batch_size = 100, num_workers = 6, shuffle = True))
 for i in task_range]
test_data = [((i*args.increment), DataLoader(data_class(root = "data", train = False, download = True, 
        transform = transforms.Compose([transforms.ToTensor(), trans(i*args.increment)])), batch_size = 100, num_workers = 6, shuffle = True))
        for i in task_range]

train_data_permutes = list(permute(train_data))
train_data_permutes = train_data_permutes[args.index]

data_tensor = torch.zeros(2, args.num_tasks+1, args.num_tasks)

device = torch.device("cuda:"+args.gpuid)
# for data in train_data_permutes[0]:
#     for X, Y in data[1]:
        
#         X = X.to(device); Y.to(device)
    
# for data in test_data:
#     for X, Y in data[1]:
        
        
#cpu = torch.device("cpu")

def train(model, data, loss_fn, optim):
        
    model.train()
    loss_agg = 2.5
    loss_num = 1
    i = 0
    while loss_agg/loss_num>1:
        #num_corr = 0
        #ftime = process_time()
        for ind, (X, Y) in enumerate(data):
            
        #    print(process_time()-ftime)
        #    ftime = process_time()
            
            X = X.to(device); Y = Y.to(device)
            Y_hat = model(X)
            loss = loss_fn(Y_hat, Y)
            optim.zero_grad()
            loss.backward()
            l = float(loss)
            loss_agg+=l
            loss_num+=1


            optim.step()
            #num_corr += torch.sum(torch.argmax(Y_hat, dim = 1)==Y)
        loss_agg, loss_num = loss_agg/loss_num, 1 
        i+=1
    print(i, loss_agg)       
           

cpu = torch.device("cpu")
def test_results(model, test_data, loss_fn):
    model.eval()
    tim = process_time()
    acc = torch.zeros(size  = (args.num_tasks,), device = device)
    loss = torch.zeros(size = (args.num_tasks,), device = device)
    for ind, (_, te_data) in enumerate(test_data):
        loss_agg = 0
        acc_agg = 0
        num_done = 0
        for num_batches, (X, Y) in enumerate(te_data):
            if num_done>9: break
            num_done += 1
            X = X.to(device); Y = Y.to(device)
            Y_hat = model(X)
            loss_curr = loss_fn(Y_hat, Y)
            loss_agg += float(loss_curr)
            num_correct = torch.sum(torch.argmax(Y_hat, dim = 1)==Y)
            acc_agg += num_correct
            
        acc[ind] = acc_agg/(1000)
        loss[ind] = loss_agg/(1000)
    #print(process_time()-tim)
    return acc, loss

        



loss_fn = torch.nn.CrossEntropyLoss()
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

  
model = SimpleCNN(num_channels = num_channels, num_classes = 10).to(device)   
model.apply(init_weights)
optim = torch.optim.SGD(model.parameters(), lr = 0.1)
for ind_task, (inc, data) in enumerate(train_data_permutes):
    init_time = process_time()
    train(model, data, loss_fn, optim)
    acc, loss = test_results(model, test_data, loss_fn)
    print(f"Task number: {ind_task}      Task increment: {inc}" )
    print(f"Accuracies: {acc}      Losses: {loss}")
    print(f"Time taken: {round(process_time()-init_time, 2)}")
    data_tensor[0][ind_task][:] = acc
    data_tensor[1][ind_task][:] = loss
    data_tensor[0][-1][ind_task] = inc
torch.save(data_tensor, f"cifar_results/{args.dataset}_{args.transform}_{args.num_tasks}_{int(args.increment)}_{args.index}_{args.seed}.pt")


    


    







