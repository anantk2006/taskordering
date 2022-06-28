
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from itertools import permutations as permute
from models import MLP, SimpleCNN
from torch.utils.data import DataLoader
from math import factorial
from time import time as process_time
import argparse
import os, sys


parser = argparse.ArgumentParser(description = "modifies task parameters")
parser.add_argument("--inc", dest = "increment", default = 30, type = float)
parser.add_argument("--tasks", dest = "num_tasks", default = 4, type = int)
parser.add_argument("--start", dest = "start", default = 0, type = int)
parser.add_argument("--end", dest = "end", default = -1, type = int)
parser.add_argument("--samples", dest = "samples", default=  1, type = int)
parser.add_argument("--data", dest = "dataset", default = "rot")
parser.add_argument("--gpu", dest = "gpuid", default = "0", type = str)

args = parser.parse_args()
torch.manual_seed(0)
#torch.use_deterministic_algorithms(True)
# args.increment = int(args.increment)
# args.num_tasks = int(args.num_tasks)
# args.start = int(args.start)
# args.end = int(args.end)
sys.stdout = open(f"outputfiles/rotatedcifar{args.start}to{args.end}", "w")

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
        return transforms.functional.adjust_brightness(X, self.bright)
if args.dataset == "noise":
    trans = Noise
    task_range = range(args.num_tasks)
elif args.dataset == "bright":
    task_range = range(1, args.num_tasks+1)
    trans = Brightness
else:
    trans = Rotation
    task_range = range(args.num_tasks)


train_data = [((i*args.increment), DataLoader(CIFAR10(root = "data", train = True, download = True, 
        transform = transforms.Compose([transforms.ToTensor(), trans(i*args.increment)])), batch_size = 100, num_workers = 6)) 
 for i in task_range]
test_data = [((i*args.increment), DataLoader(CIFAR10(root = "data", train = False, download = True, 
        transform = transforms.Compose([transforms.ToTensor(), trans(i*args.increment)])), batch_size = 100, num_workers = 6)) 
        for i in task_range]

train_data_permutes = list(permute(train_data))
if args.start>0: 
    if args.end>0:
        train_data_permutes = train_data_permutes[args.start:args.end]
    else: train_data_permutes = train_data_permutes[args.start:]
elif args.end>0: train_data_permutes = train_data_permutes[:args.end]

data_tensor = torch.zeros(args.samples, len(train_data_permutes), 2, args.num_tasks+1, args.num_tasks)

device = torch.device("cuda:"+args.gpuid)
# for data in train_data_permutes[0]:
#     for X, Y in data[1]:
        
#         X = X.to(device); Y.to(device)
    
# for data in test_data:
#     for X, Y in data[1]:
        
        
#cpu = torch.device("cpu")

def train(model, data, loss_fn, optim):
    
    model.train()
    for _ in range(50):
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
            optim.step()
            #num_corr += torch.sum(torch.argmax(Y_hat, dim = 1)==Y)
        
           

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
            if num_done>4: break
            num_done += 1
            X = X.to(device); Y = Y.to(device)
            Y_hat = model(X)
            loss_curr = loss_fn(Y_hat, Y)
            loss_agg += float(loss_curr)
            num_correct = torch.sum(torch.argmax(Y_hat, dim = 1)==Y)
            acc_agg += num_correct
            
        acc[ind] = acc_agg/(500)
        loss[ind] = loss_agg/(500)
    #print(process_time()-tim)
    return acc, loss

        



loss_fn = torch.nn.CrossEntropyLoss()
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
for i in range(args.samples):
  for ind_perm, tr_data in enumerate(train_data_permutes):
    model = SimpleCNN(num_channels = 3, num_classes = 10).to(device)


    
    model.apply(init_weights)
    optim = torch.optim.SGD(model.parameters(), lr = 0.25)
    for ind_task, (inc, data) in enumerate(tr_data):
        init_time = process_time()
        train(model, data, loss_fn, optim)
        acc, loss = test_results(model, test_data, loss_fn)
        print(f"Task number: {ind_task}      Task increment: {inc}      What permutation: {ind_perm}" )
        print(f"Accuracies: {acc}      Losses: {loss}")
        print(f"Time taken: {round(process_time()-init_time, 2)}")
        data_tensor[i][ind_perm][0][ind_task][:] = acc
        data_tensor[i][ind_perm][1][ind_task][:] = loss
        data_tensor[i][ind_perm][0][-1][ind_task] = inc
torch.save(data_tensor, f"rotated_cifar_results/{args.num_tasks}_{args.increment}_{args.start}_{args.end}_{args.samples}.pt")


    


    







