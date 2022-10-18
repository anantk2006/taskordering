import torch
import torchvision
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
from strategies import Naive, Replay


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser(description = "modifies task parameters")
parser.add_argument("--inc", dest = "increment", default = 30, type = float, help = "the level of difference between tasks")
parser.add_argument("--tasks", dest = "num_tasks", default = 4, type = int, help = "number of tasks")
parser.add_argument("--index", dest = "index", default = 0, type = int, help = "index of permutations i.e. 0-120 for 5 tasks")
parser.add_argument("--mod", dest = "transform", default = "rot", help = "the type of transform, rotation, noise, or brightness")

parser.add_argument("--gpu", dest = "gpuid", default = "0", type = str, help = "gpu id to be used")
parser.add_argument("--seed", dest = "seed", default = 0, type = int, help = "seeding for the program for reproducibility")
parser.add_argument("--print", dest = "print", default = "term", type = str, help = "where to print")
parser.add_argument("--data", dest = "dataset", default = "cifar", type = str, help = "dataset to be used")

parser.add_argument("--model", dest=  "model", default = "resnet",help = "CNN to be used")
parser.add_argument("--train-until", dest = "loop", default = "epochs", help = "how should training be terminated")
parser.add_argument("--epochs", dest = "epochs", default = 30, type = int, help="number of epochs")
parser.add_argument("--lr", dest = "lr", default=0.1, type = float, help = "learning rate")
parser.add_argument("--loss-thres", dest = "dloss", default = 0.1, type = float, help = "threshold if training until loss")

parser.add_argument("--strat", dest = "strat", default = "naive", type = str, help = "continual learning strategy")
parser.add_argument("--num-examples", dest = "num_examples", type = int, default = 1000, help = "number of training pairs to be stored")

args = parser.parse_args()
numpy.random.seed(args.seed)
random.seed(args.seed) #seeding for reproducibility
torch.manual_seed(args.seed)

if args.print == "file": #print to file if desired
    sys.stdout = open(f"outputs/{args.dataset}_{args.transform}_{args.num_tasks}_{int(args.increment)}_{args.index}.txt", "w")

class Noise:
    def __init__(self, noise):        
        self.noise = torch.zeros(size = (3,32,32) if args.dataset=="cifar" else (1,28,28))+noise       
    def __call__(self, X):
        return torch.normal(X, self.noise)
class Rotation:
    def __init__(self, angle):
        self.angle = angle        # custom transforms to augment dataset for continual learning
    def __call__(self, X):
        return transforms.functional.rotate(X, self.angle)
class Brightness:
    def __init__(self, bright):
        self.bright = bright
    def __call__(self, X):
        return transforms.functional.adjust_brightness(X, self.bright+0.4)


class RepeatChannels:
    def __init__(self, num_reps):
        self.num_reps = num_reps
    def __call__(self, X): #in order to reformat images to CNN specifications for resnet
        return X.repeat(self.num_reps, 1, 1)

if args.transform == "noise":
    trans = Noise
    
elif args.transform == "bright":
    
    trans = Brightness
else:    #uses cmd line to choose
    trans = Rotation

    
if args.dataset == "cifar":
    data_class = CIFAR10
    num_channels = 3
    
else: 
    data_class = MNIST
    num_channels = 1

VARS = {"transforms": {"cifar": {"resnet": [transforms.Resize(224)], "simple": []},
                       "mnist": {"resnet": [RepeatChannels(3), transforms.Resize(224)], "simple": []}}}



#deciding all transforms and loading the data

train_data = [((i*args.increment), DataLoader(data_class(root = "data", train = True, download = True, 
        transform = transforms.Compose([transforms.ToTensor(), trans(i*args.increment), *VARS["transforms"][args.dataset][args.model]])), batch_size = 100, shuffle = True, num_workers = 6))
 for i in range(args.num_tasks)]
test_data = [((i*args.increment), DataLoader(data_class(root = "data", train = False, download = True, 
        transform = transforms.Compose([transforms.ToTensor(), trans(i*args.increment), *VARS["transforms"][args.dataset][args.model] ])), batch_size = 100, shuffle = True, num_workers = 6))
 for i in range(args.num_tasks)]

train_data_permutes = list(permute(train_data)) #permutation to be used
train_data_permutes = train_data_permutes[args.index]

data_tensor = torch.zeros(2, args.num_tasks+1, args.num_tasks) #results tensor to be saved

if torch.cuda.is_available(): device = torch.device("cuda:"+str(int(args.gpuid)))
else: device = torch.device("cpu")




        



loss_fn = torch.nn.CrossEntropyLoss()
def init_weights(m):
    if isinstance(m, torch.nn.Linear): 
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def test_results(model, test_data, loss_fn, args, device):
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
            loss_curr = loss_fn(Y_hat, Y) #testing procedure, measures both accuracy and loss for all tasks
            loss_agg += float(loss_curr)
            num_correct = torch.sum(torch.argmax(Y_hat, dim = 1)==Y)
            acc_agg += num_correct
            
        acc[ind] = acc_agg/(1000)
        loss[ind] = loss_agg/(1000)
    #print(process_time()-tim)
    return acc, loss

  #initialization of everything

if args.model == "resnet":
    model = torchvision.models.resnet18(pretrained=False).to(device)
if args.model == "simple":
    model = SimpleCNN(num_channels = num_channels, num_classes = 10).to(device)

if args.strat == "naive":
    strategy = Naive()
else: strategy = Replay(args.num_examples, args)
model.apply(init_weights)
optim = torch.optim.SGD(model.parameters(), lr = args.lr)
for ind_task, (inc, data) in enumerate(train_data_permutes): #trains model on every task
    init_time = process_time()
    strategy.train(model, data, loss_fn, optim, args, device)
    acc, loss = test_results(model, test_data, loss_fn, args, device)
    
    print(f"Task number: {ind_task}      Task increment: {inc}" )
    print(f"Accuracies: {acc}      Losses: {loss}")
    print(f"Time taken: {round(process_time()-init_time, 2)}")
    data_tensor[0][ind_task][:] = acc
    data_tensor[1][ind_task][:] = loss
    data_tensor[0][-1][ind_task] = inc
torch.save(data_tensor, f"mnist_results/{args.dataset}_{args.transform}_{args.num_tasks}_{int(args.increment)}_{args.index}_{args.seed}_{args.model}.pt")


    


    







