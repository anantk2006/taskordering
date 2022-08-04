import numpy as np
from math import exp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import itertools
import argparse
import time
from math import factorial
parser = argparse.ArgumentParser(description="using continual learning for logistic regression to do theoretical study of \ntaskordering")
parser.add_argument("--tasks", dest = "num_tasks", type = int, default = 5, help = "number of tasks to do continual learning on")
parser.add_argument("--dataset-size", dest = "dsize", type = int, default = 50, help = "number of points in each dataset")
parser.add_argument("--dim", dest = "pdim", default = 7, type = int, help = "dimension of feature vectors")
parser.add_argument("--lr", dest = "lr", type = float, default = 0.3, help = "learning rate to use during training")
args = parser.parse_args()

class LRDataset(Dataset):
    def __init__(self, features, labels):
       
        self.features = features
        self.labels = labels
    def __getitem__(self, index):
        
        return self.features[index], self.labels[index]
    def __len__(self):
        return len(self.features)

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(args.pdim, 1, bias = False)
        self.net = nn.Sequential(self.linear, nn.Sigmoid())
        self.loss_fn = nn.BCELoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr = args.lr)
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, std=0.01)

        self.net.apply(init_weights)


    def fit(self, dataset):  
        acc = 0
        while acc/args.dsize<1:
            acc = 0
            for features, labels in dataset:            
                self.optim.zero_grad()
                preds = self(features).unsqueeze(-1)
                rounded_preds = torch.round(preds)
                acc += torch.sum(rounded_preds.reshape(len(labels))==labels)
                
                loss = self.loss_fn(preds.reshape(len(labels)), labels)
                loss.backward()
                self.optim.step()  
            
            
        self.coef_ = self.linear.weight.data
    def forward(self, features):
        return self.net(features)
    def test(self, dataset):
        c = 0
        acc_agg = 0
        for feat, lab in dataset:
           
            c+=len(feat)
            preds = self(feat).unsqueeze(-1)
            
            preds = torch.where(preds>0.5, 1, 0)
            
        #print(preds)
            acc_agg += (preds.reshape(len(lab))==lab).sum()
        return acc_agg/c
    




features = []
def get_orthogonal_vect(vect):
    random_vect = (torch.rand(args.pdim)-0.5)*12
    random_vect = random_vect - (torch.dot(random_vect, vect)/(torch.linalg.norm(vect)**2))*vect
    
    if all(torch.abs(random_vect)<5): return random_vect
    else: return None
a_means = torch.zeros(args.num_tasks, args.pdim)
c = 0
x_star = (torch.rand(args.pdim)-0.5)*12
while True:
    #generate X_star such that the mean A doesn't have any absurdly large numbrs

    a_mean = get_orthogonal_vect(x_star)
    if a_mean is not None: 
        a_means[c] = a_mean
        c+=1
        
    if c==5: break



features = a_means.unsqueeze(1).repeat(1, args.dsize, 1)
noise = torch.zeros_like(features)+1.5
features = torch.normal(features, noise)


    





labels = torch.zeros((args.num_tasks, args.dsize))

for i, task in enumerate(features):
    for j, sample in enumerate(task):
        
        P = nn.Sigmoid()(torch.dot(sample, x_star))
        
        labels[i][j] = 1 if P>0.5 else 0




all_data  = LRDataset(features.reshape(args.dsize*args.num_tasks, args.pdim), labels.reshape(args.dsize*args.num_tasks))
all_data = DataLoader(all_data, shuffle = True, batch_size= 50)

model_star = LogisticRegression()
model_star.fit(all_data)
X_star = model_star.coef_
simils = torch.zeros(args.num_tasks, args.num_tasks)
for i, a_mean in enumerate(a_means):
    
    for j, b_mean in enumerate(a_means):
        
        simils[i][j] = torch.linalg.norm(a_mean - b_mean)
simil_sums = torch.sum(simils, dim = 0)
device = torch.device("cuda:0")

dataset = [(a_means[i], LRDataset(f.to(device), l.to(device))) for i, (f,l) in enumerate(zip(features, labels))]
dataset = [(a_mean, DataLoader(i, shuffle=True, batch_size=50)) for a_mean, i in dataset]
datasets = itertools.permutations(dataset, args.num_tasks)

def get_accuracies(model, datasets):
    ret = []
    for _, dataset in datasets:
        ret.append(model.test(dataset))
    return [round(float(r),3) for r in ret]
datasets = list(datasets)
results = torch.zeros(size = (factorial(args.num_tasks), 4))
for ind, dataset in enumerate(datasets):
    
    model = LogisticRegression().to(device)
    permutation = list(itertools.permutations(range(args.num_tasks)))[ind]
    distance = 0
    init_time = time.time()
    for task_ind in range(args.num_tasks):
        if task_ind>0: distance+=torch.linalg.norm(dataset[task_ind-1][0]-dataset[task_ind][0])        
        model.fit(dataset[task_ind][1])

    
    print(f"Permutation:\t\t {permutation}")
    print(f"Difference from X*:\t {(round(dx:=float(torch.linalg.norm(model.coef_ - X_star.to(device))), 2))}")
    print(f"Distance traveled:\t {(round(dn:=float(distance), 2))}")
    print(f"Accuracies:\t\t {(acc:=get_accuracies(model, datasets[0]))}")
    print(f"Avg. accuracy:\t\t {round(a:=float(sum(acc)/args.num_tasks), 3)}")
    print(f"Time taken:\t\t {time.time()-init_time}\n\n")
    results[ind] = torch.Tensor([dx, dn, a, simil_sums[permutation[-1]]])
torch.save(results, "lgrgresults/res1.pt")
        



    
    
    

    













        