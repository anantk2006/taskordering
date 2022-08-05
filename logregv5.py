import torch
from torch import nn
from math import sin, cos, pi
from torch.utils.data import DataLoader, Dataset
import time
from itertools import permutations as permute
import time
from math import factorial

DIM = 40
NUM_TASKS = 5
INC = 1/3 * pi
DSIZE = 150
GPU = 3

w_star = torch.rand(DIM)*2-1
span_ws = [torch.Tensor([1]*(DIM-2)+[sin(x*INC), cos(x*INC)]) for x in range(NUM_TASKS)]

features = []
i = 0
for w in span_ws:
    prev = len(features)
    while len(features)==prev:
        X = torch.cat([torch.zeros(DSIZE).unsqueeze(-1),torch.rand(DSIZE, DIM-1)*2-1,], dim = 1)
        last_col = X @ w.unsqueeze(-1) 
          
        X.mT[0] = -last_col.squeeze(-1)/w[0]
        large = torch.abs(X.mT[0])>1000        
        if any(large): continue
        features.append(X)
labels = [torch.where((X @ w_star.unsqueeze(-1))>0, 1, 0) for X in features]

class LRDataset(Dataset):
    def __init__(self, X, Y):
        '''
        Initializing Dataset to be placed inside DataLoader.         
        transform_matrix is matrix to be used to rotate all points created.
        X and Y and features and labels respectively
        '''
        self.X = X
        self.Y = Y
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    def __len__(self):
        return len(self.X)


class LogisticRegression(nn.Module):
    def __init__(self):
        #declare model, loss function, optimizer, initialize weights
        super().__init__()
        self.linear = nn.Linear(DIM, 1, bias = True)
        self.net = nn.Sequential(self.linear, nn.Sigmoid())
        self.loss_fn = nn.BCELoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr = 0.01)
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.constant_(m.weight, 1)
        self.net.apply(init_weights)

    def fit(self, dataset): 
        '''
        Train the model on dataset
        '''
        while (p:=self.test(dataset))<0.97:
            for features, labels in dataset:                
                self.optim.zero_grad()
                preds = self(features).unsqueeze(-1) #iterate through dataloader
                loss = self.loss_fn(preds.reshape(len(labels), 1), labels.type(torch.float32))
                loss.backward()
                self.optim.step()  
                
        self.coef_ = self.linear.weight.data #get the weights for comparison
        return p
    def forward(self, features):
        return self.net(features)

    def test(self, dataset):
        count = 0
        acc_agg = 0
        for feat, lab in dataset: 
             
            preds = self(feat).unsqueeze(-1) #find the accuracy on a certain give dataset
            preds = torch.where(preds>0.5, 1, 0)
            acc_agg += (preds.reshape(len(lab))==lab.squeeze(-1)).sum()
          
            count+=len(feat)
        
        return acc_agg/count   
device = torch.device(f"cuda:{GPU}")
all_data  = LRDataset(torch.cat(features, dim  = 0), torch.cat(labels, dim = 0))
all_data = DataLoader(all_data, shuffle = True, batch_size= 50)

model_star = LogisticRegression()
model_star.fit(all_data)
W_star = model_star.coef_
simils = torch.zeros(NUM_TASKS, NUM_TASKS)
coefs = torch.zeros(factorial(NUM_TASKS), DIM)

dataloaders = []
for feat, lab in zip(features, labels):
    dataloaders.append(DataLoader(LRDataset(feat.to(device), lab.to(device)), batch_size= 50, shuffle = True))
def get_accuracies(model, datasets):
    ret = []
    for dataset in datasets:
        ret.append(model.test(dataset))
    return [float(r) for r in ret]

dataloaders = list(permute(dataloaders, NUM_TASKS))
results = torch.zeros(factorial(NUM_TASKS), 4)

for ind, dataset in enumerate(dataloaders):
    
    model = LogisticRegression().to(device)
    permutation = list(permute(range(NUM_TASKS)))[ind]
    distance = 0
    init_time = time.time()
    for task_ind in range(NUM_TASKS):
        if task_ind>0: 
            distance+=abs(permutation[task_ind]*INC-permutation[task_ind-1]*INC)
            #print(model.coef_-W_star.to(device)) 
            print(get_accuracies(model, dataloaders[0])) 
        model.fit(dataset[task_ind])
    print(model.coef_-W_star.to(device))
    coefs[ind] = model.coef_    
    print(f"Permutation:\t\t {permutation}")
    print(f"Difference from X*:\t {(dx:=float(torch.linalg.norm(model.coef_ - W_star.to(device))))}")
    print(f"Distance traveled:\t {(dn:=float(distance))}")
    print(f"Accuracies:\t\t {(acc:=get_accuracies(model, dataloaders[0]))}")
    print(f"Avg. accuracy:\t\t {(a:=float(sum(acc)/NUM_TASKS))}")
    print(f"Time taken:\t\t {time.time()-init_time}\n\n")
    results[ind] = torch.Tensor([dx, dn, a, permutation[-1]])
torch.save(results, "lgrgresults/res1.pt")
torch.save(coefs, "lgrgresults/coefs1.pt")


        



