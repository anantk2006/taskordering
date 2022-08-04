from math import factorial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from itertools import permutations as permute
import time

CL_DIST = (0, 0.1)
DSIZE = 100
NUM_TASKS = 5
DIM = 7
GPU = 5



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
        self.optim = torch.optim.SGD(self.net.parameters(), lr = 0.3)
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, std=0.01)
        self.net.apply(init_weights)

    def fit(self, dataset): 
        '''
        Train the model on dataset
        '''
        while (p:=self.test(dataset))<1:
            for features, labels in dataset:                
                self.optim.zero_grad()
                preds = self(features).unsqueeze(-1) #iterate through dataloader
                loss = self.loss_fn(preds.reshape(len(labels)), labels.type(torch.float32))
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
            acc_agg += (preds.reshape(len(lab))==lab).sum()
            count+=len(feat)
        return acc_agg/count   

def get_orthogonal_vect(vect):
    random_vect = (torch.rand(DIM)-0.5)*12
    random_vect = random_vect - (torch.dot(random_vect, vect)/(torch.linalg.norm(vect)**2))*vect
    
    if all(torch.abs(random_vect)<5): return random_vect
    else: return None
a_means = torch.zeros(NUM_TASKS, DIM)
c = 0
x_star = (torch.rand(DIM)-0.5)*12#generate X_star such that the mean A doesn't have any absurdly large numbrs
while True:
    

    a_mean = get_orthogonal_vect(x_star)
    if a_mean is not None: 
        a_means[c] = a_mean
        c+=1
        
    if c==5: break

centers = [(a_mean + (x_star/torch.linalg.norm(x_star))*(torch.rand(1)*(CL_DIST[1]-CL_DIST[0])+CL_DIST[0]),
            a_mean - (x_star/torch.linalg.norm(x_star))*(torch.rand(1)*(CL_DIST[1]-CL_DIST[0])+CL_DIST[0])) for a_mean in a_means]

features = []
labels = []
for c_pos, c_neg in centers:
    c_pos = c_pos.repeat(DSIZE//2, 1); c_neg = c_neg.repeat(DSIZE//2, 1)
    c_pos = c_pos + torch.square(torch.rand(DSIZE//2, DIM))*CL_DIST[0]
    c_neg = c_neg + torch.square(torch.rand(DSIZE//2, DIM))*CL_DIST[0]
    task_feat = torch.cat([c_neg, c_pos], dim = 0)
    label = torch.cat([torch.ones(DSIZE//2), torch.zeros(DSIZE//2)], dim = 0)
    features.append(task_feat)
    labels.append(label)

device = torch.device(f"cuda:{GPU}")
all_data  = LRDataset(torch.cat(features, dim  = 0), torch.cat(labels, dim = 0))
all_data = DataLoader(all_data, shuffle = True, batch_size= 50)

model_star = LogisticRegression()
model_star.fit(all_data)
X_star = model_star.coef_
simils = torch.zeros(NUM_TASKS, NUM_TASKS)
for i, a_mean in enumerate(a_means):
    
    for j, b_mean in enumerate(a_means):
        
        simils[i][j] = torch.linalg.norm(a_mean - b_mean)
simil_sums = torch.sum(simils, dim = 0)

dataloaders = []
for feat, lab in zip(features, labels):
    dataloaders.append(DataLoader(LRDataset(feat.to(device), lab.to(device)), batch_size= 50, shuffle = True))
def get_accuracies(model, datasets):
    ret = []
    for dataset in datasets:
        ret.append(model.test(dataset))
    return [round(float(r),3) for r in ret]

dataloaders = list(permute(dataloaders, NUM_TASKS))
results = torch.zeros(factorial(NUM_TASKS), 4)

for ind, dataset in enumerate(dataloaders):
    
    model = LogisticRegression().to(device)
    permutation = list(permute(range(NUM_TASKS)))[ind]
    distance = 0
    init_time = time.time()
    for task_ind in range(NUM_TASKS):
        if task_ind>0: distance+=torch.linalg.norm(a_means[permutation[task_ind]]-a_means[permutation[task_ind-1]])        
        model.fit(dataset[task_ind])
    print(f"Permutation:\t\t {permutation}")
    print(f"Difference from X*:\t {(round(dx:=float(torch.linalg.norm(model.coef_ - X_star.to(device))), 2))}")
    print(f"Distance traveled:\t {(round(dn:=float(distance), 2))}")
    print(f"Accuracies:\t\t {(acc:=get_accuracies(model, dataloaders[0]))}")
    print(f"Avg. accuracy:\t\t {round(a:=float(sum(acc)/NUM_TASKS), 3)}")
    print(f"Time taken:\t\t {time.time()-init_time}\n\n")
    results[ind] = torch.Tensor([dx, dn, a, simil_sums[permutation[-1]]])
torch.save(results, "lgrgresults/res1.pt")




