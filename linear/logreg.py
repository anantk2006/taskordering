import torch
from torch import nn
from math import sin, cos, pi
from torch.utils.data import DataLoader, Dataset
import time
from itertools import permutations as permute
import time
from math import factorial, sqrt
import random, numpy
from scipy.linalg import orth
import scipy
import sys; args = sys.argv[1:]

DIM = 501
NUM_TASKS = 8
INC = 1/18* pi
DSIZE = 500
GPU = 3
SEED = int(args[0])+10
FUNC = "log"
CYCLES = 1
ZEROS = 4
lr = 0.5 if FUNC == "lin" else 0.2
samps = 250
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)
numpy.random.seed(SEED)
ang_bet  = lambda a, b: torch.dot(a,b)/(torch.linalg.norm(b)*torch.linalg.norm(a))
proj = lambda a, b: (torch.dot(a, b)/torch.dot(b, b))*b


def get_matrix(N):
    while True :
        Phi = numpy.random.randn(N, N).astype(numpy.float32)
        g = orth(Phi)
        if (g.shape[0], g.shape[1])==(N, N):
            return torch.from_numpy(g)
    

def rot(w):
    while True:
        M = get_matrix(w.shape[0])
        return (M @ w)

span_ws = torch.zeros(NUM_TASKS, DIM)
span_ws[0] = get_matrix(DIM)@torch.Tensor([1]+[0]*(DIM-1))
for i in range(NUM_TASKS-1):
    span_ws[i+1] = get_matrix(DIM)@span_ws[i] 
    


features = []
i = 0
error = 0.1
for ind, w in enumerate(span_ws):
    prev = len(features)
   
    X = torch.zeros((DSIZE, DIM))
    e_sum = 0
    for i in range(DSIZE):
        
        nums = list(range(i+1, DIM))
        
        X[i][i] = 1
        while len(nums)>1:
            n = random.choice(nums)
            nums.remove(n)
            
            c_sum = torch.dot(X[i], w)
            
            
            errors = torch.Tensor([error-c_sum, -c_sum-error])/w[n]
            e_sum += errors[0]-errors[1]
            zero = -c_sum/w[n]      
            
            if abs(errors[1])>3:
                errors[1] = 3*(errors[1]/abs(errors[1]))
            if abs(errors[0])>3:
                errors[0] = 3*(errors[0]/abs(errors[0]))
            
            X[i][n] = random.random()*(errors[0]-errors[1])+errors[1]
            

        X[i][n] = -torch.dot(X[i], w)/w[n]

    O = get_matrix(DSIZE)
    X = O@X
    features.append(X)

def ortho(W_star, ws):
    for w in orth(ws.T).T:
        w = torch.Tensor(w)
        W_star = W_star - torch.dot(W_star, w)/(torch.dot(w,w))*w
    return W_star
while True:
    W_star = torch.rand(DIM)*2-1
    if FUNC == "log": labels = [torch.where((X @ W_star.unsqueeze(-1))>0, 1, 0) for X in features]
    else: labels = [X @ W_star.unsqueeze(-1) for X in features]

    if FUNC == "lin": break

    for label in labels:
        if label.sum()>DSIZE//3 or label.sum()<DSIZE/1.5: continue
    break



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


class Regression(nn.Module):
    def __init__(self):
        #declare model, loss function, optimizer, initialize weights
        super().__init__()
        self.linear = nn.Linear(DIM, 1, bias = False)
        self.net = nn.Sequential(self.linear, nn.Sigmoid()) if FUNC == "log" else nn.Sequential(self.linear)
        self.loss_fn = nn.BCELoss() if FUNC == "log" else nn.MSELoss()
        
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.constant_(m.weight, 0)
        
        self.net.apply(init_weights)

    def fit(self, dataset): 
        '''
        Train the model on dataset
        '''
        
        self.optim = torch.optim.SGD(self.net.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lambda epoch: lr/sqrt(epoch+1))
        t = time.time()
        
        while ((p:=self.test(dataset)[1])>0.01) if FUNC == "lin" else (self.test(dataset)[0]<1):
            
            for features, labels in dataset:   
             
                self.optim.zero_grad()
                preds = self(features).unsqueeze(-1) #predict using model
                loss = self.loss_fn(preds.flatten(), labels.flatten().to(torch.float32)) 
                
                if time.time()-t>60:
                    if time.time()-t>62:
                        exit()
                    with open("debug_loss.txt", "a+") as f:                    
                        if round(loss.item(),6)%1 == 0:
                            f.write(str(loss.item())+"\n")
                            f.write("f"+str(features)+"\n")
                            f.write("w"+str(self.linear.weight.data)+"\n")
                            f.write(str(torch.mm(features, self.linear.weight.data.reshape(DIM, 1)))+"\n")
                            f.write(str(p) + "\n")
                
                loss.backward()
                self.optim.step() # backpropagation
             
            self.scheduler.step()
           
        self.coef_ = self.linear.weight.data #get the weights for comparison
        
        return 0

    def forward(self, features):
        return self.net(features)

    def acc_calc(self, preds, labels):
        diffs = abs(preds-labels)
        g = torch.where(diffs<1, 1, 0) 
        return g.sum().item()

    def test(self, dataset):
        '''
        Tests the code on given dataset, only one task
        '''
        count = 0
        acc_agg = 0
        loss = 0
        for feat, lab in dataset: 
            ##print(lab)
            preds = self(feat).unsqueeze(-1) #find the accuracy on a certain give dataset
            pr = torch.where(preds>0.5, 1, 0)
            ##print(preds.reshape(len(lab))-lab.reshape(len(lab)))
            
            if FUNC == "lin": acc_agg += self.acc_calc(preds, lab)
            else: acc_agg += (pr.reshape(len(lab))==lab.squeeze(-1)).sum()
            
            loss += torch.linalg.norm(preds.reshape(len(lab))- lab.reshape(len(lab)))**2
            count+=len(feat)
            
       
        if FUNC == "log": return acc_agg/count, loss/count   
        else: return acc_agg/count, loss
        




simils = torch.zeros(NUM_TASKS, NUM_TASKS)
coefs = torch.zeros(factorial(NUM_TASKS), DIM)

dataloaders = []
for feat, lab in zip(features, labels):
    dataloaders.append(DataLoader(LRDataset(feat.to(device), lab.to(device)), batch_size= 50, shuffle = True))
def get_accuracies(model, datasets):
    ret_a = []; ret_l = []
    for dataset in datasets:
        ret_a.append((t:=model.test(dataset))[0])
        ret_l.append(t[1])
    return [float(r) for r in ret_a], ret_l

def get_distances(w, w_s):
    if torch.dot(w, w_s)>0:
        return torch.linalg.norm(w - (torch.dot(w, w_s)/torch.dot(w_s, w_s))*w_s)
    else: return torch.linalg.norm(w)


dataloaders = list(permute(dataloaders, NUM_TASKS))
results = torch.zeros(samps, 4)
w_distances = torch.zeros(samps, NUM_TASKS)

W_star = W_star.to(device)

losses = torch.zeros(samps, NUM_TASKS+1, NUM_TASKS) 
for s, ind in enumerate(torch.randint(0, factorial(NUM_TASKS), (250,))):
    dataset = dataloaders[ind]
    model = Regression().to(device)
    permutation = list(permute(range(NUM_TASKS)))[ind]
    distance = 0
    init_time = time.process_time()
    c = -1
    
    for task_ind in range(NUM_TASKS):
        if task_ind>0: 
            #place to do any and all testing calculations
            distance+=abs(permutation[task_ind]*INC-permutation[task_ind-1]*INC)
            c = model.coef_.detach().clone()
            
            c = c.flatten()
            W_star = W_star.flatten() 
            w_distances[s][task_ind-1] = get_distances(c, W_star)
            wi = span_ws[task_ind-1].to(device)
           
            lambda1 = (torch.dot(c, W_star)/torch.dot(W_star, W_star))
            
           
          
        model.fit(dataset[task_ind])
        

        c = model.coef_.flatten() 

        losses[s][task_ind] = torch.Tensor((g:=get_accuracies(model, dataloaders[0]))[1])
        print(g[0]) #print accuracies to gauge everything

    losses[s][-1] = torch.Tensor(permutation)        
    
    
    acc, loss = get_accuracies(model, dataset)
    coefs[s] = model.coef_  
    dx=float(get_distances(model.coef_.squeeze(0), W_star))
    a=float(sum(acc)/NUM_TASKS)
    dn=float(distance)
    print(f"Permutation:\t\t {permutation}")
    print(f"Difference from X*:\t {(dx:=float(get_distances(model.coef_.squeeze(0), W_star)))}")
    print(f"Distance traveled:\t {(dn:=float(distance))}")
    
    print(f"Accuracies:\t\t {(acc)}")
    print(f"Avg. accuracy:\t\t {(a:=float(sum(acc)/NUM_TASKS))}")
    print(f"Time taken:\t\t {time.process_time()-init_time}\n\n")
    results[s] = torch.Tensor([dx, dn, a, permutation[-1]])
    
torch.save(results, f"lgrgresults/res{SEED}30.pt")
torch.save(coefs, f"lgrgresults/coefs{SEED}30.pt")
torch.save(w_distances, f"lgrgresults/w_distances{SEED}30.pt")
torch.save(losses, f"lgrgresults/losses{SEED}30.pt")
torch.save(span_ws, f"lgrgresults/span_ws{SEED}30.pt")


        



