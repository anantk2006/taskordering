import torch
from torch import nn
from math import sin, cos, pi
from torch.utils.data import DataLoader, Dataset
import time
from itertools import permutations as permute
import time
from math import factorial, sqrt
import random, numpy
import scipy
from scipy.linalg import orth
import sys; args = sys.argv[1:]

DIM = 501
NUM_TASKS = 5
INC = 1/18* pi
DSIZE = 500
GPU = 3
SEED = int(args[0])+10
FUNC = "log"
CYCLES = 1
ZEROS = 4
lr = 0.5 if FUNC == "lin" else 0.5
samps = 0
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)
numpy.random.seed(SEED)
ang_bet  = lambda a, b: torch.dot(a,b)/(torch.linalg.norm(b)*torch.linalg.norm(a))
proj = lambda a, b: (torch.dot(a, b)/torch.dot(b, b))*b

datagen = "good" # Choices: ["general", "good"]


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

# Data generation.
if datagen == "general":
    if samps == 0:
        span_ws = torch.Tensor([[0]*(DIM-2)+[sin((x+2.5)*INC), cos((x+2.5)*INC)] for x in range(NUM_TASKS)])
        span_ws = rot(span_ws.T).T
    else:
        span_ws = torch.zeros(NUM_TASKS, DIM)
        span_ws[0] = get_matrix(DIM)@torch.Tensor([1]+[0]*(DIM-1))
        for i in range(NUM_TASKS-1):
            span_ws[i+1] = get_matrix(DIM)@span_ws[i] 
elif datagen == "good":
    span_ws = torch.Tensor([[0]*(DIM-2)+[sin((x+2.5)*INC) if abs(sin((x+2.5)*INC))>0.01 else 0, cos((x+2.5)*INC) if cos((x+2.5)*INC)>0.01 else 0] for x in range(NUM_TASKS)])
    span_ws = rot(span_ws.T).T

else:
    raise NotImplementedError

features = []
i = 0
error = 0.25
for ind, w in enumerate(span_ws):
    prev = len(features)

    X = torch.zeros((DSIZE, DIM))
    e_sum = 0
    for i in range(DSIZE):
        
        if datagen == "general":
            nums = list(range(i+1, DIM))
            X[i][i] = 1
        elif datagen == "good":
            nums = set(range(i+1, DIM))
            #nums = {random.randint(i+1, DIM-1)}
            #nums = {DIM -1}
            X[i][i] = random.random()+1
        else:
            raise NotImplementedError

        while len(nums)>1:

            if datagen == "general":
                n = random.choice(nums)
                nums.remove(n)
            elif datagen == "good":
                n = nums.pop()
            else:
                raise NotImplementedError
            
            c_sum = torch.dot(X[i], w)
            
            
            errors = torch.Tensor([error-c_sum, -c_sum-error])/w[n]
            e_sum += errors[0]-errors[1]
            zero = -c_sum/w[n]      
            
            if abs(errors[1])>3:
                errors[1] = 3*(errors[1]/abs(errors[1]))
            if abs(errors[0])>3:
                errors[0] = 3*(errors[0]/abs(errors[0]))
            
            X[i][n] = random.random()*(errors[0]-errors[1])+errors[1]

        if datagen == "general":
            nums = [nums.pop()]+list(set(range(DIM)))  
            while True:  
                n = nums.pop()
                last = -(torch.dot(X[i], w)-X[i][n]*w[n])/w[n]
                if abs(last)<5: 
                    X[i][n] = last
                    break
        elif datagen == "good":
            n = nums.pop()
            X[i][n] = -torch.dot(X[i], w)/w[n]
        else:
            raise NotImplementedError

    if datagen == "general":
        O = get_matrix(DSIZE)
        for i in range(DSIZE):
            O[:, i] = (O[:, i]/torch.linalg.norm(O[:, i]))*((DIM-i)**0.3)
        X = O@X
    elif datagen == "good":
        while True:
            try:
                X = get_matrix(DSIZE)@X
                break
            except: 
                fgdh = 0
    else:
        raise NotImplementedError

    features.append(X)
    print(ind)
#print(features[0])
def ortho(W_star, ws):
    for w in orth(ws.T).T:
        w = torch.Tensor(w)
        W_star = W_star - torch.dot(W_star, w)/(torch.dot(w,w))*w
    return W_star

while True:
    W_star = ortho(torch.rand(DIM)*2-1, span_ws)
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
                torch.nn.init.constant_(m.weight, 0.001)
        
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
                        print("x")
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
        #print(self.coef_)
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
        
# If using "good" data generation, recompute W_star.
# if datagen == "good":
#     a = LRDataset(torch.cat(features, dim  = 0), torch.cat(labels, dim = 0))
#     a = DataLoader(a, shuffle = True, batch_size= 50)
#     model_star = Regression()
#     model_star.fit(a)
#     W_star = model_star.coef_.to(device).flatten()


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
results = torch.zeros(samps if samps!=0 else factorial(NUM_TASKS), 4)
w_distances = torch.zeros(samps if samps!=0 else factorial(NUM_TASKS), NUM_TASKS)
W_star = W_star.to(device)

losses = torch.zeros(samps if samps!=0 else factorial(NUM_TASKS), NUM_TASKS+1, NUM_TASKS) 
for s, ind in enumerate(torch.randint(0, factorial(NUM_TASKS), (samps,)) if samps != 0 else range(factorial(NUM_TASKS))):
    print(s, ind)
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
    print(s)
    
torch.save(results, f"../lgrgresults/res{SEED}30.pt")
torch.save(coefs, f"../lgrgresults/coefs{SEED}30.pt")
torch.save(w_distances, f"../lgrgresults/w_distances{SEED}30.pt")
torch.save(losses, f"../lgrgresults/losses{SEED}30.pt")
torch.save(span_ws, f"../lgrgresults/span_ws{SEED}30.pt")
