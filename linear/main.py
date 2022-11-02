import sys; args = sys.argv[1:]

import time, random
from itertools import permutations as permute
from math import sin, cos, pi, factorial, sqrt

import torch
from torch.utils.data import DataLoader
from scipy import linalg
import numpy

from models import Regression
from dataset import LRDataset


NUM_TASKS = 5
INC = 1/18* pi
DSIZE = 500
DIM = 501
samps = 0

GPU = 3
SEED = int(args[0])
FUNC = "log"
datagen = "general" # Choices: ["general", "good"]

lr = 0.5 if FUNC == "lin" else 0.1 if datagen=="general" else 0.5
BATCH_SIZE = 50

ERROR = 0.05 #max ERROR that data row can stray from 0 to minimize last element
BALANCE = 0.3 # counter balance exponent to balance out X from numerical stray resulting from triangularity
DATA_ENTRY_MAX = 5

ang_bet  = lambda a, b: torch.dot(a,b)/(torch.linalg.norm(b)*torch.linalg.norm(a)) #angle similarity metrics 
proj = lambda a, b: (torch.dot(a, b)/torch.dot(b, b))*b

def generate_data():
    """ Generate training features. """
    def get_matrix(N):
        while True : #sometimes scipy bugs out, so we need a while loops
            Phi = numpy.random.randn(N, N).astype(numpy.float32)
            g = linalg.orth(Phi)
            if (g.shape[0], g.shape[1])==(N, N):
                return torch.from_numpy(g)
        #returns matrix to multiple with triangular X to balance it out    

    def rot(w):
        M = get_matrix(w.shape[0])
        return (M @ w) #rotates vector

    def get_span_ws():
        if datagen == "general":
            if samps == 0:
                span_ws = torch.Tensor([[0]*(DIM-2)+[sin((x)*INC), cos((x)*INC)] for x in range(NUM_TASKS)])
                span_ws = rot(span_ws.T).T #creates num_tasks span_ws and rotates them into DIM space
            else:
                span_ws = torch.zeros(NUM_TASKS, DIM)
                span_ws[0] = get_matrix(DIM)@torch.Tensor([1]+[0]*(DIM-1)) # creates num_tasks arbritrary vectors by
                #incrementally rotating them
                for i in range(NUM_TASKS-1):
                    span_ws[i+1] = get_matrix(DIM)@span_ws[i] 
        elif datagen == "good":
            span_ws = torch.Tensor([[0]*(DIM-2)+[sin((x)*INC), cos((x)*INC)] for x in range(NUM_TASKS)])
            span_ws = rot(span_ws.T).T
        else:
            raise NotImplementedError
        return span_ws

    span_ws = get_span_ws()

    features = []
    i = 0
    
    for ind, w in enumerate(span_ws):
        X = torch.zeros((DSIZE, DIM))
        e_sum = 0
        for i in range(DSIZE):
            
            if datagen == "general":
                nums = list(range(i+1, DIM))
                X[i][i] = 1
            elif datagen == "good":
                nums = set(range(i+1, DIM)) #which numbers to fill in
                X[i][i] = random.random()+1
            else:
                raise NotImplementedError

            while len(nums)>1:

                if datagen == "general":
                    n = random.choice(nums) # changes every time
                    nums.remove(n)
                elif datagen == "good":
                    n = nums.pop() #this is same every time because of hashing
                else:
                    raise NotImplementedError
               
                c_sum = torch.dot(X[i], w)
                
                
                
                errors = torch.Tensor([ERROR-c_sum, -c_sum-ERROR])/w[n]
                e_sum += errors[0]-errors[1]
                    
                
                if abs(errors[1])>DATA_ENTRY_MAX:
                    errors[1] = DATA_ENTRY_MAX*(errors[1]/abs(errors[1])) #stops data from getting too large
                if abs(errors[0])>DATA_ENTRY_MAX:
                    errors[0] = DATA_ENTRY_MAX*(errors[0]/abs(errors[0]))
                
                X[i][n] = random.random()*(errors[0]-errors[1])+errors[1]

            if datagen == "general":
                nums = [nums.pop()]+list(set(range(DIM)))  
                while True:  #finding optimal location to place last number to prevent overflow
                    n = nums.pop()
                    last = -(torch.dot(X[i], w)-X[i][n]*w[n])/w[n]
                    if abs(last)<DATA_ENTRY_MAX: 
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
                O[:, i] = (O[:, i]/torch.linalg.norm(O[:, i]))*((DIM-i)**BALANCE) # balance out columns of matrix
            X = O@X
        elif datagen == "good":
            X = get_matrix(DSIZE)@X                  
        else:
            raise NotImplementedError

        features.append(X)
        print(f"Task Data {ind} created")
    return features, span_ws


def generate_solution_labels(features, span_ws):
    """ Generate labels for given features based on ground-truth weights. """
    def ortho(W_star, ws):
        for w in linalg.orth(ws.T).T:
            w = torch.Tensor(w)
            W_star = W_star - torch.dot(W_star, w)/(torch.dot(w,w))*w
        return W_star

    while True:
        W_star = ortho(torch.rand(DIM)*2-1, span_ws) #orthogonalize W-star to remove biases
        if FUNC == "log": labels = [torch.where((X @ W_star.unsqueeze(-1))>0, 1, 0) for X in features]
        else: labels = [X @ W_star.unsqueeze(-1) for X in features]

        if FUNC == "lin": break

        for label in labels: #ensure that percentage of 1s is between 1/3 and 2/3 - balance in labels
            if label.sum()>DSIZE//3 or label.sum()<DSIZE/1.5: continue
        break
    return W_star, labels


def train(W_star, dataloaders, device):
    """ Run continual learning with a linear model. """  

    def get_accuracies(model, datasets):
        ret_a = []; ret_l = []
        for dataset in datasets:
            ret_a.append((t:=model.test(dataset))[0])
            ret_l.append(t[1])
        return [float(r) for r in ret_a], ret_l #return test accuracies and losses for the model

    def get_distances(w, w_s):
        if torch.dot(w, w_s)>0:
            return torch.linalg.norm(w - (torch.dot(w, w_s)/torch.dot(w_s, w_s))*w_s) #return residual from w to W_star
        else: return torch.linalg.norm(w)


    dataloaders = list(permute(dataloaders, NUM_TASKS))

    results = torch.zeros(samps if samps!=0 else factorial(NUM_TASKS), 4)
    w_distances = torch.zeros(samps if samps!=0 else factorial(NUM_TASKS), NUM_TASKS)
    coefs = torch.zeros(factorial(NUM_TASKS), DIM)
    losses = torch.zeros(samps if samps!=0 else factorial(NUM_TASKS), NUM_TASKS+1, NUM_TASKS) 

    W_star = W_star.to(device)
    
    for s, ind in enumerate(torch.randint(0, factorial(NUM_TASKS), (samps,)) if samps != 0 else range(factorial(NUM_TASKS))):
        dataset = dataloaders[ind]
        model = Regression(DIM, FUNC).to(device)
        permutation = list(permute(range(NUM_TASKS)))[ind]
        distance = 0
        init_time = time.process_time()        
        for task_ind in range(NUM_TASKS):
                           
            model.fit(dataset[task_ind], lr)            
            c = model.coef_.flatten() 
            losses[s][task_ind] = torch.Tensor((g:=get_accuracies(model, dataloaders[0]))[1])

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
    return results, losses, w_distances, coefs

def main():    
    """ Main function for linear continual learning experiments. """

    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    random.seed(SEED)
    numpy.random.seed(SEED)

    # Set up training data.
    features, span_ws = generate_data()
    W_star, labels = generate_solution_labels(features, span_ws)
    
    dataloaders = [
        DataLoader(
            LRDataset(feat.to(device), lab.to(device)), batch_size=BATCH_SIZE, shuffle=True
        )
        for feat, lab in zip(features, labels)
    ]

    # Perform training and save results.
    results = train(W_star, dataloaders, device)
    torch.save(results, f"../lgrgresults/results{SEED}.pt")
    

if __name__ == "__main__":
    main()
