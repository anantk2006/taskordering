import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from itertools import permutations as permute
from time import time
from math import sin, cos, pi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-points", dest = "dsize", type = int, default = 100, 
                    help = "number of points to be classified per task")
args = parser.parse_args()

def generate_matrices(theta, phi, psi):
    '''
    Generates basis vectors in rotation matrix defined by 3 Eulers angles
    Theta, phi, psi define yaw, pitch, and roll respectively
    '''
    sindeg = lambda angle: sin(angle*pi/180)
    neg_sin = lambda angle: -sindeg(angle)
    cosdeg = lambda angle: cos(angle*pi/180)

    rotation_list = [cosdeg, neg_sin, sindeg, cosdeg] #declaring functions to be used in rotation matrix
    R1, R2, R3 = [torch.zeros(size = (3,3)), torch.zeros(size = (3,3)), torch.zeros(size = (3,3))]
    for i in range(2):
        for j in range(2):

            R1[i][j] = rotation_list[i*2+j](theta)
            R2[i*2][j*2] = rotation_list[i*2+j](phi) #assign values to each locations
            R3[i+1][j+1] = rotation_list[i*2+j](psi)
    R1[2][2] = R2[1][1] = R3[0][0] = 1 #Assign ones in random paces; similar to homogenous coordinates
    
    R = R3 @ R2 @ R1 
    
    return R #multiply matrices together and return the basis vector composition

class LRRotationDataset(Dataset):
    def __init__(self, X, Y, transform_matrix):
        '''
        Initializing Dataset to be placed inside DataLoader.         
        transform_matrix is matrix to be used to rotate all points created.
        X and Y and features and labels respectively
        '''
        self.X = torch.bmm(transform_matrix.repeat(args.dsize, 1, 1), X.unsqueeze(2)).squeeze(-1)
        self.Y = Y
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    def __len__(self):
        return len(self.X)


class LogisticRegression(nn.Module):
    def __init__(self):
        #declare model, loss function, optimizer, initialize weights
        super().__init__()
        self.linear = nn.Linear(3, 1, bias = True)
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



def get_accuracies(model, datasets):
    ret = []
    for dataset in datasets:
        ret.append(model.test(dataset))
    return [round(float(r),3) for r in ret]

def calculate_BWT(data):
    bwt = 0
    for i in range(1, 6):
        for j in range(0, i):                
            bwt += data[j][j] - data[i][j]
    return bwt/15
def ang_dist(a1, a2):
    s = 0
    for i in range(3):
        _1, _2 = a1[i], a2[i]
        if _1>180: _1 -= 360 
        if _2>180: _2 -= 360 
        s+= abs(_1-_2)
    return s

def train_CL(permuted_dataloaders, permuted_matrices, angles):
    results = torch.zeros(size = (720, 3))
    for i, permutation in enumerate(permuted_dataloaders):
        model = LogisticRegression()
        init_time = time()
        distance = 0
        curr_acc = torch.zeros(6, 6)
        for j, dataloader in enumerate(permutation):
            if j>0: distance += ang_dist(angles[i][j], angles[i][j-1])
            model.fit(dataloader)    
            curr_acc[j] = torch.Tensor(get_accuracies(model, permutation) ) 
        results[i][0] = distance
        results[i][2] = calculate_BWT(curr_acc)
        print(f"Distance traveled:\t {distance}")
        print(f"Accuracies:\t\t {(acc:=get_accuracies(model, permutation))}")
        results[i][1] = ((round(float(sum(acc)/6), 3)))
        print(f"Avg. accuracy:\t\t {results[i][1]}")
        print(f"Time taken:\t\t {time()-init_time}\n\n")
    return results



if __name__ == "__main__":
    angles = torch.rand(6, 3)*120
    matrices = [generate_matrices(*set_ang) for set_ang in angles] #get basis vectors
    
    X = torch.normal(torch.zeros(args.dsize, 3), torch.ones(args.dsize, 3)*3) 
    res = 1/(1+torch.exp(torch.bmm(torch.rand(3).unsqueeze(0).repeat(args.dsize, 1).unsqueeze(1),X.unsqueeze(2))))
    Y = torch.where(res>0.5, 1, 0).squeeze(1).squeeze(1)
    
    
    
    
    
    dataloaders = [DataLoader(LRRotationDataset(X, Y, M), shuffle = True, batch_size = 50) for M in matrices] #reassemble matrices
    permuted_dataloaders = list(permute(dataloaders)) #generate various task orderings

    #Generating X values using 0,0 and X, Y plane as dividing barrier to make labels
    point_put = torch.zeros(6, 3)
    og_point = torch.ones(size = (3,))
    for i, matrix in enumerate(matrices):
        point_put[i] = matrix @ og_point
    
    
    results = train_CL(permuted_dataloaders, list(permute(matrices)), a:=list(permute(list(angles))))
    torch.save(results, "results.pt")
    torch.save(angles, "angles.pt")


    
    
    





    
    
    

    

