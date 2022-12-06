import torch
from torch import nn
from math import sqrt
import time
class Regression(nn.Module):
    def __init__(self, DIM, FUNC):
        #declare model, loss function, optimizer, initialize weights
        super().__init__()
        self.linear = nn.Linear(DIM, 1, bias = False)
        self.net = nn.Sequential(self.linear, nn.Sigmoid()) if FUNC == "log" else nn.Sequential(self.linear)
        self.loss_fn = nn.BCELoss() if FUNC == "log" else nn.MSELoss()
        self.function = FUNC
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.constant_(m.weight, 1e-6) # not zero because that leads to weird convergence issues
        
        self.net.apply(init_weights)

    def fit(self, dataset, lr): 
        '''
        Train the model on dataset
        '''
        
        self.optim = torch.optim.SGD(self.net.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lambda epoch: lr/sqrt(epoch+1))
        t = time.time()
      
        while (self.test(dataset)[1]>0.01) if self.function == "lin" else ((p:=self.test(dataset))[0]<1 or p[1]>0.002):
            
            if time.time()-t> 8: break
            for features, labels in dataset:   
                self.optim.zero_grad()
                preds = self(features).unsqueeze(-1) #predict using model
                loss = self.loss_fn(preds.flatten(), labels.flatten().to(torch.float32))                
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
          
            preds = self(feat).unsqueeze(-1) #find the accuracy on a certain give dataset
            pr = torch.where(preds>0.5, 1, 0)            
            if self.function == "lin": acc_agg += self.acc_calc(preds, lab)
            else: acc_agg += (pr.reshape(len(lab))==lab.squeeze(-1)).sum()
            
            loss += torch.linalg.norm(preds.reshape(len(lab))- lab.reshape(len(lab)))**2
            count+=len(feat)
            
        if self.function == "log": return acc_agg/count, loss/count   
        else: return acc_agg/count, loss
