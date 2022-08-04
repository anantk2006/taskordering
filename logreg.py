from cProfile import label
import numpy as np
from math import exp, e
np.random.seed(0)
import torch
import torch.nn as nn

'''
Making synthetic data
First generate W* and then generate all the different clusters
'''
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7, 1, bias = False)
        self.net = nn.Sequential(self.linear, nn.Sigmoid())
        self.loss_fn = nn.BCELoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr = 1)
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, std=0.01)

        self.net.apply(init_weights)

   
    def fit(self, features, labels):  
        
        while (p:=self.test(features, labels))<0.999:
            
            self.optim.zero_grad()
            preds = self(features).unsqueeze(-1)
            loss = self.loss_fn(preds.reshape(len(labels)), labels)
            loss.backward()
            self.optim.step()   
            
        self.coef_ = self.linear.weight.data
    def forward(self, features):
        return self.net(features)
    def test(self, features, labels):
        preds = self(features).unsqueeze(-1)
        
        preds = torch.where(preds>0.5, 1, 0)
        #print(preds)
        return (preds.reshape(len(labels))==labels).sum()/len(labels)
    




features = []

while True:
    x_star = (torch.rand(7)-0.5)*6 #generate X_star such that the mean A doesn't have any absurdly large numbrs

    a_mean = (1/x_star)/(7/e)
    if all(a_mean<5):
        break



features = torch.zeros(5, 50, 7)
for noise in [i*4 for i in range(1, 6)]:
    features[int(noise/4)-1] = torch.from_numpy(np.random.normal(a_mean.numpy(), noise, size = (50, 7)))
    
    



model = LogisticRegression()

labels = torch.zeros((5, 50))

for i, task in enumerate(features):
    for j, sample in enumerate(task):
        
        P = 1/(1+exp(torch.dot(sample, x_star)))
        
        labels[i][j] = 1 if P>0.5 else 0

all_data  = features.reshape(250, 7)
all_labels = labels.reshape(250)
model_star = LogisticRegression()
model_star.fit(all_data, all_labels)
x_star = model_star.coef_


for ind, task_num in enumerate([0, 1, 2, 3, 4]):
    
    model.fit(features[task_num], labels[task_num])    
    print(torch.linalg.norm(model.coef_ - x_star))
    print(f"current coef: {model.coef_}")
    print(f"X_star: {x_star}")
    

    













        