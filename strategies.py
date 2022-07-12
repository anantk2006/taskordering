from numpy import float16
import torch
import torch.nn as nn
from time import time as process_time
from torch.utils.data import DataLoader
from PIL import Image
import numpy


class Naive:
    def train(self, model, data, loss_fn, optim, args, device):
        ''' model : model to be trained
            data: training dataset
            loss_fn: Cross-Entropy most likely
            optim: optimizer
            args: cmd line input used to determine certain specs
            device: cuda to train on'''
        model.train()
        loss_agg = 2.5 #numbers to use to track loss and terminate on certain loss threshold if need be
        loss_num = 1
        ep = 0
        while (loss_agg/loss_num>args.dloss if args.loop == "loss" else ep<args.epochs):
            #either end on epoch or end after certain loss threshold
            for ind, (X, Y) in enumerate(data): 
                # if ind == 0:
                    
                #     print(X[0].permute(1, 2, 0).cpu().numpy().shape)
                #     da = Image.fromarray((X[0][0]*256).permute(1, 2, 0).cpu().numpy().astype(numpy.uint8))
                #     da.save("graphs/im.png")            
                X = X.to(device); Y = Y.to(device)
                Y_hat = model(X)
                loss = loss_fn(Y_hat, Y)
                optim.zero_grad()
                loss.backward()
                l = float(loss)
                loss_agg+=l
                loss_num+=1
                if ind>100 and loss_agg/loss_num<args.dloss and args.loop == "loss": break
                optim.step()
            loss_agg, loss_num = loss_agg/loss_num, 1 
            ep+=1
    

    

class Replay:
    def __init__(self, num_examples, args):
        '''
        num_examples: number of images allowed to be stored
        '''
        self.num_examples = num_examples
        self.examples = []
        self.task_ind = 0
        self.dataloaders = [None]*args.num_tasks
        
        
    def reconstruct_examples(self):
        '''
        After training, one must reduce the number of examples for all tasks as 
        one new task must be included for.
        '''
        for ind, ex in enumerate(self.examples):
            if len(ex)>(self.num_examples//(self.task_ind+1)): self.examples[ind] = ex[:self.num_examples//(self.task_ind+1)]
                        
            self.dataloaders[ind] = DataLoader(self.examples[ind], batch_size = 100, shuffle = True, num_workers=9)
        self.task_ind+=1

    def train(self, model, data, loss_fn, optim, args, device):
        model.train()
        loss_agg = 2.5
        loss_num = 1
        ep = 0
        curr_ex = []
        needed = self.num_examples//(self.task_ind+1)
        samples = [int(i) for i in list(torch.randint(0, 600, (needed,)))]
        samples = {i:samples.count(i) for i in samples} 
        #randomly sample some n number of numbers, and sample examples from those minibatches
        
        
        while (loss_agg/loss_num>args.dloss if args.loop == "loss" else ep<args.epochs):  
            temp = [] 
            for i in self.dataloaders:
                if i: temp.append(iter(i))
                else: temp.append(None)

            for ind, (X, Y) in enumerate(data):     
                
                X = X.to(device); Y = Y.to(device)
                optim.zero_grad()
                
                Y_hat = model(X)
                loss = loss_fn(Y_hat, Y) # train with similar procedures to above Naive training
                
                loss.backward()
                optim.step()
                l = float(loss)
                
                loss_agg+=l
                loss_num+=1
                '''Every n minibatches, the code goes through 1 minibatch of each past task and backpropagates on that task
                    n = num_of_minibatches/((num_of_examples/batch_size)/num_of_previous_task)'''
                if self.task_ind>0 and not ind%(600//((self.num_examples//100)//(self.task_ind))):
                    for dl in temp:
                        if not dl: break #if the time has come to replay, then get the past examples
                        
                        self.train_replay(model, next(dl), loss_fn, optim, device)
                        
                        
                if ind>100 and loss_agg/loss_num<args.dloss and args.loop == "loss": break
                if ep == 0 and ind in samples: 
                    batch_ind = torch.randint(0, 100, (samples[ind],)) #randomly sample some indices to get exs from
                    for i in batch_ind:
                        curr_ex.append((X[i].cpu(), Y[i].cpu())) #need to move them to CPU, or too much memory is used
                    
                
            loss_agg, loss_num = loss_agg/loss_num, 1 
            ep+=1
        
        self.examples.append(curr_ex)
        self.reconstruct_examples()
    
    def train_replay(self, model, ex, loss_fn, optim,device):
       
        X, Y = ex
              #train on a single example
        X = X.to(device); Y = Y.to(device)
        Y_hat = model(X)
        optim.zero_grad()
        loss = loss_fn(Y_hat, Y)
        
        loss.backward()               
        optim.step()
        
    
   

# class AGEM:
#     def __init__(self):
    


