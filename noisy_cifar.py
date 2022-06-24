import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from itertools import permutations as permute
from models import MLP, SimpleCNN
from torch.utils.data import DataLoader
from math import factorial
from time import time as process_time

torch.manual_seed(0)
#torch.use_deterministic_algorithms(True)

stdevs = [0, 0.1, 0.2, 0.3]
NUM_TASKS = 4

class Noise:
    def __init__(self, noise):
        self.noise = noise
        
    def __call__(self, X):
        
        noisy_X = torch.normal(X, noise)
        #if torch.sum(noisy_X - X)>0.1: print("x")
        return noisy_X


train_data = [((stdevs[i]), DataLoader(CIFAR10(root = "data", train = True, download = True, 
        transform = transforms.Compose([transforms.ToTensor(), Noise(stdevs[i])])), batch_size = 100)) 
 for i in range(NUM_TASKS)]
test_data = [((stdevs[i]), DataLoader(CIFAR10(root = "data", train = False, download = True, 
        transform = transforms.Compose([transforms.ToTensor(), Noise(stdevs[i])])), batch_size = 100)) 
 for i in range(NUM_TASKS)]

train_data_permutes = list(permute(train_data)) 

data_tensor = torch.zeros(factorial(NUM_TASKS), 2, NUM_TASKS+1, NUM_TASKS)

device = torch.device("cuda:0")
# for data in train_data_permutes[0]:
#     for X, Y in data[1]:
        
#         X = X.to(device); Y.to(device)
    
# for data in test_data:
#     for X, Y in data[1]:
        
        
#cpu = torch.device("cpu")

def train(model, data, loss_fn, optim):
    
    model.train()
    for _ in range(50):
        num_corr = 0
        for ind, (X, Y) in enumerate(data):
            X = X.to(device); Y = Y.to(device)
            Y_hat = model(X)
            loss = loss_fn(Y_hat, Y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            num_corr += torch.sum(torch.argmax(Y_hat, dim = 1)==Y)
    print("train accuracy")
    print(num_corr/((ind+1)*100))            

cpu = torch.device("cpu")
def test_results(model, test_data, loss_fn):
    model.eval()
    tim = process_time()
    acc = torch.zeros(size  = (NUM_TASKS,), device = device)
    loss = torch.zeros(size = (NUM_TASKS,), device = device)
    for ind, (_, te_data) in enumerate(test_data):
        loss_agg = 0
        acc_agg = 0
        num_done = 0
        for num_batches, (X, Y) in enumerate(te_data):
            if num_done>4: break
            num_done += 1
            X = X.to(device); Y = Y.to(device)
            Y_hat = model(X)
            loss_curr = loss_fn(Y_hat, Y)
            loss_agg += float(loss_curr)
            # print(Y_hat)
            # print(torch.argmax(Y_hat, dim = 1))
            # print(Y)
            num_correct = torch.sum(torch.argmax(Y_hat, dim = 1)==Y)
            acc_agg += num_correct
            
        acc[ind] = acc_agg/(500)
        loss[ind] = loss_agg/(500)
    print(process_time()-tim)
    return acc, loss

        



loss_fn = torch.nn.CrossEntropyLoss()
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

for ind_perm, tr_data in enumerate(train_data_permutes):
    model = SimpleCNN(num_channels = 3, num_classes = 10).to(device)


    
    model.apply(init_weights)
    optim = torch.optim.SGD(model.parameters(), lr = 0.25)
    for ind_task, (noise, data) in enumerate(tr_data):
        init_time = process_time()
        train(model, data, loss_fn, optim)
        acc, loss = test_results(model, test_data, loss_fn)
        print(f"Task number: {ind_task}      Task noise: {noise}      What permutation: {ind_perm}" )
        print(f"Accuracies: {acc}      Losses: {loss}")
        print(f"Time taken: {round(process_time()-init_time, 2)}")
        data_tensor[ind_perm][0][ind_task][:] = acc
        data_tensor[ind_perm][1][ind_task][:] = loss
        data_tensor[ind_perm][0][-1][ind_task] = noise
torch.save(data_tensor, "rotated_cifar10_data.pt")


    


    







