from avalanche.benchmarks import SplitMNIST, PermutedMNIST, SplitCIFAR10
import itertools

import torch
import argparse

from math import factorial
import trainer

num_tasks = 5
parser = argparse.ArgumentParser(description = "Run task ordering simulations")
parser.add_argument("--strat", dest = "strat", type = str)
parser.add_argument("--num-perms", dest = "num_perms", default = factorial(num_tasks))
parser.add_argument("--data", dest = "dataset", default = "cifar10")
parser.add_argument("--model", dest = "model", default = "simple")
args = parser.parse_args()

if args.dataset == "cifar10":
    data = SplitCIFAR10(n_experiences=num_tasks)
    num_channels = 3
elif args.dataset == "smnist":
    data = SplitMNIST(n_experiences=num_tasks)
    num_channels = 1
else: 
    data = PermutedMNIST(n_experiences=num_tasks)
    num_channels = 1


permutations = itertools.permutations(range(num_tasks), num_tasks)
train_set = data.train_stream
test_set = data.test_stream
train_set = list(train_set)
data_tensor = torch.zeros(size = (len(permutations:=list(permutations)), 2, num_tasks+1, num_tasks))
#sys.stdout = open(os.devnull, "w")
iter_delete = factorial(num_tasks)//(args.num_perms)
num_deleted = 0
i = 0
while i<factorial(num_tasks) and num_deleted<factorial(num_tasks)-args.num_perms:
    permutations.remove(i)
    i+=iter_delete; num_deleted+=1
for ind, perm in enumerate(permutations):
    
#split_MNIST = SplitMNIST(n_experiences=num_tasks, return_task_id=True)
#permuted_MNIST = PermutedMNIST(n_experiences=num_tasks)

    
    train_set = [i[1] for i in sorted(list(zip(perm, train_set)))]
    results = trainer.run(args.strat, train_set, test_set, data, 10, 
            num_channels=num_channels, device = "cuda" if torch.cuda.is_available() else "cpu", model = args.model)
   

    for IND, res in enumerate(results):
        
        for key, val in res.items():
            if key[:-3].endswith("Task") and "eval_phase" in key and "Acc" in key:
                data_tensor[ind][0][IND][int(key[-1])] = val   
                print(val)            
            if key[:-3].endswith("Task") and "eval_phase" in key and "Loss" in key:
                data_tensor[ind][1][IND][int(key[-1])] = val
        for i in range(num_tasks): data_tensor[ind][0][-1][i] = data_tensor[ind][1][-1][i] = perm[i]      
            

torch.save(data_tensor, f"{args.strat}.pt")

      
    
    
