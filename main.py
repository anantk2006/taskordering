from avalanche.benchmarks import SplitMNIST, PermutedMNIST, SplitCIFAR10
import itertools
import naive
import icarl
import replay
import sys, os
import torch
import argparse
import agem
import ewc
from math import factorial

num_tasks = 5
parser = argparse.ArgumentParser(description = "Run task ordering simulations")
parser.add_argument("--strat", dest = "model", type = str)
parser.add_argument("--num-perms", dest = "num_perms", default = factorial(num_tasks))
args = parser.parse_args()


data = SplitCIFAR10(n_experiences=num_tasks)


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
    if args.model:
        if args.model == "naive":
            results = naive.run(train_set, test_set, data, 10)
        if args.model == "agem":
            results = agem.run(train_set, test_set, data, 10)
        if args.model == "icarl":
            results = icarl.run(train_set, test_set, data, 10)
        if args.model == "replay":
            results = replay.run(train_set, test_set, data, 10)
        if args.model == "ewc":
            results = ewc.run(train_set, test_set, data, 10)
    else: results = naive.run(train_set, test_set, data, 10)

    for IND, res in enumerate(results):
        #print(f"AFTER EXPERIENCE {IND}")
        for key, val in res.items():
            if key[:-3].endswith("Task") and "eval_phase" in key and "Acc" in key:
                data_tensor[ind][0][IND][int(key[-1])] = val                
            if key[:-3].endswith("Task") and "eval_phase" in key and "Loss" in key:
                data_tensor[ind][1][IND][int(key[-1])] = val
        for i in range(num_tasks): data_tensor[ind][0][-1][i] = data_tensor[ind][1][-1][i] = perm[i]      
            

torch.save(data_tensor, f"{model}.pt")

      
    
    
