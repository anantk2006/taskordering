from avalanche.benchmarks import SplitMNIST, PermutedMNIST, SplitCIFAR10
import itertools
import naive
import sys, os
import torch

num_tasks = 5

split_cifar = SplitCIFAR10(n_experiences=num_tasks, return_task_id=True)


permutations = itertools.permutations(range(num_tasks), num_tasks)
train_cifar = split_cifar.train_stream
test_cifar = split_cifar.test_stream
train_cifar = list(train_cifar)
data_tensor = torch.zeros(size = (len(permutations:=list(permutations)), 2, num_tasks+1, num_tasks))
#sys.stdout = open(os.devnull, "w")
for ind, perm in enumerate(permutations[:3]):
    
#split_MNIST = SplitMNIST(n_experiences=num_tasks, return_task_id=True)
#permuted_MNIST = PermutedMNIST(n_experiences=num_tasks)

    
    train_cifar = [i[1] for i in sorted(list(zip(perm, train_cifar)))]
    results = naive.run(train_cifar, test_cifar, split_cifar, 10)
    for IND, res in enumerate(results):
        #print(f"AFTER EXPERIENCE {IND}")
        for key, val in res.items():
            if key[:-3].endswith("Task") and "eval_phase" in key and "Acc" in key:
                data_tensor[ind][0][IND][int(key[-1])] = val                
            if key[:-3].endswith("Task") and "eval_phase" in key and "Loss" in key:
                data_tensor[ind][1][IND][int(key[-1])] = val
        for i in range(num_tasks): data_tensor[ind][0][-1][i] = data_tensor[ind][1][-1][i] = perm[i]      
            

torch.save(data_tensor, "naive.pt")

      
    
    
