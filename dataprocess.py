import torch
import matplotlib.pyplot as plt
naive_data = torch.load("naive_simple_cifar10.pt")
print(naive_data)
num_tasks = 5
def calculate_forgetting(data):
    avg_forgettings = torch.zeros(len(data))
    for ind, perm_data in enumerate(data):
        peak_sum = 0        
        for i, task_num in enumerate(perm_data[1][-1]):
            
            peak_sum+=perm_data[1][i][int(task_num.item())]
        avg_forgettings[ind] = torch.sum(perm_data[1][-2]) - peak_sum
    return avg_forgettings
def calculate_FWT(data):
    fwd_transfers = torch.zeros(len(data))
    for ind, perm_data in enumerate(data):
        R_matrix = perm_data[0]
        forward_sum = 0
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i<R_matrix[-1][j]: forward_sum +=R_matrix[i][int(R_matrix[-1][j])]
        fwd_transfers[ind] = forward_sum/(num_tasks*(num_tasks-1)/2)
    return fwd_transfers
def calculate_BWT(data):
    bwts = torch.zeros(len(data))
    for ind, perm_data in enumerate(data):
        for i in range(1, num_tasks):
            for j in range(num_tasks):
                if int(perm_data[0][-1][j])>i-1: continue
                bwts[ind] += perm_data[0][i][int(perm_data[0][-1][j])] - perm_data[0][int(perm_data[0][-1][j])][int(perm_data[0][-1][j])]
    return torch.Tensor([i/(num_tasks*(num_tasks-1)/2) for i in bwts])
def calculate_accuracy(data):
    avgs = torch.zeros(len(data))
    for i, perm_data in enumerate(data):
        accuracies = perm_data[0][-2]
        avgs[i] = torch.sum(perm_data[0][-2]).item()/num_tasks
    return avgs


bwt = calculate_BWT(naive_data)
fwt = calculate_FWT(naive_data)
frgtng = calculate_forgetting(naive_data)
acc = calculate_accuracy(naive_data)


fig, axs = plt.subplots(1, 2, tight_layout=True)
print(fwt.shape, bwt.shape)
# We can set the number of bins with the *bins* keyword argument.
#axs[0].hist([int(i*100) for i in list(fwt)], bins=[i*100 for i in [0.4, 0.45, 0.5,0.55,  0.6, 0.65]])
#axs[1].hist([int(i*100) for i in list(bwt], bins=[i*100 for i in [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]])
axs[0].hist(fwt.numpy(), bins = 30)
axs[1].hist(bwt.numpy(), bins = 30)
plt.show()







        





        
