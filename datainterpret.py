import torch
import matplotlib.pyplot as plt
sample_data = torch.load("cifar_results/noise_5_003_all_5.pt")

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
                if i<R_matrix[-1][j]: forward_sum +=R_matrix[i][int(R_matrix[-1][j])//30]
        fwd_transfers[ind] = forward_sum/(num_tasks*(num_tasks-1)/2)
    return fwd_transfers
def calculate_BWT(data):
    bwts = torch.zeros(len(data))
    for ind, perm_data in enumerate(data):
        for i in range(1, num_tasks):
            for j in range(num_tasks):
                if int(perm_data[0][-1][j])//30>i-1: continue
                bwts[ind] += perm_data[0][i][int(perm_data[0][-1][j])//30] - perm_data[0][int(perm_data[0][-1][j])//30][int(perm_data[0][-1][j])//30]
    return torch.Tensor([i/(num_tasks*(num_tasks-1)/2) for i in bwts])
def calculate_accuracy(data):
    avgs = torch.zeros(len(data))
    for i, perm_data in enumerate(data):
        accuracies = perm_data[0][-2]
        avgs[i] = torch.sum(perm_data[0][-2]).item()/num_tasks
    return avgs

def get_distances(data):
    ret = torch.zeros(len(data))
    for ind, tensor in enumerate(data):
        perm = tensor[0][-1]

        for i in range(1, len(perm)):
            ret[ind] += abs(perm[i]-perm[i-1])
    return ret
def get_displacements(data):
    ret = torch.zeros(len(data))
    for ind, tensor in enumerate(data):
        perm = tensor[0][-1]
        ret[ind] = perm[-1]+perm[-2]-(perm[0]+perm[1])
    return ret
bwt, fwt, frgtng, acc = torch.zeros(size = (120,)),torch.zeros(size = (120,)),torch.zeros(size = (120,)),torch.zeros(size = (120,))
div_nums = torch.zeros(size = (120,))
for ind, sample in enumerate(sample_data):
    acc_add = calculate_accuracy(sample)
    bwt_add = calculate_BWT(sample)
    fwt_add = calculate_FWT(sample)
    frg_add = calculate_forgetting(sample)
    div_nums += torch.where(acc_add>0, torch.ones(120), torch.zeros(120))
    bwt += bwt_add
    fwt += fwt_add
    frgtng += frg_add
    acc += acc_add


bwt, fwt, frgtng, acc = bwt/div_nums, fwt/div_nums, frgtng/div_nums, acc/div_nums
for i in range(120):
    if acc[i]>0.61:
        print(sample_data[0][i][0][-1])
for i in range(100):
    print()
for i in range(120):
    if acc[i]<0.61:
        print(sample_data[0][i][0][-1])
    
# for i in range(len(acc)):    
#     print(f"Permutation: {sample_data[0][i][0][-1]}, Accuracy: {acc[i]}, Forgetting: {frgtng[i]}, FWT: {fwt[i]}, BWT: {bwt[i]}")

distances = get_distances(sample_data[0])
disps = get_displacements(sample_data[0])
fig, axs = plt.subplots(1,4, tight_layout=True)
print(fwt.shape, bwt.shape)
# We can set the number of bins with the *bins* keyword argument.
#axs[0].hist([int(i*100) for i in list(fwt)], bins=[i*100 for i in [0.4, 0.45, 0.5,0.55,  0.6, 0.65]])
#axs[1].hist([int(i*100) for i in list(bwt], bins=[i*100 for i in [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]])
axs[0].hist(acc.numpy(), bins = 30)
axs[1].hist(frgtng.numpy(), bins = 30)
axs[2].scatter(distances.numpy(), acc.numpy(), s = 1)
axs[3].scatter(disps.numpy(), acc.numpy(), s = 1)
plt.savefig("4_30.png")







        





        
