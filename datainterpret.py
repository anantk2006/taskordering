import torch
import matplotlib.pyplot as plt
from math import factorial

sample_data = torch.load("cifar_results/cifar_rot_5_30_all_5.pt")
num_tasks = 5
num_perms = factorial(num_tasks)
#sample_data = sample_data.reshape(5, num_perms, 2, num_tasks+1, num_tasks)
print(sample_data.shape)
sample_data = sample_data.permute(1, 0, 2, 3 ,4)



#print(sample_data)

# s = torch.zeros(size = (1,num_perms,2,6,num_tasks))
# s[0] = sample_data
# sample_data = s
# print(sample_data.shape)

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
        ret[ind] = perm[-1]
        
    
    return ret
bwt, fwt, frgtng, acc = torch.zeros(size = (num_perms,)),torch.zeros(size = (num_perms,)),torch.zeros(size = (num_perms,)),torch.zeros(size = (num_perms,))
div_nums = torch.zeros(size = (num_perms,))
for ind, sample in enumerate(sample_data):
    acc_add = calculate_accuracy(sample)
    
    bwt_add = calculate_BWT(sample)
    fwt_add = calculate_FWT(sample)
    frg_add = calculate_forgetting(sample)
    div_nums += torch.where(acc_add>0.25, torch.ones(num_perms), torch.zeros(num_perms))
    #div_nums += torch.ones(num_perms)
    bwt += bwt_add
    fwt += fwt_add
    frgtng += frg_add
    acc += acc_add


bwt, fwt, frgtng, acc = bwt/div_nums, fwt/div_nums, frgtng/div_nums, acc/div_nums
print(acc)
#acc = torch.where(acc>0.15, acc, torch.ones_like(acc)*0.5)
# for i in range(num_perms):
#     if acc[i]>0.42:
#         print(sample_data[0][i][0][-1])
# for i in range(100):
#     print()
# for i in range(num_perms):
#     if acc[i]<0.42:
#         print(sample_data[0][i][0][-1])
    
acc_list = list(acc)
l = zip(acc_list, list(torch.arange(num_perms)))
l = sorted(l)

for i in range(len(l)):    
    ii = l[i][1]
    print(f"Permutation: {sample_data[0][ii][0][-1]}, Accuracy: {acc[ii]}, Forgetting: {frgtng[ii]}, FWT: {fwt[ii]}, BWT: {bwt[ii]}")

distances = get_distances(sample_data[0])
disps = get_displacements(sample_data[0])
fig, axs = plt.subplots(1,2, tight_layout=True)
print(fwt.shape, bwt.shape)
dic = {}
for ind, val in enumerate(distances):
    print(ind, val)
    v = round(float(val))
    if v in dic: dic[v].append(float(acc[ind]))
    else: dic[v] = [float(acc[ind])]

axs[0].scatter(distances.numpy(), acc.numpy(), s = 1)
#axs[0].set_xticklabels(dic.keys())
# #axs[1].hist(frgtng.numpy(), bins = 30)

dic = {}
for ind, val in enumerate(disps):
    print(ind, val)
    v = round(float(val), 2)
    if v in dic: dic[v].append(float(acc[ind]))
    else: dic[v] = [float(acc[ind])]

axs[1].boxplot(dic.values(), showfliers=False)

# axs.boxplot(dic.values())



axs[1].set_xticklabels(dic.keys())
axs[0].set_title("Distance vs accuracy")
axs[0].set_xlabel("Path distance")
axs[0].set_ylabel("accuracy")
axs[1].set_title("Ending angle vs Accuracy")
axs[1].set_xlabel("Ending angle")
axs[1].set_ylabel("accuracy %")
# axs.set_title("Distance vs accuracy")
# axs.set_xlabel("Path distance")
# axs.set_ylabel("accuracy %")

plt.savefig("graphs/res.png")







        





        
