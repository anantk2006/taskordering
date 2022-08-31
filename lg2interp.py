import torch
import matplotlib.pyplot as plt
import itertools
data = torch.load("lgrgresults/lg_8_res.pt")
losses = torch.load("lgrgresults/lg_8_losses.pt")

data = torch.stack(data, dim = 0)
losses = torch.stack(losses, dim = 0)


data = data.sum(0)/40
losses = losses.sum(0)/40

def calculate_forgetting(data):
    avg_forgettings = torch.zeros(len(data))
    for ind, perm_data in enumerate(data):
        peak_sum = 0        
        for i, task_num in enumerate(perm_data[-1]):
            
            peak_sum+=perm_data[i][int(task_num.item())]
        avg_forgettings[ind] = torch.sum(perm_data[-2]) - peak_sum
        print(perm_data)
    return avg_forgettings
frg = calculate_forgetting(losses)
perms = torch.Tensor(list(itertools.permutations(range(5))))
c = 0
for ind in range(120):
    if data[ind][2]>0.97: print(perms[ind]); c+=1
print(c)

fig, axs = plt.subplots(2, 2, tight_layout = True)
col = []
for i in perms:
    col += ["r"] if i[-2] == 2 else ["b"]
    

axs[0][0].scatter(data.mT[0].numpy(), data.mT[1].numpy(), s = 0.1, c = col)
axs[1][0].scatter(data.mT[1].numpy(), frg, s = 0.1, c = col)
axs[0][1].scatter(frg, data.mT[0].numpy(), s = 0.1, c = col)
axs[1][1].scatter(data.mT[3], data.mT[0], s = 0.1, c = col )

plt.savefig("pn.png")


print(list(data.mT[0]))

z = sorted(zip([i.item() for i in frg], [[i.item() for i in perm] for perm in perms]), reverse= True)
for s in z:
    print(s)

# z = sorted(zip([i.item() for i in (ws.mT[-1]-ws.mT[0])], [perm[0]-perm[-1] for perm in perms], [perm for perm in perms]), reverse= False)
# for s in z:
#     print(s)