import torch
import matplotlib.pyplot as plt
from itertools import permutations as permute

t = torch.load("results.pt")
a = torch.load("angles.pt")

permuted_a = list(permute(list(a)))

for i in range(720): 
    if t[i][1] > 0.67: print(permuted_a[i][-1])
print("\n\n\n\n\n\n\n\n\n")
for i in range(720): 
    if t[i][1] < 0.67: print(permuted_a[i][-1])

def ang_dist(a1, a2):
    s = 0
    for i in range(3):
        _1, _2 = a1[i], a2[i]
        if _1>180: _1 -= 360 
        if _2>180: _2 -= 360 
        s+= abs(_1-_2)
    return s


acc_list = list(t.mT[1])
l = zip(acc_list, list(torch.arange(720)))
l = sorted(l)
a1, a2, a3, a4 = "[", "]", "{", "}"
for i in range(len(l)):    
    ii = l[i][1]
    print("\n\n")
    print(f"Permutation: {str(torch.cat([b.unsqueeze(0) for b in permuted_a[ii]], dim = 0).tolist()).replace(a1, a3).replace(a2, a4)}, Accuracy: {t[ii][1]}")
# sim = torch.zeros(size = (6,6))
# for i in range(6):
#     for j in range(6):
#         sim[i][j] = torch.abs(a[i] - a[j])
# print(sim)
# print(sim.sum(dim = 1))
for an in a:
    for i in range(3):
        _1 = an[i]
        if _1>180: _1 -= 360 
        an[i] = _1
print(str(a).replace("[", "{").replace("]", "}"))

dic = {}

for ind, p in enumerate(permuted_a):
    v = tuple(p[-1].tolist())
    if v in dic: dic[v].append((t[ind][1]).item())
    else: dic[v] = [(t[ind][1]).item()]


fig, axs = plt.subplots(1,2, tight_layout = True)


axs[0].boxplot(dic.values(), showfliers = False)
axs[1].scatter(t.mT[0].numpy(), t.mT[1].numpy(), s = 1)
axs[0].set_xticklabels([*range(1, 7)])
axs[0].set_title("Accuracies for each ending angle")
axs[0].set_xlabel("Which tuple of angles")
axs[0].set_ylabel("Accuracy")
axs[1].set_title("Path distance versus accuracy")
axs[1].set_xlabel("Path distance")
axs[1].set_ylabel("Accuracy")

plt.savefig("c.png")