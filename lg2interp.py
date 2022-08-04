import torch
import matplotlib.pyplot as plt
import itertools

data = torch.load("lgrgresults/res1.pt")
perms = torch.Tensor(list(itertools.permutations(range(5))))
c = 0
for ind in range(120):
    if data[ind][2]>0.97: print(perms[ind]); c+=1
print(c)

fig, axs = plt.subplots(2, 2, tight_layout = True)

axs[0][0].scatter(data.mT[0].numpy(), data.mT[1].numpy(), s = 0.1)
axs[1][0].scatter(data.mT[0].numpy(), data.mT[2].numpy(), s = 0.1)
axs[0][1].scatter(data.mT[1].numpy(), data.mT[2].numpy(), s = 0.1)
axs[1][1].scatter(data.mT[0].numpy(), perms.mT[-2], s = 0.1 )

plt.savefig("pn.png")