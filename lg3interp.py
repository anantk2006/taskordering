import torch
import matplotlib.pyplot as plt
import itertools
data = torch.load("lgrgresults/lgres21.pt")
losses = torch.load("lgrgresults/lglosses21.pt")


data = torch.stack(data, dim = 0)
losses = torch.stack(losses, dim = 0)
span_ws = torch.stack(torch.load("lgrgresults/lgws21"))
print(data.shape, losses.shape, span_ws.shape)
ang_bet  = lambda a, b: torch.dot(a,b)/torch.linalg.norm(a)/torch.linalg.norm(b)
distances = []
displacements = []
ending_simils = []
loss_graph = []

for sample in range(8):
    for perm in range(250):
        permutation = losses[sample][perm][-1]
        order = [span_ws[sample][int(i)] for i in permutation]
        distances.append(sum(torch.abs(torch.acos(ang_bet(order[i], order[i+1]))) for i in range(7)))
        print([torch.abs(torch.acos(ang_bet(order[i], order[i+1]))) for i in range(7)])
        displacements.append(data[sample][perm][0])
        ending_simils.append(sum(torch.abs(torch.acos(ang_bet(order[i], order[-1]))) for i in range(7)))
        loss_graph.append(losses[sample][perm][-2].sum())
fig, axs = plt.subplots(1, 2, tight_layout = True)

axs[0].scatter(distances, displacements, s = 0.1)
axs[1].scatter(ending_simils, displacements, s = 0.1)
plt.savefig("pn.png")




