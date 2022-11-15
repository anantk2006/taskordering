import torch
import matplotlib.pyplot as plt
import itertools
from math import factorial
res = torch.load("../lgrgresults/lgresranked.pt")


data = torch.stack([d[0] for d in res], dim = 0)
losses = torch.stack([d[1] for d in res], dim = 0)
print(data.shape, losses.shape)
if len(data[0])==factorial(len(losses[0][0][-1])):
#print(data.shape, losses.shape)


    data = data.sum(0)/(d:=len(data))
    losses = losses.sum(0)/d
    
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
    axs[1][1].scatter(data.mT[3], frg, s = 0.1, c = col )

    plt.savefig("pn.png")


    print(list(data.mT[0]))

    z = sorted(zip([i.item() for i in data.mT[0]], [[i.item() for i in perm] for perm in perms]), reverse= True)
    for s in z:
        print(s)
else:
    span_ws = [d[4] for d in res]
    

    ang_bet  = lambda a, b: torch.dot(a,b)/torch.linalg.norm(a)/torch.linalg.norm(b)
    distances = []
    displacements = []
    ending_simils = []
    loss_graph = []

    for sample in range(len(span_ws)):
        for perm in range(len(data[0])):
            permutation = losses[sample][perm][-1]
            order = [span_ws[sample][int(i)] for i in permutation]
            distances.append(sum(torch.abs(torch.acos(ang_bet(order[i], order[i+1]))) for i in range(7)))
            #print([torch.abs(torch.acos(ang_bet(order[i], order[i+1]))) for i in range(7)])
            displacements.append(data[sample][perm][0])
            ending_simils.append(sum(torch.abs(torch.acos(ang_bet(order[i], order[-1]))) for i in range(7)))
            loss_graph.append(losses[sample][perm][-2].sum())
    fig, axs = plt.subplots(1, 2, tight_layout = True)

    axs[0].scatter(distances, displacements, s = 0.1)
    axs[1].scatter(ending_simils, displacements, s = 0.1)
    plt.savefig("pn.png")


# z = sorted(zip([i.item() for i in (ws.mT[-1]-ws.mT[0])], [perm[0]-perm[-1] for perm in perms], [perm for perm in perms]), reverse= False)
# for s in z:
#     print(s)
