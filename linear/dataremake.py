import torch
l = []

for i in range(10):
    #if i == 16: continue
    if torch.load(f"../lgrgresults/results{i}.pt")[0].mT[0].max()>1.3: continue
    l.append(torch.load(f"../lgrgresults/results{i}.pt"))
    
torch.save(l, "../lgrgresults/lgresrankedloss3.pt")
