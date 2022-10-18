import torch
l = []
l2 = []
l3 = []
for i in (set(range(10,20))-{16,13}):
    #if i == 16: continue
    print(torch.load(f"lgrgresults/res{i}30.pt")[1])
    l.append(torch.load(f"lgrgresults/res{i}30.pt"))
    l2.append(torch.load(f"lgrgresults/losses{i}30.pt"))
    l3.append(torch.load(f"lgrgresults/span_ws{i}30.pt"))
torch.save(l, "lgrgresults/lgres21.pt")
torch.save(l2, "lgrgresults/lglosses21.pt")
torch.save(l3, "lgrgresults/lgws21")