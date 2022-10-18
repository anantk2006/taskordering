import torch
l = []
l2 = []
for i in range(10, 50):
    l.append(torch.load(f"lgrgresults/res{i}30.pt"))
    l2.append(torch.load(f"lgrgresults/losses{i}30.pt"))
torch.save(l, "lgrgresults/lg_8_res.pt")
torch.save(l2, "lgrgresults/lg_8_losses.pt")