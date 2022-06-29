import torch


t = [torch.load(f"cifar_results/rot_5_0_{i}_{i+15}_5.pt") for i in range(0, 120, 15)]
data_tensor5 = torch.cat(t, dim = 1)

torch.save(data_tensor5, "cifar_results/003noise5tasks.pt")

    