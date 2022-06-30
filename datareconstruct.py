import torch


t = [torch.load(f"cifar_results/cifar_rot_5_30_{i}_{i+15}_1_20.pt") for i in range(0, 120, 15)]
data_tensor5 = torch.cat(t, dim = 1)

torch.save(data_tensor5, "cifar_results/cifar_rot_5_30_all_1_20.pt")

    