import torch
data_tensor = torch.zeros(120, 2, 6, 5)
for i in range(0, 120, 15):
    o_data = torch.load(f"rotated_cifar_results/5_30_0_{i}_{i+15}.pt")
    data_tensor[i:i+15] = o_data.squeeze(0)

t = [torch.load(f"rotated_cifar_results/5_30.0_{i}_{i+15}_5.pt") for i in range(0, 120, 15)]
data_tensor5 = torch.cat(t, dim = 1)

torch.save(torch.cat([data_tensor.unsqueeze(0), data_tensor5], dim = 0), "30deg5tasks.pt")

    