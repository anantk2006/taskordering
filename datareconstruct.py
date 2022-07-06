import torch
full_data = []
for i in range(120):
    index_tensor = []
    for j in range(5):
        index_tensor.append(torch.load(f"mnist_results/mnist_noise_5_0_{i}_{j}_resnet.pt").unsqueeze(0))
    full_data.append(torch.cat(index_tensor, dim = 0).unsqueeze(0))
torch.save(torch.cat(full_data, dim = 0), "mnist_results/mnist_noise_5_003_all_5_resnet.pt")
