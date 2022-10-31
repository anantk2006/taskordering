from torch.utils.data import Dataset


class LRDataset(Dataset):
    def __init__(self, X, Y):
        """
        Initializing Dataset to be placed inside DataLoader.         
        X and Y and features and labels respectively.
        """
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
