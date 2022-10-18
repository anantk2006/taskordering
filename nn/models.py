import torch.nn as nn
class ResNet(nn.Module):
    def __init__(self, num):
        super().__init__()

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, num_channels = 1):
        super(SimpleCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Linear(64, num_classes)


    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
class MLP(nn.Module):
    def __init__(self, num_classes = 10, num_channels = 3):
        super(MLP, self).__init__()
        self.seq = nn.Sequential(nn.Linear(784*num_channels, 784), nn.Linear(784, 256), nn.Linear(256, num_classes))
    def forward(self, X):
        return self.seq(X)
