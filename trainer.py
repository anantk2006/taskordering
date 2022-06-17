from torch.utils.data import DataLoader
from avalanche.benchmarks import SplitMNIST, PermutedMNIST, SplitCIFAR10
import avalanche as avl

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics
from torch.optim.lr_scheduler import MultiStepLR
from avalanche.training import ICaRL
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torchvision import transforms
import numpy as np
import torch
from models import SimpleCNN, MLP
def icarl_cifar100_augment_data(img):
    img = img.numpy()
    padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode='constant')
    random_cropped = np.zeros(img.shape, dtype=np.float32)
    crop = np.random.randint(0, high=8 + 1, size=(2,))

    # Cropping and possible flipping
    if np.random.randint(2) > 0:
        random_cropped[:, :, :] = \
            padded[:, crop[0]:(crop[0]+32), crop[1]:(crop[1]+32)]
    else:
        random_cropped[:, :, :] = \
            padded[:, crop[0]:(crop[0]+32), crop[1]:(crop[1]+32)][:, :, ::-1]
    t = torch.tensor(random_cropped)
    return t

def run(strat, train, test, benchmark, num_classes, num_channels = 3, device = "cuda", model = "simple"):
    patterns_per_exp = 256
    epochs = 1
    mem_strength = 0.5 
    
    train_mb_size = 10
    seed = 0
    sample_size = 1000
    memory_size = 5000
   
    if model == "simple":
        model = SimpleCNN(num_classes=num_classes, num_channels = num_channels)
    if model == "resnet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    if model == "mlp":
        model = MLP(num_classes=  num_classes, num_channels=  num_channels)

    criterion = CrossEntropyLoss()
    interactive_logger = avl.logging.InteractiveLogger()
    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True), 
            metrics.loss_metrics(epoch = True, experience= True, stream = True),
            #metrics.forward_transfer_metrics(experience=True),
            loggers=[interactive_logger], benchmark=benchmark)
    
    if strat == "naive":
        lr = 0.1
        cl_strategy = avl.training.Naive(
        model, SGD(model.parameters(), lr=lr, momentum=0.), criterion,
        train_mb_size=train_mb_size, train_epochs=epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin, plugins=[])
    if strat == "icarl":
        lr_base = 2
        lr_milestones = [49, 63]
        lr_factor = 5
        wght_decay = 0.00001
        optim = SGD(model.parameters(), lr=lr_base,
                weight_decay=wght_decay, momentum=0.9)
        sched = LRSchedulerPlugin(
            MultiStepLR(optim, lr_milestones, gamma=1.0 / lr_factor))
        cl_strategy = ICaRL(
            model.feature_extractor, model.classifier, optim,
            memory_size,
            buffer_transform=transforms.Compose([icarl_cifar100_augment_data]),
            fixed_memory=True, train_mb_size=train_mb_size,
            train_epochs=epochs, eval_mb_size=train_mb_size,
            plugins=[sched], device=device, evaluator=evaluation_plugin
        )
    if strat == "ewc":
        lr = 0.05
        cl_strategy = avl.training.EWC(
            model, SGD(model.parameters(), lr=lr, momentum=0.), criterion,
            train_mb_size=train_mb_size, train_epochs=epochs, eval_mb_size=128,
            device=device,evaluator=evaluation_plugin, ewc_lambda = 100000, plugins=[])
    if strat == "agem":
        lr = 0.05
        cl_strategy = avl.training.AGEM(
            model, SGD(model.parameters(), lr=lr, momentum=0.), criterion,
            patterns_per_exp=patterns_per_exp, sample_size=sample_size,
            train_mb_size=train_mb_size, train_epochs=epochs, eval_mb_size=128,
            device=device, evaluator=evaluation_plugin, plugins=[])
    

        
    
    
    res = None
    for experience in train:
        
        cl_strategy.train(experience)
        yield cl_strategy.eval(test)
    

    













