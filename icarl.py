import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import SGD
from torchvision import transforms



from avalanche.benchmarks import SplitCIFAR100
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import *
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.training import ICaRL


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


def run(train, test, benchmark, num_classes):
    
    nb_exp = 5
    batch_size = 10
    memory_size = 500
    epochs = 1
    lr_base = 2
    lr_milestones = [49, 63]
    lr_factor = 5
    wght_decay = 0.00001
    train_mb_size = 256
    seed = 2222

    
    
    device = torch.device(f"cuda:{0}"
                          if torch.cuda.is_available() else "cpu")

    

    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(experience=True, stream=True),
        benchmark=benchmark,
        loggers=[interactive_logger])

    # _____________________________Strategy
    model = make_icarl_net(num_classes=num_classes)
    model.apply(initialize_icarl_net)

    optim = SGD(model.parameters(), lr=lr_base,
                weight_decay=wght_decay, momentum=0.9)
    sched = LRSchedulerPlugin(
        MultiStepLR(optim, lr_milestones, gamma=1.0 / lr_factor))

    strategy = ICaRL(
        model.feature_extractor, model.classifier, optim,
        memory_size,
        buffer_transform=transforms.Compose([icarl_cifar100_augment_data]),
        fixed_memory=True, train_mb_size=batch_size,
        train_epochs=epochs, eval_mb_size=batch_size,
        plugins=[sched], device=device, evaluator=eval_plugin
    )
    # Dict to iCaRL Evaluation Protocol: Average Incremental Accuracy
    
    # ___________________________________________train and eval
    for i, exp in enumerate(train):
        strategy.train(exp, num_workers=4)
        res = strategy.eval(test, num_workers=4)
        yield res

    



