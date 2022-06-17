from torch.utils.data import DataLoader
from avalanche.benchmarks import SplitMNIST, PermutedMNIST, SplitCIFAR10
import avalanche as avl
from avalanche.models import SimpleCNN
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics

def run(train, test, benchmark, num_classes, num_channels):
    patterns_per_exp = 256
    epochs = 1
    mem_strength = 0.5 
    learning_rate = 0.1
    train_mb_size = 10
    seed = 0
    sample_size = 1000
    model = SimpleCNN(num_classes=num_classes)

    criterion = CrossEntropyLoss()
    interactive_logger = avl.logging.InteractiveLogger()
    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True), 
            metrics.loss_metrics(epoch = True, experience= True, stream = True),
            #metrics.forward_transfer_metrics(experience=True),
            loggers=[interactive_logger], benchmark=benchmark)
#     cl_strategy = avl.training.AGEM(
#             model, SGD(model.parameters(), lr=learning_rate, momentum=0.), criterion,
#             patterns_per_exp=patterns_per_exp, sample_size=sample_size,
#             train_mb_size=train_mb_size, train_epochs=epochs, eval_mb_size=128,
#             device="cpu", evaluator=evaluation_plugin, plugins=[])
    cl_strategy = avl.training.Naive(
        model, SGD(model.parameters(), lr=learning_rate, momentum=0.), criterion,
        train_mb_size=train_mb_size, train_epochs=epochs, eval_mb_size=128,
        device="cuda", evaluator=evaluation_plugin, plugins=[])
    
    
    res = None
    for experience in train:
        
        cl_strategy.train(experience)
        yield cl_strategy.eval(test)
    

    













