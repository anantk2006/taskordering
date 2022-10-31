import torch
from torch.utils.data import DataLoader
from dataset import LRDataset


BATCH_SIZE = 50


def generate_data():
    """ Generate training features. """
    raise NotImplementedError


def generate_solution():
    """ Generate global ground-truth weights. """
    raise NotImplementedError


def generate_labels(features, W_star):
    """ Generate labels for given features based on ground-truth weights. """
    raise NotImplementedError


def train(features, W_star, labels, dataloaders):
    """ Run continual learning with a linear model. """
    raise NotImplementedError
    return {}


def main():
    """ Main function for linear continual learning experiments. """

    # Set up training data.
    features = generate_data()
    W_star = generate_solution()
    labels = generate_labels(features, W_star)
    dataloaders = [
        DataLoader(
            LRDataset(feat.to(device), lab.to(device)), batch_size=BATCH_SIZE, shuffle=True
        )
        for feat, lab in zip(features, labels)
    ]

    # Perform training and save results.
    results = train(features, W_star, labels, dataloaders)
    torch.save(results, f"../lgrgresults/results{SEED}.pt")


if __name__ == "__main__":
    main()
