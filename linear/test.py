import torch
import numpy

import main


def test_nullspaces_and_rank_good():

    features, span_ws = main.generate_data(datagen="good")
    passed = True
    for ind, feat in enumerate(features):
        m = torch.max(torch.abs(feat@span_ws[ind]))
        if m>0.01:
            print(feat@span_ws[ind])
            passed = False
        print(numpy.linalg.matrix_rank(feat))

    return passed, span_ws, features


def test_nullspaces_and_rank_general():

    features, span_ws = main.generate_data(datagen="general")
    passed = True
    for ind, feat in enumerate(features):
        m = torch.max(torch.abs(feat@span_ws[ind]))
        if m>0.01:
            print(feat@span_ws[ind])
            passed = False
        print(numpy.linalg.matrix_rank(feat))

    return passed, span_ws, features


def test_labels(features, span_ws):
    W_star, labels = main.generate_solution_labels(features, span_ws)
    for ind, lab in enumerate(labels):
        if (torch.where(features[ind]@W_star.unsqueeze(-1)>0, 1, 0)-lab).sum()>0.01:
            return False, labels, W_star

        if lab.sum()>lab.numel()/1.5 or lab.sum()<lab.numel()/3:
            print(lab.sum())
            return False, labels, W_star
    return True, labels, W_star


if __name__ == "__main__":
    test1, span_ws, features = test_nullspaces_and_rank_good()
    test2, span_ws, features = test_nullspaces_and_rank_general()
    test3, labels, W_star = test_labels(features, span_ws)
    print(test1, test2, test3)
