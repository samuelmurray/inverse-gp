import torch
import gpytorch

from gp import GP


def run():
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP(likelihood)


if __name__ == '__main__':
    run()
