import torch
import gpytorch

from gp import GP
from acquisition_function import AcquisitionFunction
from simple_simulator import SimpleSimulator
from simulator import Simulator


def train(model, likelihood):
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(model.get_inputs())
        # Calc loss and backprop gradients
        loss = -mll(output, model.train_targets)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covariance_module.base_kernel.log_lengthscale.item(),
            model.likelihood.log_noise.item()
        ))
        optimizer.step()


def run():
    simulator = Simulator()
    simple_simulator = SimpleSimulator()
    x = torch.linspace(0, 1, 20)
    y = simulator(x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP(x, y, likelihood)
    print(model.get_inputs()[0])

    train(model, likelihood)

    acquisition_function = AcquisitionFunction(model, simple_simulator)

    max_iter = 10
    for i in range(max_iter):
        candidate_set = torch.linspace(-1, 2, 100)
        expected_improvement = acquisition_function(x, y, candidate_set)
        best_index = torch.argmax(expected_improvement)
        x_new = torch.unsqueeze(candidate_set[best_index], 0)
        y_new = simulator(x_new)
        x = torch.cat((x, x_new), 0)
        y = torch.cat((y, y_new), 0)


if __name__ == '__main__':
    run()
