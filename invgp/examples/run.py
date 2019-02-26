import time

import gpytorch
from matplotlib import pyplot as plt
import torch

from invgp.acquisition_function import ExpectedImprovement
from invgp.model import GP
from invgp.simulator import HeavySimulator, SimpleSimulator


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
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covariance_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()


def run():
    heavy_simulator = HeavySimulator()
    simple_simulator = SimpleSimulator()
    x = torch.linspace(0, 1, 20)
    y = heavy_simulator(x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP(x, y, likelihood)
    print(model.get_inputs()[0])

    train(model, likelihood)

    acquisition_function = ExpectedImprovement(model, simple_simulator)

    max_iter = 10
    for i in range(max_iter):
        candidate_set = torch.linspace(-1, 2, 100)
        expected_improvement = acquisition_function(x, y, candidate_set)
        best_index = torch.argmax(expected_improvement)
        x_new = torch.unsqueeze(candidate_set[best_index], 0)
        y_new = heavy_simulator(x_new)
        x = torch.cat((x, x_new), 0)
        y = torch.cat((y, y_new), 0)

        with torch.no_grad():
            test_x = torch.linspace(-1, 2, 200)
            observed_pred = likelihood(model(test_x))
            # Initialize plot
            f, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            ax1.plot(x.numpy(), y.numpy(), 'k*')
            # Plot predictive means as blue line
            ax1.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax1.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax1.set_ylim([-3, 3])
            ax1.legend(['Observed Data', 'Mean', 'Confidence'])

            ax2.plot(candidate_set.numpy(), expected_improvement.numpy())
            ax2.scatter(x_new, expected_improvement[best_index])

            plt.show()
            time.sleep(1)


if __name__ == '__main__':
    run()
