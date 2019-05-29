import gpytorch
import torch
from matplotlib import pyplot as plt

from invgp.acquisition_function import ExpectedImprovement
from invgp.model import GP, SimulatorGP
from invgp.simulator import FlowSimulator, HeavySimulator, SimpleSimulator


def train(model: GP, likelihood: gpytorch.likelihoods.Likelihood) -> None:
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


def run() -> None:
    dt = 0.01
    nt = 200
    heavy_simulator = FlowSimulator(dt, nt)
    simple_simulator = SimpleSimulator()
    x = torch.linspace(0, 1, 20).unsqueeze(1)
    y = heavy_simulator(x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SimulatorGP(x, y, likelihood, simple_simulator)

    train(model, likelihood)

    acquisition_function = ExpectedImprovement(model)

    max_iter = 10
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8))
    ax1.scatter(x.numpy()[:, 0], y.numpy(), c='k')
    ax1.set_xlim([-1, 2])
    ax2.set_xlim([-1, 2])
    ax3.set_xlim([-1, 2])
    ax1.set_ylim([-3, 3])
    ax2.set_ylim([-3, 3])
    for i in range(max_iter):
        candidate_set = torch.linspace(-1, 2, 100).unsqueeze(1)
        expected_improvement = acquisition_function(x, y, candidate_set)
        best_index = torch.argmax(expected_improvement)
        x_new = candidate_set[best_index].unsqueeze(1)
        y_new = heavy_simulator(x_new)
        x = torch.cat((x, x_new), 0)
        y = torch.cat((y, y_new), 0)
        ax3.scatter(x_new, expected_improvement[best_index], c='k', marker='*')
        ax3.annotate(i, (x_new, expected_improvement[best_index]))

    with torch.no_grad():
        test_x = torch.linspace(-1, 2, 200).unsqueeze(1)
        observed_pred = likelihood(model(test_x))
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax2.scatter(x.numpy()[:, 0], y.numpy(), c='k')
    # Plot predictive means as blue line
    ax2.plot(test_x.numpy()[:, 0], observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax2.fill_between(test_x.numpy()[:, 0], lower.numpy(), upper.numpy(), alpha=0.5)
    ax1.legend(['Initial Data'])
    ax2.legend(['Mean', 'Observed Data', 'Confidence'])
    ax3.legend(['Expected Improvement'])
    plt.show()


if __name__ == '__main__':
    run()
