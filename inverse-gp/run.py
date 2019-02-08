import torch
import gpytorch

from gp import GP

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
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP(likelihood)


if __name__ == '__main__':
    run()
