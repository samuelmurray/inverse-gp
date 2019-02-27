# inverse-gp
Solving inverse problems using Gaussian processes. The approach is similar to Bayesian optimisation.

[![Build Status](https://travis-ci.com/samuelmurray/inverse-gp.svg?token=metTeQBqcky3teaepvwx&branch=master)](https://travis-ci.com/samuelmurray/inverse-gp)

[![codecov](https://codecov.io/gh/samuelmurray/inverse-gp/branch/master/graph/badge.svg?token=UCU63YXn80)](https://codecov.io/gh/samuelmurray/inverse-gp)

## Installing with pip
inverse-gp requires Python 3.6 or higher.
To install inverse-gp, clone the repository and install with git:

```
$ git clone https://github.com/samuelmurray/inverse-gp
$ pip install -e .
```

This will install all required dependencies. 

## Examples
In order to run the examples, some additional packages are required, listed in `setup.py`. Install those separately, or run

```
$ pip install -e .[examples]
```

From the root directory of repository, run

```
$ python3 invgp/examples/run.py
```

This will create a GP with some given inputs (shown in first figure), then iteratively add new points according to the expected improvement of each input location. The final model is shown in the second plot, with the ordering of new points given in the third plot.

## Running tests
We use pytest and codecov for testing our code. First, install pytest, pytest-cov and codecov: 

```
$ pip install pytest pytest-cov codecov
```

Run pytest from root directory of repository:

```
$ pytest --cov./
```

### Without codecov
To run the tests without codecov, run the following:

```
$ pip install pytest
$ pytest
```