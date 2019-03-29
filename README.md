# inverse-gp
Solving inverse problems using Gaussian processes. The approach is similar to Bayesian optimisation.

[![Build Status](https://travis-ci.com/samuelmurray/inverse-gp.svg?token=metTeQBqcky3teaepvwx&branch=master)](https://travis-ci.com/samuelmurray/inverse-gp)

[![codecov](https://codecov.io/gh/samuelmurray/inverse-gp/branch/master/graph/badge.svg?token=UCU63YXn80)](https://codecov.io/gh/samuelmurray/inverse-gp)

## Installing
inverse-gp requires Python 3.6 or higher.

### Installing with Pipfile
The recommended way of installing inverse-gp is using the provided Pipfile. Clone the repository and install with pipenv:

```
$ git clone https://github.com/samuelmurray/inverse-gp
$ cd inverse-gp
$ pipenv sync
```

This will install all required dependencies in a new virtualenv.

For instructions on how to install and use pipenv, see https://pipenv.readthedocs.io/en/latest/.

### Installing with pip
To support installation with pip, we provide a requirements.txt file. Clone the repository, (optionnally) create a virtualenv, and install with pip:

```
$ git clone https://github.com/samuelmurray/inverse-gp
$ cd inverse-gp
$ # [Create and activate virtualenv]
$ pip install -r requirements.txt
```

## Examples
From the repository's root directory, run

```
$ python3 invgp/examples/run.py
```

This will create a GP with some given inputs (shown in first figure), then iteratively add new points according to the expected improvement of each input location. The final model is shown in the second plot, with the ordering of new points given in the third plot.

## Running tests
We use pytest and codecov for testing our code. First, install the extra dev requirements: 

```
$ pipenv sync --dev
```

Or using pip:

```
$ pip install -r requirements-dev.txt
```

Run pytest from the repository's root directory:

```
$ pytest --cov./
```

### Without codecov
To run the tests without codecov, run the following:

```
$ pytest
```
