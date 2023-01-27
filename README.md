# ReducedBasis

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mfherbst.github.io/ReducedBasis.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mfherbst.github.io/ReducedBasis.jl/dev/)
[![Build Status](https://github.com/mfherbst/ReducedBasis.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mfherbst/ReducedBasis.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/mfherbst/ReducedBasis.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mfherbst/ReducedBasis.jl)

ReducedBasis.jl is a Julia package that uses the reduced basis method to
accelerate the solution of a parametrized eigenvalue problems across the
parameter domain.

In the RBM approach, a surrogate model is assembled by projecting the full
problem onto a basis consisting of only a few tens of parameter snapshots.
The package focuses on a greedy strategy that selects snapshots by maximally
reducing the estimated error with each additional snapshot.
Once the RBM surrogate is assembled, observables or post-processing can proceed
at any parameter value with only a modest complexity scaling independently
from the dimension of the initial eigenvalue problem.

For more details [see our documentation](https://mfherbst.github.io/ReducedBasis.jl/stable/).
