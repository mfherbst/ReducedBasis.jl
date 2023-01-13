# ReducedBasis

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mfherbst.github.io/ReducedBasis.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mfherbst.github.io/ReducedBasis.jl/dev/)
[![Build Status](https://github.com/mfherbst/ReducedBasis.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mfherbst/ReducedBasis.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/mfherbst/ReducedBasis.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mfherbst/ReducedBasis.jl)

[ReducedBasis.jl](https://github.com/mfherbst/ReducedBasis.jl) is a Julia package that uses [reduced basis methods](http://dx.doi.org/10.1007/978-3-319-22470-1) (RBM) to accelerate the modeling of parametrized eigenvalue problems.

In the RBM approach, a surrogate model is assembled by projecting the full problem onto a basis consisting of only a few tens of parameter snapshots, the only instances where the lowest eigenvectors, so-called snapshots, are computed.
The package focuses on a greedy strategy that selects snapshots by maximally reducing the estimated error with each additional snapshot.
Once the RBM surrogate is assembled, physical observables can be computed for any parameter value with only a modest complexity, which scales independently from the dimension of the initial eigenvalue problem.

Currently, the motivating application resides in quantum spin systems, following recent developments of [RBM approaches in quantum many-body physics](https://arxiv.org/abs/2110.15665).
Nonetheless, [ReducedBasis.jl](https://github.com/mfherbst/ReducedBasis.jl) is intended to be generally applicable to parametrized eigenvalue problems with a low-dimensional parameter space.
Key steps of the RBM procedure, such as the snapshot solving method, error estimates or the assembly strategy, can therefore be easily customized.
Moreover, the package integrates with [ITensors.jl](https://itensor.github.io/ITensors.jl/stable/) that allows the use of tensor network methods, in particular the [density matrix renormalization group](https://doi.org/10.48550/arXiv.1008.3477) using matrix product states.
