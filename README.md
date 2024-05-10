# ReducedBasis

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mfherbst.github.io/ReducedBasis.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mfherbst.github.io/ReducedBasis.jl/dev/)
[![Build Status](https://github.com/mfherbst/ReducedBasis.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mfherbst/ReducedBasis.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/mfherbst/ReducedBasis.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mfherbst/ReducedBasis.jl)

ReducedBasis.jl is a Julia package that uses the reduced basis (RB) method to
accelerate the solution of a parametrized eigenvalue problems across the
parameter domain.

In the RB approach, a surrogate model is assembled by projecting the full
problem onto a basis consisting of only a few tens of parameter snapshots.
The package focuses on a greedy strategy that selects snapshots by maximally
reducing the estimated error with each additional snapshot.
Once the RB surrogate is assembled, observables or post-processing can proceed
at any parameter value with only a modest complexity scaling independently
from the dimension of the initial eigenvalue problem.

For more details [see our documentation](https://mfherbst.github.io/ReducedBasis.jl/stable/).

If you find this work useful, please cite:
```
@article{Brehmer2023,
  title = {Reduced basis surrogates for quantum spin systems based on tensor networks},
  author = {Brehmer, Paul and Herbst, Michael F. and Wessel, Stefan and Rizzi, Matteo and Stamm, Benjamin},
  journal = {Phys. Rev. E},
  volume = {108},
  issue = {2},
  pages = {025306},
  numpages = {14},
  year = {2023},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.108.025306},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.108.025306}
}
```
