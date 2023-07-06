# ReducedBasis.jl

ReducedBasis.jl is a Julia package that
uses the [reduced basis (RB) method](http://dx.doi.org/10.1007/978-3-319-22470-1)
to accelerate the solution of a parametrized eigenvalue problems across the parameter domain.

In the RB approach, a surrogate model is assembled by projecting the full
problem onto a basis consisting of only a few tens of parameter snapshots.
The package focuses on a greedy strategy that selects snapshots by maximally
reducing the estimated error with each additional snapshot.
Once the RB surrogate is assembled, physical observables can be computed for
any parameter value with only a modest complexity, which scales independently
from the dimension of the initial eigenvalue problem.

Currently, the motivating application resides in quantum spin systems,
following recent developments of
RB approaches in quantum many-body physics[^1][^2].
Nonetheless, ReducedBasis is intended to be generally applicable to parametrized eigenvalue
problems with a low-dimensional parameter space.
Key steps of the RB procedure, such as the snapshot solving method, error
estimates or the assembly strategy, can therefore be easily customized.
Moreover, the package integrates with [ITensors.jl](https://itensor.github.io/ITensors.jl/stable/)
that allows the use of tensor network methods,
in particular the [density matrix renormalization group](https://tensornetwork.org/mps/algorithms/dmrg/)
using matrix product states.

[^1]: M. F. Herbst, B. Stamm, S. Wessel, and M. Rizzi, Surrogate models for quantum spin systems based on reduced-order modeling, [Phys. Rev. E **105**, 045303 (2022)](https://link.aps.org/doi/10.1103/PhysRevE.105.045303).

[^2]: P. Brehmer, M. F. Herbst, S. Wessel, M. Rizzi, and B. Stamm, Reduced basis surrogates of quantum spin systems based on tensor networks (2023), [arXiv:2304.13587](https://doi.org/10.48550/arXiv.2304.13587).
