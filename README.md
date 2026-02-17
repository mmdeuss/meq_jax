# MEQ-JAX: Magnetic EQuilibrium Solver in JAX

MEQ-JAX is a rewrite of the [MEQ Matlab package](https://gitlab.epfl.ch/spc/tcv/tbx/meq)
in JAX, enabling automatic differentiation, automatic batching, and running on
accelerators like GPU and TPU. Currently MEQ-JAX is experimental, and only
supports the [FGE](https://arxiv.org/abs/2512.06847) (Free-Boundary
Grad-Shafranov Evolutive) subset of MEQ, not FGS or FBT. MEQ-JAX is as close as
possible to an exact line-for-line rewrite of MEQ, with divergence where JAX
syntax and efficiency requires it. MEQ-JAX has not been significantly optimized,
and so is currently slower than MEQ, but there is significant room for
performance optimization and we welcome anyone who wants to improve it.

## Installation

For initialization and testing, the original Matlab MEQ package is still needed.
First, install [Octave](https://octave.org/) and then MEQ, following the
instructions in the README for [MEQPy](https://github.com/google-deepmind/meqpy).
Then, from inside the MEQ-JAX folder, simply run

```
python -m pip install .
```

Note that if you are using OpenBLAS, you need to disable multithreading, by
setting `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`, otherwise execution
will freeze. This is not an issue with MKL or Apple Accelerate.

A Dockerfile is also provided for even simpler installation.

## Example usage

We provide a minimal example which uses the [MEQPy package](https://github.com/google-deepmind/meqpy)
to initialize settings and then compares the JAX and Octave results. Simply run

```
MAT_ROOT=<path to MEQ> python meq_jax_example.py
```

Note that as currently implemented, MEQPy is not set up to initialize MEQ with
current diffusion equations. If a CDE is desired for your use case, you will
need to set up MEQ directly through calls to Octave instead of using the MEQPy
interface.

## Sharp edges

Note that, while we sought to have as broad test coverage as possible, many
branches and options remain untested, and some are not fully implemented. For
instance, doublets are not yet fully supported. Also note that shapes are almost
always transposed relative to Matlab. This is because during development, we
directly compared exported JAX code to MEX code in C, and Matlab arrays are
column-major while NumPy/JAX arrays are row-major. The `struct_to_dataclass`
function in `utils` converts an LY struct in Matlab format to a Python dataclass
in the format expected by MEQ-JAX, while `dataclass_to_octave_shape` converts
the FGE output back to Matlab shapes.

## Credit

MEQ-JAX is joint work by David Pfau, Antoine Dedieu, Joel Jennings, Wolfgang
Lehrach, Norman Casagrande, Matthias Lochbrunner, Tomek Nanowski, Craig Donner
and Federico Felici. Gemini was used extensively. To cite MEQ-JAX, use:

```
@software{meq_jax_gitlab,
  author = {David Pfau, Antoine Dedieu, Joel Jennings, Wolfgang Lehrach,
  Norman Casagrande, Matthias Lochbrunner, Tomek Nanowski, Craig Donner,
  Federico Felici and MEQ-JAX Contributors},
  title = {{MEQ-JAX}},
  url = {https://gitlab.epfl.ch/spc/tcv/tbx/meq_jax},
  year = {2026},
}
```


