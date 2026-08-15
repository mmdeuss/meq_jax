# Copyright 2026 The meq_jax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX root finding functions."""
# Branched from TORAX, with permission: https://torax.readthedocs.io/
import contextlib
import dataclasses
import functools
import os
from typing import Callable, Final

import jax
import jax.numpy as jnp
import numpy as np

# Delta is a vector. If no entry of delta is above this magnitude, we terminate
# the delta loop. This is to avoid getting stuck in an infinite loop in edge
# cases with bad numerics.
MIN_DELTA: Final[float] = 1e-7


def env_bool(name: str, default: bool) -> bool:
  """Get a bool from an environment variable.

  Args:
    name: The name of the environment variable.
    default: The default value of the bool.

  Returns:
    value: The value of the bool.
  """
  if name not in os.environ:
    return default
  str_value = os.environ[name]
  if str_value in ['1', 'True', 'true']:
    return True
  if str_value in ['0', 'False', 'false']:
    return False
  raise ValueError(f'Unrecognized boolean string {str_value}.')


# If True, `error_if` functions will raise errors.  Otherwise they are
# pass throughs.
# Default to False, because host_callbacks are incompatible with the
# persistent compilation cache.
_ERRORS_ENABLED: bool = env_bool('TORAX_ERRORS_ENABLED', False)


@functools.cache
def get_dtype() -> type(jnp.float32):
  # Default TORAX JAX precision is f64
  precision = os.getenv('JAX_PRECISION', 'f64')
  assert precision == 'f64' or precision == 'f32', (
      'Unknown JAX precision environment variable: %s' % precision
  )
  return jnp.float64 if precision == 'f64' else jnp.float32


@functools.cache
def get_int_dtype() -> type(jnp.int32):
  # Default TORAX JAX precision is f64
  precision = os.getenv('JAX_PRECISION', 'f64')
  assert precision == 'f64' or precision == 'f32', (
      'Unknown JAX precision environment variable: %s' % precision
  )
  return jnp.int64 if precision == 'f64' else jnp.int32


@contextlib.contextmanager
def enable_errors(value: bool):
  """Enables / disables `error_if` inside a code block.

  Example:

  with enable_errors(False):
    my_sim.run() # NaNs etc will be ignored

  Args:
    value: Sets `errors_enabled` to this value

  Yields:
    Cleanup function restoring previous value
  """
  global _ERRORS_ENABLED
  previous_value = _ERRORS_ENABLED
  _ERRORS_ENABLED = value
  yield
  if previous_value is not None:
    _ERRORS_ENABLED = previous_value


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RootMetadata:
  iterations: jax.Array
  residual: jax.Array
  last_tau: jax.Array
  error: jax.Array


def root_newton_raphson(
    fun: Callable[[jax.Array], jax.Array],
    x0: jax.Array | np.ndarray,
    *,
    maxiter: int = 30,
    tol: float = 1e-5,
    coarse_tol: float = 1e-2,
    delta_reduction_factor: float = 0.5,
    tau_min: float = 0.01,
    log_iterations: bool = False,
    use_jax_custom_root: bool = True,
    custom_jac: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, RootMetadata]:
  """A differentiable Newton-Raphson root finder.

  A similar API to scipy.optimize.root.

  Args:
    fun: The function to find the root of.
    x0: The initial guess of the location of the root.
    maxiter: Quit iterating after this many iterations reached.
    tol: Quit iterating after the average absolute value of the residual is <=
      tol.
    coarse_tol: Coarser allowed tolerance for cases when solver develops small
      steps in the vicinity of the solution.
    delta_reduction_factor: Multiply by delta_reduction_factor after each failed
      line search step.
    tau_min: Minimum delta/delta_original allowed before the newton raphson
      routine resets at a lower timestep.
    log_iterations: If true, output diagnostic information from within iteration
      loop.
    use_jax_custom_root: If true, use jax.lax.custom_root to allow for
      differentiable solving. This can increase compile times even when no
      derivatives are requested.
    custom_jac: If provided, use this function to compute the Jacobian of `fun`
      instead of jax.jacfwd.

  Returns:
    A tuple `(x_root, RootMetadata(...))`.
  """

  def _newton_raphson(f, x, jacobian_fun=None):
    residual_fun = f
    init_x_new_vec = x
    if jacobian_fun is None:
      jacobian_fun = jax.jacfwd(f)
    # initialize state dict being passed around Newton-Raphson iterations
    residual_vec_init_x_new = residual_fun(init_x_new_vec)
    initial_state = {
        'x': init_x_new_vec,
        # jax.lax.custom_root is broken with aux outputs of integer type. Use
        # float for the iterations https://github.com/jax-ml/jax/issues/24295.
        'iterations': jnp.array(0, dtype=get_dtype()),
        'residual': residual_vec_init_x_new,
        'last_tau': jnp.array(1.0, dtype=get_dtype()),
    }

    # carry out iterations.
    cond_fun = functools.partial(
        _cond, tol=tol, tau_min=tau_min, maxiter=maxiter
    )
    body_fun = functools.partial(
        _body,
        jacobian_fun=jacobian_fun,
        residual_fun=residual_fun,
        log_iterations=log_iterations,
        delta_reduction_factor=delta_reduction_factor,
    )
    output_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    x_out = output_state.pop('x')
    return x_out, output_state

  # jax.lax.custom_root allows for differentiating through the solver,
  # efficiently. As the solver has a jax.lax.while_loop, it cannot be
  # reverse-mode differentiated. But even if we could, this would be highly
  # inefficient. This uses the implicit function theorem to differentiate
  # through the solver with only needing the result of the solver,
  # rather than the entire solver computational graph.
  # See also this discussion:
  # https://docs.jax.dev/en/latest/advanced-autodiff.html#example-implicit-function-differentiation-of-iterative-implementations

  def back(g, y):
    return jnp.linalg.solve(jax.jacfwd(g)(y), y)

  if use_jax_custom_root:
    if custom_jac is not None:
      raise ValueError(
          'custom_jac is not compatible with use_jax_custom_root.'
      )
    x_out, metadata = jax.lax.custom_root(
        f=fun,
        initial_guess=x0,
        solve=_newton_raphson,
        tangent_solve=back,
        has_aux=True,
    )
  else:
    x_out, metadata = _newton_raphson(fun, x0, jacobian_fun=custom_jac)

  # Tell the caller whether or not x_new successfully reduces the residual below
  # the tolerance by providing an extra output, error.
  # error = 0: residual converged within fine tolerance (tol)
  # error = 1: not converged. Possibly backtrack to smaller dt and retry
  # error = 2: residual not strictly converged but is still within reasonable
  # tolerance (coarse_tol). Can occur when solver exits early due to small steps
  # in solution vicinity. Proceed but provide a warning to user.
  error = _error_cond(
      residual=metadata['residual'], coarse_tol=coarse_tol, tol=tol
  )
  # Workaround for https://github.com/google/jax/issues/24295: cast iterations
  # to the correct int dtype.
  metadata['iterations'] = metadata['iterations'].astype(get_int_dtype())
  return x_out, RootMetadata(**metadata, error=error)  # pytype: disable=bad-return-type


def _error_cond(residual: jax.Array, coarse_tol: float, tol: float):
  return jax.lax.cond(
      _residual_scalar(residual) < tol,
      lambda: 0,  # Called when True
      lambda: jax.lax.cond(  # Called when False
          _residual_scalar(residual) < coarse_tol,
          lambda: 2,  # Called when True
          lambda: 1,  # Called when False
      ),
  )


def _residual_scalar(x):
  # Matches the MEQ convergence criterion max(abs(F(x))) (see fgep.m tolF).
  return jnp.max(jnp.abs(x))


def _cond(
    state: dict[str, jax.Array],
    tau_min: float,
    maxiter: int,
    tol: float,
) -> bool:
  """Check if exit condition reached for Newton-Raphson iterations."""
  iteration = state['iterations'][...]
  return jnp.bool_(
      jnp.logical_and(
          jnp.logical_and(
              _residual_scalar(state['residual']) > tol, iteration < maxiter
          ),
          state['last_tau'] > tau_min,
      )
  )


def _body(
    input_state: dict[str, jax.Array],
    jacobian_fun: Callable[[jax.Array], jax.Array],
    residual_fun: Callable[[jax.Array], jax.Array],
    log_iterations: bool,
    delta_reduction_factor: float,
) -> dict[str, jax.Array]:
  """Calculates next guess in Newton-Raphson iteration."""
  dtype = input_state['x'].dtype
  a_mat = jacobian_fun(input_state['x'])
  rhs = -input_state['residual']
  # delta = x_new - x_old
  # tau = delta/delta0, where delta0 is the delta that sets the linearized
  # residual to zero. tau < 1 when needed such that x_new meets
  # conditions of reduced residual and valid state quantities.
  # If tau < taumin while residual > tol, then the routine exits with an
  # error flag, leading to either a warning or recalculation at lower dt
  initial_delta_state = {
      'x': input_state['x'],
      'delta': jnp.linalg.solve(a_mat, rhs),
      'residual_old': input_state['residual'],
      'residual_new': input_state['residual'],
      'tau': jnp.array(1.0, dtype=dtype),
  }
  output_delta_state = _compute_output_delta_state(
      initial_delta_state, residual_fun, delta_reduction_factor
  )

  output_state = {
      'x': input_state['x'] + output_delta_state['delta'],
      'residual': output_delta_state['residual_new'],
      'iterations': jnp.array(input_state['iterations'][...], dtype=dtype) + 1,
      'last_tau': output_delta_state['tau'],
  }
  if log_iterations:
    jax.debug.print(
        'Iteration: {iteration:d}. Residual: {residual:.16f}. tau = {tau:.6f}',
        iteration=output_state['iterations'].astype(get_int_dtype()),
        residual=_residual_scalar(output_state['residual']),
        tau=output_delta_state['tau'],
    )

  return output_state


def _compute_output_delta_state(
    initial_state: dict[str, jax.Array],
    residual_fun: Callable[[jax.Array], jax.Array],
    delta_reduction_factor: float,
):
  """Updates output delta state.

  The residual at the candidate point is computed once per candidate and
  carried in the loop state (rather than recomputed inside the loop condition
  and again after the loop), so each backtracking round costs exactly one
  residual evaluation.
  """
  # Avoid sanity checking inside residual, since we directly
  # afterwards check sanity on the output (NaN checking)
  # TODO(b/312453092) consider instead sanity-checking x_new
  with enable_errors(False):
    initial_state = initial_state | dict(
        residual_new=residual_fun(initial_state['x'] + initial_state['delta']),
    )

    delta_body_fun = functools.partial(
        _delta_body,
        residual_fun=residual_fun,
        delta_reduction_factor=delta_reduction_factor,
    )
    output_delta_state = jax.lax.while_loop(
        _delta_cond, delta_body_fun, initial_state
    )
  return output_delta_state


def _delta_cond(delta_state: dict[str, jax.Array]) -> bool:
  """Check if delta obtained from Newton step is valid.

  Args:
    delta_state: see `delta_body`.

  Returns:
    True if the new value of `x` causes any NaNs or has increased the residual
    relative to the old value of `x`.
  """
  residual_scalar_x_old = _residual_scalar(delta_state['residual_old'])
  residual_scalar_x_new = _residual_scalar(delta_state['residual_new'])
  return jnp.bool_(
      jnp.logical_and(
          jnp.max(jnp.abs(delta_state['delta'])) > MIN_DELTA,
          jnp.logical_or(
              residual_scalar_x_old < residual_scalar_x_new,
              jnp.isnan(residual_scalar_x_new),
          ),
      ),
  )


def _delta_body(
    input_delta_state: dict[str, jax.Array],
    residual_fun: Callable[[jax.Array], jax.Array],
    delta_reduction_factor: float,
) -> dict[str, jax.Array]:
  """Reduces step size for this Newton iteration."""
  delta = input_delta_state['delta'] * delta_reduction_factor
  return input_delta_state | dict(
      delta=delta,
      tau=jnp.array(input_delta_state['tau'][...], dtype=get_dtype())
      * delta_reduction_factor,
      residual_new=residual_fun(input_delta_state['x'] + delta),
  )