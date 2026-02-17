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

"""JAX implementation of bspsum."""

import jax
import jax.numpy as jnp
import jaxtyping as jt


at = lambda x, i: jax.lax.dynamic_slice(x, (i,), (1,))[0]


# pylint: disable=invalid-name
def _bspsum_single(t, c, x, d):
  """Evaluates a B-spline combination for a single point."""
  nT = t.shape[0]
  nC = c.shape[0]
  k = nT - nC

  # Find the index j in t such that t[j] <= x < t[j+1]
  j = jnp.argmin(x > t, axis=0) - 1
  j = jnp.clip(j, k-1, nT-k)

  # Load coefficients into w
  # While somewhat inelegant, the padding allows shape polymorphism without
  # adding a constraint.
  w = jax.lax.dynamic_slice(jnp.pad(c, (k-1, 0)), (j,), (k,))[::-1]

  # Derivatives
  def body_fun_deriv(r, w_r):
    kk = k-1-r
    def body_fun_inner_deriv(s, w_s):
      o = at(t, j-s+kk) - at(t, j-s)
      term = jnp.where(o == 0, 0, (w_s[s] - w_s[s + 1]) * kk / o)
      return w_s.at[s].set(term)

    w_r = jax.lax.fori_loop(0, k-r-1, body_fun_inner_deriv, w_r)
    return w_r

  w = jax.lax.fori_loop(0, d, body_fun_deriv, w)

  # Evaluation
  def body_fun_eval(r, w_r):
    kk = k-d-r
    # Inner loop for evaluation
    def body_fun_inner_eval(s, w_s):
      ti = at(t, j-s)
      tj = at(t, j-s+kk-1)
      o = tj-ti
      o = jnp.where(o == 0, 0, (x-ti) / o)
      term = w_s[s] * o + (1-o) * w_s[s+1]
      return w_s.at[s].set(term)

    w_r = jax.lax.fori_loop(0, kk-1, body_fun_inner_eval, w_r)
    return w_r

  w = jax.lax.fori_loop(0, k-d-1, body_fun_eval, w)

  return w[0]


def bspsum(
    T: jt.Float[jax.Array, "nT"],
    C: jt.Float[jax.Array, "mC nC"],
    X: jt.Float[jax.Array, "nX"],
    d: jnp.float64 = 0,
    p: jnp.float64 = 0,
    npar: jnp.float64 = 1,
) -> jt.Float[jax.Array, "mC nX"]:
  """Evaluates B-spline base function combinations.

  Args:
      T: Knot sequence.
      C: Coefficients of the combination.
      X: Points at which to evaluate.
      d: Derivative order.
      p: Flag for alternative C/X shape handling (0 or 1).
      npar: Number of threads (not used in JAX implementation).

  Returns:
      Array of evaluated spline combinations.
  """
  del p, npar

  # k = T.shape[0] - C.shape[0]
  # if d >= k or X.shape[0] == 0:
  #   return jnp.zeros((X.shape[0], C.shape[1]), dtype=jnp.float64)

  # C is [mC, nC], X is [nX]
  # Output should be [mC, nX]
  # (This is reversed from the Matlab implementation)
  def eval_single_c(c_col):
    return jax.vmap(_bspsum_single,
                    in_axes=(None, None, 0, None))(
                        T, c_col, X, jnp.array(d).astype(jnp.int32))

  # Vectorize over the columns of C
  return jax.vmap(eval_single_c, in_axes=0, out_axes=0)(C)


def bspsum2(
    T: jt.Float[jax.Array, "nT"],
    C: jt.Float[jax.Array, "mC nC"],
    X: jt.Float[jax.Array, "nX"],
    d: jnp.float64,
) -> jt.Float[jax.Array, "mC nX"]:
  return bspsum(T, C, X, d, 0., 1.)


def bspsum3(
    T: jt.Float[jax.Array, "nT"],
    C: jt.Float[jax.Array, "mC nC"],
    X: jt.Float[jax.Array, "nX"],
) -> jt.Float[jax.Array, "mC nX"]:
  return bspsum2(T, C, X, 0.)
