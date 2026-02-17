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

"""cs2deval majic implementation."""

import jax
import jax.numpy as jnp
import jaxtyping as jt


# pylint: disable=invalid-name
def csint(rho, d, fac=1.0):
  """Computes the 1D cubic B-spline basis functions and their derivatives.

  Args:
      rho: Relative position within the knot interval [0, 1).
      d: Derivative order (0, 1, 2, or 3).
      fac: Factor for scaling the derivatives.

  Returns:
      A tuple (c0, c1, c2, c3) representing the basis function values or
      derivatives.
  """
  # TODO(pfau): Replace this all with JAX autodiff.
  def d0_fn(rho):
    c0 = rho
    c3 = 1.0 - rho
    c1 = 0.5 * (c0 * (2.0 - rho) + c3 * (rho + 1.0))
    c0 = 0.5 * (c0 * rho)
    c3 = 0.5 * (c3 * (1.0 - rho))
    c2 = 1.0 / 3.0 * (c1 * (2.0 - rho) + c3 * (rho + 2.0))
    c1 = 1.0 / 3.0 * (c0 * (3.0 - rho) + c1 * (rho + 1.0))
    c0 = 1.0 / 3.0 * (c0 * rho)
    c3 = 1.0 / 3.0 * (c3 * (1.0 - rho))
    return jnp.array([c3, c2, c1, c0])

  def d1_fn(rho, fac):
    c0 = (rho) * fac
    c3 = (1.0 - rho) * fac
    c1 = 0.5 * (c0 * (2.0 - rho) + c3 * (rho + 1.0))
    c0 = 0.5 * (c0 * rho)
    c3 = 0.5 * (c3 * (1.0 - rho))
    c2 = c3 - c1
    c1 = c1 - c0
    c3 = -c3
    return jnp.array([c3, c2, c1, c0])

  def d2_fn(rho, fac):
    c0 = (rho) * fac
    c1 = (1.0 - 3.0 * rho) * fac
    c2 = (3.0 * rho - 2.0) * fac
    c3 = (1.0 - rho) * fac
    return jnp.array([c3, c2, c1, c0])

  def d3_fn(fac):
    c0 = fac
    c1 = -3.0 * fac
    c2 = 3.0 * fac
    c3 = -fac
    return jnp.array([c3, c2, c1, c0])

  return jnp.where(
      d == 0,
      d0_fn(rho),
      jnp.where(
          d == 1,
          d1_fn(rho, fac),
          jnp.where(
              d == 2,
              d2_fn(rho, fac),
              d3_fn(fac),
          ),
      ),
  )


def _cs2deval_normal_branch(Tx, Ty, C, xp, yp):
  """Evaluates a 2D cubic B-spline and its derivatives at a single point."""
  # Remember NumPy/JAX is row-major while Matlab is column-major
  nCx = C.shape[1]
  nCy = C.shape[0]

  idx = jnp.where(Tx[0] == Tx[1], 0.0, 1.0 / (Tx[1] - Tx[0]))
  idy = jnp.where(Ty[0] == Ty[1], 0.0, 1.0 / (Ty[1] - Ty[0]))

  # TODO(pfau): make the return signature depend on dmax

  ix = jnp.floor((xp - Tx[0]) * idx).astype(int)
  ix = jnp.clip(ix, 3, nCx - 1)

  iy = jnp.floor((yp - Ty[0]) * idy).astype(int)
  iy = jnp.clip(iy, 3, nCy - 1)

  # Although clunky, need the dynamic slice to work with vmap.
  dx = (xp - jax.lax.dynamic_slice(Tx, (ix,), (1,))[0]) * idx
  dy = (yp - jax.lax.dynamic_slice(Ty, (iy,), (1,))[0]) * idy

  a0 = csint(dx, 0, 1.0)
  b0 = csint(dy, 0, 1.0)

  C_slice = jax.lax.dynamic_slice(C, (iy - 3, ix - 3), (4, 4)).T

  c0 = a0 @ C_slice
  v = jnp.dot(b0, c0)

  facx = idx
  facy = idy
  a1 = csint(dx, 1, facx)
  b1 = csint(dy, 1, facy)

  c1 = a1 @ C_slice
  vx = jnp.dot(b0, c1)
  vy = jnp.dot(b1, c0)

  facx *= idx
  facy *= idy
  a2 = csint(dx, 2, facx)
  b2 = csint(dy, 2, facy)

  c2 = a2 @ C_slice
  vxx = jnp.dot(b0, c2)
  vxy = jnp.dot(b1, c1)
  vyy = jnp.dot(b2, c0)

  facx *= idx
  facy *= idy
  a3 = csint(dx, 3, facx)
  b3 = csint(dy, 3, facy)

  c3 = a3 @ C_slice
  vxxx = jnp.dot(b0, c3)
  vxxy = jnp.dot(b1, c2)
  vxyy = jnp.dot(b2, c1)
  vyyy = jnp.dot(b3, c0)

  vxxxy = jnp.dot(b1, c3)
  vxxyy = jnp.dot(b2, c2)
  vxyyy = jnp.dot(b3, c1)
  vxxxyy = jnp.dot(b2, c3)
  vxxyyy = jnp.dot(b3, c2)
  vxxxyyy = jnp.dot(b3, c3)

  return tuple(
      x.astype(C.dtype)
      for x in [
          v,
          vx,
          vy,
          vxx,
          vxy,
          vyy,
          vxxx,
          vxxy,
          vxyy,
          vyyy,
          vxxxy,
          vxxyy,
          vxyyy,
          vxxxyy,
          vxxyyy,
          vxxxyyy,
      ]
  )


def _cs2deval(Tx, Ty, C, xp, yp):
  """Evaluates a 2D cubic B-spline."""

  def mask_branch(Tx_in, Ty_in, C_in, xp_in, yp_in):
    del Tx_in, Ty_in, yp_in
    return tuple(
        jnp.full_like(xp_in, -jnp.inf, dtype=C_in.dtype) for _ in range(16)
    )

  return jax.lax.cond(
      jnp.isinf(xp) | jnp.isinf(yp),
      mask_branch,
      _cs2deval_normal_branch,
      Tx,
      Ty,
      C,
      xp,
      yp,
  )


def cs2devalmex(
    Tx: jt.Float[jax.Array, "nCx_plus_4"],
    Ty: jt.Float[jax.Array, "nCy_plus_4"],
    C: jt.Float[jax.Array, "nCy nCx"],
    xp: jt.Float[jax.Array, "np"],
    yp: jt.Float[jax.Array, "np"],
    dmax: jnp.float64,
) -> tuple[
    jt.Float[jax.Array, "np"],  # V
    jt.Float[jax.Array, "np"],  # Vx
    jt.Float[jax.Array, "np"],  # Vy
    jt.Float[jax.Array, "np"],  # Vxx
    jt.Float[jax.Array, "np"],  # Vxy
    jt.Float[jax.Array, "np"],  # Vyy
    jt.Float[jax.Array, "np"],  # Vxxx
    jt.Float[jax.Array, "np"],  # Vxxy
    jt.Float[jax.Array, "np"],  # Vxyy
    jt.Float[jax.Array, "np"],  # Vyyy
    jt.Float[jax.Array, "np"],  # Vxxxy
    jt.Float[jax.Array, "np"],  # Vxxyy
    jt.Float[jax.Array, "np"],  # Vxyyy
    jt.Float[jax.Array, "np"],  # Vxxxyy
    jt.Float[jax.Array, "np"],  # Vxxyyy
    jt.Float[jax.Array, "np"],  # Vxxxyyy
]:
  """Evaluates a 2D cubic B-spline and its derivatives.

  Args:
      Tx: Knot vector in x (regularly spaced).
      Ty: Knot vector in y (regularly spaced).
      C: Coefficient matrix.
      xp: x-coordinates for evaluation.
      yp: y-coordinates for evaluation.
      dmax: Maximum derivative order to compute other than 0, 1, 2, or 3.

  Returns:
      A tuple of arrays: (V, Vx, Vy, Vxx, Vxy, Vyy)
      with the value of the interpolated points and their derivatives. Note that
      the original MEX function goes up to third derivatives, but MEQ only ever
      uses second derivatives, so we fix the return signature for now.
  """
  del dmax  # Only here for consistency with the MEX interface.
  return jax.vmap(_cs2deval, in_axes=(None, None, None, 0, 0))(
      Tx, Ty, C, xp.flatten(), yp.flatten()
  )
