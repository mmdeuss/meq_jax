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

"""rtcics jax implementation.

This overrides the pure Matlab implementation in
  third_party/meq/rtcics.m
"""

import jax
import jax.numpy as jnp
import jaxtyping as jt
from meq_jax._src import types
from meq_jax._src.cs2devalmex import cs2devalmex  # pylint: disable=g-importing-member


cs2deval = jax.jit(cs2devalmex)


# pylint: disable=invalid-name


def rtcics(
    a: jt.Float[jt.Array, 'np no'],
    Fx: jt.Float[jt.Array, 'nr nz'],
    F: jt.Float[jt.Array, 'np'],
    Opy: jt.Float[jt.Array, 'nr-2 nz-2'],
    ro: jt.Float[jt.Array, 'np'],
    zo: jt.Float[jt.Array, 'np'],
    Opo: jt.Float[jt.Array, 'np'],
    cr: jt.Float[jt.Array, 'no'],
    cz: jt.Float[jt.Array, 'no'],
    L: types.StaticData,
):
  """JAX implementation of rtcics."""
  assert L.P is not None
  assert L.G is not None

  c = Fx @ L.Mz
  c = c.T @ L.Mr
  tol = L.P.tolq * L.drx

  # Get dimensions
  np = F.shape[0]
  no = cr.shape[0]
  nr = L.nrx
  nz = L.nzx
  niter = L.P.iterq
  Opy_ravel = jnp.asarray(Opy.ravel())

  # Broadcast arrays
  sith = jnp.broadcast_to(cz, (np, no))
  coth = jnp.broadcast_to(cr, (np, no))
  s2th = jnp.broadcast_to(cz**2, (np, no))
  c2th = jnp.broadcast_to(cr**2, (np, no))
  scth = jnp.broadcast_to((cz * cr * 2), (np, no))
  psth = jnp.broadcast_to(F[..., None], (np, no))

  ro = jnp.broadcast_to(ro, (np, no))
  zo = jnp.broadcast_to(zo, (np, no))

  # Value at origin points
  rk = L.taur
  zk = L.tauz
  Fo, *_ = cs2deval(rk, zk, c, ro.ravel(), zo.ravel(), 0.0)

  # Gauss-Newton iteration using jax.lax.while_loop for JIT-compatibility
  def loop_body(_, val):
    a_prev, converged = val

    # Define the update step for one iteration
    def update_step(a_operand):
      # Calculate r and z coordinates
      r = (ro + a_operand * coth).ravel()
      z = (zo + a_operand * sith).ravel()

      # Evaluate spline and its derivatives
      f, gr, gz, hrr, hrz, hzz, *_ = cs2deval(rk, zk, c, r, z, 0.0)

      p = f
      dp = gz * sith.ravel() + gr * coth.ravel()
      d2p = hrr * c2th.ravel() + hrz * scth.ravel() + hzz * s2th.ravel()

      # Calculate the update step 'da'
      # Add a small epsilon to avoid division by zero
      denom = d2p + jnp.finfo(d2p.dtype).eps
      term = dp**2 + 2 * d2p * (psth.ravel() - p)
      da = (jnp.sqrt(jnp.maximum(term, 0)) * jnp.sign(dp) - dp) / denom

      # Proximity to domain boundary
      assert L.G is not None
      assert L.G.rx is not None
      assert L.G.zx is not None
      jr = jnp.floor(jnp.maximum((r - L.G.rx[0]) * L.idrx, 0))
      jz = jnp.floor(jnp.maximum((z - L.G.zx[0]) * L.idzx, 0))
      jr = jnp.clip(jr, 0, nr - 2).astype(jnp.int32)
      jz = jnp.clip(jz, 0, nz - 2).astype(jnp.int32)

      # Proximity to domain boundary
      jr = jr.ravel()
      jz = jz.ravel()
      kq = (nz - 2) * (jr - 1) + jz - 1.0  # Substract one compared to matlab
      kq = kq.astype(jnp.int32)

      # Initialize four arrays to hold the domain of each neighbor
      Op0 = jnp.zeros(jr.shape, dtype=jnp.int8)  # Lower-left
      Op1 = jnp.zeros(jr.shape, dtype=jnp.int8)  # Lower-right
      Op2 = jnp.zeros(jr.shape, dtype=jnp.int8)  # Upper-left
      Op3 = jnp.zeros(jr.shape, dtype=jnp.int8)  # Upper-right

      # Use boolean masks to safely look up values from Opy for each neighbor
      # Lower-left
      mask = (jr > 0) & (jz > 0)
      safe_indices = jnp.clip(kq, 0, Opy_ravel.shape[0] - 1)
      all_new_vals = Opy_ravel[safe_indices]
      Op0 = jnp.where(mask, all_new_vals, Op0)

      # Lower-right
      mask = (jr < nr - 2) & (jz > 0)
      safe_indices = jnp.clip(kq + nz - 2, 0, Opy_ravel.shape[0] - 1)
      all_new_vals = Opy_ravel[safe_indices]
      Op1 = jnp.where(mask, all_new_vals, Op1)

      # Upper-left
      mask = (jr > 0) & (jz < nz - 2)
      safe_indices = jnp.clip(kq + 1, 0, Opy_ravel.shape[0] - 1)
      all_new_vals = Opy_ravel[safe_indices]
      Op2 = jnp.where(mask, all_new_vals, Op2)

      # Upper-right
      mask = (jr < nr - 2) & (jz < nz - 2)
      safe_indices = jnp.clip(kq + nz - 1, 0, Opy_ravel.shape[0] - 1)
      all_new_vals = Opy_ravel[safe_indices]
      Op3 = jnp.where(mask, all_new_vals, Op3)

      # Stack neighbor domains and compare with the origin domain
      zz = (
          jnp.stack([Op0, Op1, Op2, Op3], axis=-1) - Opo.ravel()[:, jnp.newaxis]
      )

      # Classify points based on neighbor domains
      is_different_any = jnp.any(zz != 0, axis=-1)
      is_different_all = jnp.all(zz != 0, axis=-1)

      ptype = jnp.zeros_like(da, dtype=jnp.int32)
      # Adding Opo disables domain checks for gaps
      ptype = jnp.where(is_different_any & Opo, 1, ptype)
      ptype = jnp.where(is_different_all & Opo, 2, ptype)

      # Condition 1: Wrong domain
      mask_ptype2 = ptype == 2
      assert L.drx is not None
      da = jnp.where(mask_ptype2, jnp.asarray(-L.drx, dtype=da.dtype), da)

      # Condition 2: Derivative reverses close to boundary
      grad_check_mask = dp * (Fo - psth.ravel()) > 0
      mask_ptype1 = grad_check_mask & (ptype == 1)
      da = jnp.where(mask_ptype1, jnp.asarray(-L.drx, dtype=da.dtype), da)

      # Condition 3: Escape regions of incorrect gradient
      mask_ptype0 = grad_check_mask & (ptype == 0)
      update_val_ptype0 = jnp.sign((p - psth.ravel()) * dp) * L.drx
      da = jnp.where(
          mask_ptype0, jnp.asarray(update_val_ptype0, dtype=da.dtype), da
      )
      da = da.reshape(a_operand.shape)

      # Update 'a' and check for convergence
      a_new = jnp.maximum(a_operand + da, 0)
      is_converged = jnp.max(jnp.abs(da)) < tol

      return a_new, is_converged

    # If already converged, just pass the values through.
    return jax.lax.cond(
        converged,
        lambda op: op,
        lambda op: update_step(op[0]),
        (a_prev, converged),
    )

  a_final, s_final = jax.lax.fori_loop(0, niter, loop_body, (a, False))

  return a_final, s_final
