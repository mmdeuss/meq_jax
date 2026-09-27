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

"""rtcimex majic implementation.

This overrides the implementation in
  third_party/meq/mexc/rtcimex.c
whose pure Matlab implementation is in
  third_party/meq/mexm/rtcimex.m

We separetely handle the cases where the last 4 inputs are present or not.
"""

import jax.numpy as jnp
import jaxtyping as jt


# pylint: disable=invalid-name
def rtcimex(
    a: jt.Float[jt.Array, "npq noq"],
    er0: jt.Float[jt.Array, "n"],
    ez0: jt.Float[jt.Array, "n"],
    Fx: jt.Float[jt.Array, "n"],
    F: jt.Float[jt.Array, "n"],
    c: jt.Float[jt.Array, "n"],
    s: jt.Float[jt.Array, "n"],
    Opy: jt.Float[jt.Array, "n"],
    Fo: jt.Float[jt.Array, "n"],
    Opo: jt.Float[jt.Array, "n"],
    dap: jt.Float[jt.Array, ""],
):
  """JAX implementation of rtcimex."""
  nr, nz = Fx.shape
  no = c.shape[0]

  er = er0 + a * c
  ez = ez0 + a * s

  jr = jnp.floor(jnp.maximum(er, 0)).astype(jnp.int32)
  jr = jnp.minimum(jr, nr - 2)

  jz = jnp.floor(jnp.maximum(ez, 0)).astype(jnp.int32)
  jz = jnp.minimum(jz, nz - 2)

  frq = er - jr
  fzq = ez - jz

  # Bilinear interpolation of Fx
  # Get values at the four corners of the grid cell
  f0 = Fx[jr, jz]
  f1 = Fx[jr + 1, jz] - f0
  f2 = Fx[jr, jz + 1] - f0
  f3 = Fx[jr + 1, jz + 1] - f0 - f1 - f2

  f4 = fzq * f3 + f1
  fw = f0 + frq * f4 + fzq * f2  # Interpolated value
  df = (frq * f3 + f2) * s + f4 * c  # Derivative of interpolated value

  fq = jnp.broadcast_to(F[:, None], (F.shape[0], no))

  # Proximity to domain boundary
  kq = (nz - 2) * (jr - 1) + jz - 1.0  # Substract one compared to matlab
  kq = kq.astype(jnp.int32)
  Opy_ravel = Opy.ravel()

  # Initialize four arrays to hold the domain of each neighbor
  Op0 = jnp.zeros(jr.shape, dtype=jnp.int8)  # Lower-left
  Op1 = jnp.zeros(jr.shape, dtype=jnp.int8)  # Lower-right
  Op2 = jnp.zeros(jr.shape, dtype=jnp.int8)  # Upper-left
  Op3 = jnp.zeros(jr.shape, dtype=jnp.int8)  # Upper-right

  # Use boolean masks to safely look up values from Opy for each neighbor
  # Lower-left
  mask = jnp.logical_and(jr > 0, jz > 0)
  safe_indices = jnp.clip(kq, 0, Opy_ravel.shape[0] - 1)
  all_new_vals = Opy_ravel[safe_indices]
  Op0 = jnp.where(mask, all_new_vals, Op0)

  # Lower-right
  mask = jnp.logical_and(jr < nr - 2, jz > 0)
  safe_indices = jnp.clip(kq + nz - 2, 0, Opy_ravel.shape[0] - 1)
  all_new_vals = Opy_ravel[safe_indices]
  Op1 = jnp.where(mask, all_new_vals, Op1)

  # Upper-left
  mask = jnp.logical_and(jr > 0, jz < nz - 2)
  safe_indices = jnp.clip(kq + 1, 0, Opy_ravel.shape[0] - 1)
  all_new_vals = Opy_ravel[safe_indices]
  Op2 = jnp.where(mask, all_new_vals, Op2)

  # Upper-right
  mask = jnp.logical_and(jr < nr - 2, jz < nz - 2)
  safe_indices = jnp.clip(kq + nz - 1, 0, Opy_ravel.shape[0] - 1)
  all_new_vals = Opy_ravel[safe_indices]
  Op3 = jnp.where(mask, all_new_vals, Op3)

  # Stack neighbor domains and compare with the origin domain
  zz = jnp.stack([Op0, Op1, Op2, Op3], axis=-1) - Opo[:, jnp.newaxis]

  # Classify points based on neighbor domains
  is_different_any = jnp.any(zz != 0, axis=-1)
  is_different_all = jnp.all(zz != 0, axis=-1)

  ptype = jnp.zeros_like(fq, dtype=jnp.int32)
  # Adding Opo disables domain checks for gaps
  ptype = jnp.where(jnp.logical_and(is_different_any, Opo), 1, ptype)
  ptype = jnp.where(jnp.logical_and(is_different_all, Opo), 2, ptype)

  # Initial step update calculation
  da = jnp.where(df != 0, (fq - fw) / df, 0)

  # Condition 1: Wrong domain (if in plasma domain)
  mask_ptype2 = ptype == 2
  da = jnp.where(mask_ptype2, -dap, da)

  # Condition 2: Derivative reverses close to boundary
  # Compare against zero exactly, as libmeq/rtci.c does. Where a contour
  # touches an origin point, Fo - fq is about one ULP, but its sign still
  # picks the right branch. A tolerance here loses that, and the step walks
  # to the far intersection instead of the near one.
  grad_check_mask = (df * (Fo - fq)) > 0
  mask_ptype1 = jnp.logical_and(grad_check_mask, ptype == 1)
  da = jnp.where(mask_ptype1, -dap, da)

  # Condition 3: Escape regions of incorrect gradient
  mask_ptype0 = jnp.logical_and(grad_check_mask, ptype == 0)
  update_val_ptype0 = jnp.sign((fw - fq) * df) * dap
  da = jnp.where(mask_ptype0, update_val_ptype0, da)

  # Clip to the maximum allowed step size
  da = jnp.clip(da, -dap, dap)

  # Update a and calculate max change
  a = jnp.abs(a + da)
  dan = jnp.max(jnp.abs(da))

  return a, dan


# pylint: enable=invalid-name
