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

"""JAX implementation of the meqIyJac.m file."""

import jax.numpy as jnp
import jaxtyping as jt


# pylint: disable=invalid-name
def meqIyJac(
    lxy: jt.Bool[jt.Array, 'lxy1 lxy2'],
    ag: jt.Float[jt.Array, 'ng'],
    mask: jt.Bool[jt.Array, 'ny'],
    dTygdFy: jt.Float[jt.Array, 'ng ny'],
    dTygdF0: jt.Float[jt.Array, 'nD ng ng'],
    dTygdF1: jt.Float[jt.Array, 'nD ng ng'],
    dF0dFx: jt.Float[jt.Array, 'nx nD'],
    dF1dFx: jt.Float[jt.Array, 'nx nD'],
) -> tuple[
    jt.Float[jt.Array, '1 ny'],
    jt.Float[jt.Array, 'nD ny nD'],
    jt.Float[jt.Array, 'nD ny nD'],
    jt.Float[jt.Array, 'nx ny nx'],
]:
  """Computes analytical gradient of Iy with respect to Fy,F0,F1 and total Fx.

  Port of meqIyJac.m

  Args:
    lxy: L.lxy parameter from matlab.
    ag: ag vector.
    mask: Mask for plasma points.
    dTygdFy: d(T*g)/dFy from bfct derivative.
    dTygdF0: d(T*g)/dF0 from bfct derivative.
    dTygdF1: d(T*g)/dF1 from bfct derivative.
    dF0dFx: dF0/dFx from flux derivative.
    dF1dFx: dF1/dFx from flux derivative.

  Returns:
    dIypdFy, dIypdF0, dIypdF1, dIypdFx.
    If icsint=True, dIypdFx is shape (nyp, nx), else (ny, nx).
  """
  dIypdFy_nomask = jnp.einsum('ij,i->j', dTygdFy, ag)
  dIypdF0_nomask = jnp.einsum('ijk,j->ik', dTygdF0, ag)
  dIypdF1_nomask = jnp.einsum('ijk,j->ik', dTygdF1, ag)

  # A bit of gymnastics to avoid variable-shaped arrays. This is equivalent to
  # jj = lxy.T.flatten().nonzero()[0]
  # jj = jj[mask]
  # lxy_mask = jnp.zeros(lxy.size)
  # lxy_mask[jj] = 1
  lxy_flat = lxy.flatten().astype(jnp.int32)
  lxy_ranks = jnp.cumsum(lxy_flat) - 1
  safe_lxy_ranks = jnp.where(lxy_flat, lxy_ranks, 0)
  mask_broadcasted = mask[safe_lxy_ranks]
  lxy_mask = lxy_flat & mask_broadcasted

  # Masks where the nonzero entries count up instead of being all ones.
  mask_sum = jnp.cumsum(mask) * mask
  lxy_mask_sum = jnp.cumsum(lxy_mask) * lxy_mask

  x_mask, y_mask = jnp.meshgrid(lxy_mask_sum, mask_sum)

  dIypdFx = jnp.where(
      jnp.logical_and(
          jnp.logical_and(x_mask == y_mask, x_mask != 0), y_mask != 0),
      jnp.tile(dIypdFy_nomask[:, None], (1, lxy.size)), 0)

  dIypdFx = dIypdFx.T
  dIypdFx = dIypdFx + dF0dFx @ dIypdF0_nomask + dF1dFx @ dIypdF1_nomask

  return dIypdFy_nomask, dIypdF0_nomask, dIypdF1_nomask, dIypdFx
