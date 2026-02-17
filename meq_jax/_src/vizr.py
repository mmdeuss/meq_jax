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

"""vizr majic implementation."""

import jax.numpy as jnp
import jaxtyping as jt

_PI2 = 6.283185307179586
_PI_MU0_16_INV = 15831.43494411528  # 1/(16*pi*mu0)


# pylint: disable=invalid-name
def vizrmex(
    Fx: jt.Float[jt.Array, 'nr nz'],
    Opy: jt.Integer[jt.Array, 'nr-2 nz-2'],
    ry: jt.Float[jt.Array, 'nr-2'],
    iry: jt.Float[jt.Array, 'nr-2'],
    rBt: jt.Float[jt.Array, ''],
    drx: jt.Float[jt.Array, ''],
    dzx: jt.Float[jt.Array, ''],
) -> tuple[
    jt.Float[jt.Array, ''],
    jt.Float[jt.Array, ''],
    jt.Float[jt.Array, ''],
]:
  """Computes volume integrals on plasma grid.

  See meq/vizrmex.m for more details.
  """
  drdz = drx / dzx
  dzdr = dzx / drx
  ds = drx * dzx

  dFx2_r = jnp.square(Fx[1:-1, 2:] - Fx[1:-1, :-2])
  dFx2_z = jnp.square(Fx[2:, 1:-1] - Fx[:-2, 1:-1])

  # How many grid points on each Z contain plasma.
  zeros = jnp.zeros_like(Opy)
  has_plasma = (Opy > zeros).astype(jnp.int8)
  ni = jnp.sum(has_plasma, axis=1)
  yi = jnp.sum(ni * iry)
  yv = jnp.sum(ni * ry)

  inner = dFx2_r * drdz + dFx2_z * dzdr
  masked = inner * has_plasma
  masked = jnp.sum(masked, axis=1) * iry
  yp = jnp.sum(masked)

  Wp = _PI_MU0_16_INV * yp
  Ft0 = rBt * ds * yi
  Vp = _PI2 * ds * yv  # Volume = 2pi\int RdRdZ = 2*pi*sum(ni*Ri)*dr*dz
  return Wp, Ft0, Vp
