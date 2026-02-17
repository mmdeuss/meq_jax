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

"""JAX implementation of the meqintQ.m file."""

import einops
import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import bfct_router
from meq_jax._src import cizr
from meq_jax._src import types


# pylint: disable=invalid-name
def meqintQ(
    L: types.StaticData,
    F0: jt.Float[jt.Array, 'L_F0'],
    F1: jt.Float[jt.Array, 'L_F0'],
    rBt: float,
    ag: jt.Float[jt.Array, 'L_nd ag2'],  # Is ag2 always 1?
    Fx: jt.Float[jt.Array, 'b1+2 b2+2'],
    Opy: jt.Int[jt.Array, 'b1 b2'],
    L_ny: int,
    Bfp: types.BfpData,
) -> jt.Float[jt.Array, '7 1 nQ']:
  """Returns meqintQ(L,F0,F1,rBt,ag,Fx,Opy)."""
  # Note: Removed ag==0 special case

  Opy = jnp.where(Opy > L.nD, 0, Opy)

  # Normalized flux grid
  FQ = L.pQ**2

  assert L.rry is not None
  rry = L.rry.flatten()
  irry = 1.0 / rry
  masky = Opy.flatten() > 0
  rry = rry * masky
  irry = irry * masky

  G, IG = bfct_router.bfct91(F=Fx[1:-1, 1:-1], F0=F0, F1=F1, Bfp=Bfp)

  assert L.fPg is not None
  assert L.fTg is not None
  assert L.TDg is not None
  fac = (rry[:, None] * L.fPg[None]) + (irry[:, None] * L.fTg[None])
  # Deselect basis functions that are inactive in certain domains
  fac = fac * L.TDg[:, jnp.maximum(0, Opy.flatten() - 1)].T
  Tyg = fac * G.T
  ITyg = fac * IG.T

  RBp2y = ((Fx[2:, 1:-1] - Fx[:-2, 1:-1]) / (4 * jnp.pi * L.drx)) ** 2 + (
      (Fx[1:-1, 2:] - Fx[1:-1, :-2]) / (4 * jnp.pi * L.dzx)
  ) ** 2

  # Equivalent layout to matlab computation
  Vy = jnp.hstack(
      [
          (Tyg @ (L.idsx * ag)),
          (ITyg @ (L.idsx * ag * L.fPg[:, None]) * 1.5),
          (RBp2y.flatten() * irry / 4e-7)[:, None],
          (2 * jnp.pi * rry)[:, None],
          (ITyg @ (ag * L.fTg[:, None] * 2e-7 / rBt * L.idsx)),
          (rBt * irry)[:, None],
          jnp.tile(L.idsx, (L_ny,))[:, None],
      ],
  )

  # Deal with transpose between matlab and octave
  jax_vy = einops.rearrange(
      Vy,
      '(b2 b1) nV -> b1 b2 nV',
      b1=Opy.shape[1],
      b2=Opy.shape[0],
      nV=Vy.shape[-1],
  )

  # TODO(adedieu): cizr does not assume Opy is transposed from Matlab
  IVQD = cizr.cizr(
      Fx.T,
      Opy.T,
      Vy=jax_vy,
      dsx=L.dsx,
      FNQ=FQ,
      F0=F0,
      F1=F1,
  ).T

  IpQ = IVQD[:, :, 0]
  WkQ = IVQD[:, :, 1]
  WpQ = IVQD[:, :, 2]
  VpQ = IVQD[:, :, 3]
  FtQ = IVQD[:, :, 4]
  Ft0Q = IVQD[:, :, 5]
  OpQ = IVQD[:, :, 6]

  return IpQ, WkQ, WpQ, VpQ, FtQ, Ft0Q, OpQ
