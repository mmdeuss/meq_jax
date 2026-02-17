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

"""JAX version of bfdoublet.m, which is specific to the doublet with mantle case."""

import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import bfab
from meq_jax._src import bfct
from meq_jax._src import bfctnD
from meq_jax._src import types


# pylint: disable=invalid-name


def bfct1_doublet_mantle(
    Fx: jt.Float[jt.Array, 'nry+2 nzy+2'],
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
    Opy: jt.Int[jt.Array, 'nry nzy'],
    ry: jt.Float[jt.Array, 'nry'],
    iry: jt.Float[jt.Array, 'nry'],
    Bfp: types.BfpData,
) -> tuple[
    jt.Float[jt.Array, 'nP+nT nry*nzy'],
    jt.Float[jt.Array, 'nP+nT'],
    None,
    jt.Float[jt.Array, 'nP+nT'],
]:
  """Doublet with mantle version of bfct1."""
  Tyg, TpDg, _, ITpDg = bfctnD.bfct1_nD(
      Fx=Fx,
      F0=F0,
      F1=F1,
      Opy=Opy,
      ry=ry,
      iry=iry,
      Bfp=Bfp,
  )

  Fy = Fx[1:-1, 1:-1]
  nP, nT = Bfp.bfp_list[2]
  this_Bfp = types.BfpData(nP=int(nP), nT=int(nT))
  GA3, IGA3 = bfct.bfct5(F0[2], F1[2], this_Bfp)

  mask1 = (Opy == 1).astype(jnp.float32)
  fac1 = (mask1 * ry[:, None]).reshape(1, -1) * Bfp.fPg3[:, None] + (
      mask1 * iry[:, None]
  ).reshape(1, -1) * Bfp.fTg3[:, None]

  mask2 = (Opy == 2).astype(jnp.float32)
  fac2 = (mask2 * ry[:, None]).reshape(1, -1) * Bfp.fPg3[:, None] + (
      mask2 * iry[:, None]
  ).reshape(1, -1) * Bfp.fTg3[:, None]

  # Update Tyg
  Tyg = Tyg.at[Bfp.igm - 1, :].add((fac1 + fac2) * GA3[:, None])

  # Update TpDg
  TpDg = TpDg.at[Bfp.igm - 1, 0].add(jnp.sum(fac1 * GA3[:, None], axis=1))
  TpDg = TpDg.at[Bfp.igm - 1, 1].add(jnp.sum(fac2 * GA3[:, None], axis=1))

  # Update ITpDg
  delta = IGA3[:, None] + (Fy.flatten()[None] - F0[2]) * GA3[:, None]

  ITpDg = ITpDg.at[Bfp.igm - 1, 0].add(jnp.sum(fac1 * delta, axis=1))
  ITpDg = ITpDg.at[Bfp.igm - 1, 1].add(jnp.sum(fac2 * delta, axis=1))

  return Tyg, TpDg, None, ITpDg


def bfct2_doublet_mantle(
    FN: jt.Float[jt.Array, 'nr nz'],
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
    Bfp: types.BfpData,
) -> tuple[jt.Float[jt.Array, 'nQ nP+nT'], jt.Float[jt.Array, 'nQ nP+nT']]:
  """Doublet with mantle version of bfct2."""
  gQg, IgQg = bfctnD.bfct2_nD(FN, F0, F1, Bfp)
  nP, nT = Bfp.bfp_list[2]
  gA3, IgA3 = bfab.fN(jnp.array(0), nP=int(nP), nT=int(nT))
  gQg = gQg.at[Bfp.igm - 1, :2].add(gA3[:, None])
  IgQg = IgQg.at[Bfp.igm - 1, :2].add(IgA3[:, None])

  # We have to scale the added linear ramp in the first two domains by
  IgQg = IgQg.at[Bfp.igm - 1, 0].add(
      (FN[None] - 1) * gA3[:, None] * (F0[2] - F0[0]) / (F1[2] - F0[2])
  )
  IgQg = IgQg.at[Bfp.igm - 1, 1].add(
      (FN[None] - 1) * gA3[:, None] * (F0[2] - F0[1]) / (F1[2] - F0[2])
  )
  return gQg, IgQg


def bfct3_doublet_mantle(
    ag: jt.Float[jt.Array, 'nP+nT'],
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
    ids: jt.Float[jt.Array, ''],
    Bfp: types.BfpData,
) -> tuple[
    jt.Float[jt.Array, 'nP+nT nry*nzy'],
    jt.Float[jt.Array, 'nP+nT'],
    jt.Float[jt.Array, 'nP+nT nry*nzy'],
    jt.Float[jt.Array, 'nP+nT'],
]:
  """Doublet with mantle version of bfct2."""
  return bfctnD.bfct3_nD(ag, F0, F1, ids, Bfp)


def bfct5_doublet_mantle(
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
    Bfp: types.BfpData,
) -> tuple[
    jt.Float[jt.Array, 'nP+nT'],
    jt.Float[jt.Array, 'nP+nT'],
]:
  """Doublet with mantle version of bfct5."""
  GAg, IGAg = bfctnD.bfct5_nD(
      F0=F0,
      F1=F1,
      Bfp=Bfp,
  )
  # Offset between Jax and Matlab indexing
  GAg = GAg.at[Bfp.igm - 1, :2].add(GAg[Bfp.igm - 1, 2])
  IGAg = IGAg.at[Bfp.igm - 1, 0].add(
      IGAg[Bfp.igm - 1, 2] + (F0[2] - F0[0]) * GAg[Bfp.igm - 1, 2]
  )
  IGAg = IGAg.at[Bfp.igm - 1, 1].add(
      IGAg[Bfp.igm - 1, 2] + (F0[2] - F0[1]) * GAg[Bfp.igm - 1, 2]
  )
  return GAg, IGAg


def bfct11_doublet_mantle(
    Fx: jt.Float[jt.Array, 'nry+2 nzy+2'],
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
    Opy: jt.Int[jt.Array, 'nry nzy'],
    ry: jt.Float[jt.Array, 'nry'],
    iry: jt.Float[jt.Array, 'nry'],
    Bfp: types.BfpData,
) -> tuple[
    jt.Float[jt.Array, 'nP+nT nry*nzy'],  # dTygdFy
    jt.Float[jt.Array, 'nP+nT nry*nzy'],  # dTygdF0
    jt.Float[jt.Array, 'nP+nT nry*nzy'],  # dTygdF1
    jt.Float[jt.Array, 'nP+nT nry*nzy'],  # dITygdF0
    jt.Float[jt.Array, 'nP+nT nry*nzy'],  # dITygdF1
]:
  """Doublet with mantle version of bfct11."""
  dTygdFy, dTygdF0, dTygdF1, dITygdF0, dITygdF1 = bfctnD.bfct11_nD(
      Fx=Fx,
      F0=F0,
      F1=F1,
      Opy=Opy,
      ry=ry,
      iry=iry,
      Bfp=Bfp,
  )

  Fy = Fx[1:-1, 1:-1]
  nP, nT = Bfp.bfp_list[2]
  this_Bfp = types.BfpData(nP=int(nP), nT=int(nT))
  GA3, _ = bfct.bfct5(F0[2], F1[2], this_Bfp)

  dGA3dF0, dGA3dF1, dIGA3dF0, dIGA3dF1 = bfct.bfct15(
      F0=F0[2], F1=F1[2], Bfp=this_Bfp
  )

  mask1 = (Opy == 1).astype(jnp.float32)
  fac1 = (mask1 * ry[:, None]).reshape(1, -1) * Bfp.fPg3[:, None] + (
      mask1 * iry[:, None]
  ).reshape(1, -1) * Bfp.fTg3[:, None]

  mask2 = (Opy == 2).astype(jnp.float32)
  fac2 = (mask2 * ry[:, None]).reshape(1, -1) * Bfp.fPg3[:, None] + (
      mask2 * iry[:, None]
  ).reshape(1, -1) * Bfp.fTg3[:, None]

  # Update
  dTygdF0 = dTygdF0.at[2, Bfp.igm - 1, :].add((fac1 + fac2) * dGA3dF0[:, None])
  dTygdF1 = dTygdF1.at[2, Bfp.igm - 1, :].add((fac1 + fac2) * dGA3dF1[:, None])

  add_dITygdF0 = (
      dIGA3dF0[:, None]
      + (Fy.flatten() - F0[2]) * dGA3dF0[:, None]
      - GA3[:, None]
  )
  dITygdF0 = dITygdF0.at[2, Bfp.igm - 1, :].add((fac1 + fac2) * add_dITygdF0)

  add_dITygdF1 = dIGA3dF1[:, None] + (Fy.flatten() - F0[2]) * dGA3dF1[:, None]
  dITygdF1 = dITygdF1.at[2, Bfp.igm - 1, :].add((fac1 + fac2) * add_dITygdF1)

  return dTygdFy, dTygdF0, dTygdF1, dITygdF0, dITygdF1


def bfct91_doublet_mantle(
    F: jt.Float[jt.Array, 'nr nz'],
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
    Bfp: types.BfpData,
) -> tuple[
    jt.Float[jt.Array, 'nP+nT nr*nz'], jt.Float[jt.Array, 'nP+nT nr*nz']
]:
  """Doublet with mantle version of bfct91."""
  G, IG = bfctnD.bfct91_nD(F=F, F0=F0, F1=F1, Bfp=Bfp)
  nP, nT = Bfp.bfp_list[2]
  this_Bfp = types.BfpData(nP=int(nP), nT=int(nT))
  GA3, IGA3 = bfct.bfct5(F0[2], F1[2], this_Bfp)

  sIp = ((F1[2] - F0[2]) >= 0).astype(jnp.float32)
  sIp = 2 * sIp - 1
  mask = sIp * F.flatten() < sIp * F0[2]

  G = G.at[Bfp.igm - 1, :].set(
      G[Bfp.igm - 1, :] * ~mask[None] + GA3[:, None] * mask[None]
  )
  IG = IG.at[Bfp.igm - 1, :].set(
      IG[Bfp.igm - 1, :] * ~mask[None]
      + (IGA3[:, None] + (F.flatten() - F0[2]) * GA3[:, None]) * mask[None]
  )
  return G, IG
