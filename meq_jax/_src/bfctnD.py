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

"""Mimics the logic in bfgenD.m to assemble single-domain basis function sets, defined in bfct.py, into one multi-domain basis function set."""

import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import bfct
from meq_jax._src import types


# pylint: disable=invalid-name


def _process(F1, F0, nbf, iDbf_list):
  """Processing similar to Matlab bfgenD.m."""
  F0bf = jnp.zeros((nbf,))
  F1bf = jnp.zeros((nbf,))

  Finc = jnp.all(F1 >= F0)
  for ibf in range(nbf):
    iD = iDbf_list[ibf]
    assert isinstance(iD, tuple)

    subset_F0 = F0[jnp.array(iD) - 1]
    subset_F1 = F1[jnp.array(iD) - 1]

    im0 = jnp.where(Finc, jnp.argmin(subset_F0), jnp.argmax(subset_F0))
    im1 = jnp.where(Finc, jnp.argmax(subset_F1), jnp.argmin(subset_F1))

    F0bf = F0bf.at[ibf].set(subset_F0[im0])
    F1bf = F1bf.at[ibf].set(subset_F1[im1])

  return F0bf, F1bf


def bfct1_nD(
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
  """Multi-domain version of bfct1."""
  F0bf, F1bf = _process(F1, F0, Bfp.nbf, Bfp.iDbf_list)

  Tyg = jnp.zeros((Bfp.ng, Opy.size))
  TpDg = jnp.zeros((Bfp.ng, Bfp.nD))
  ITpDg = jnp.zeros((Bfp.ng, Bfp.nD))
  ig_start = 0

  # NOTE: we cannot vamp since nP and nT are static arguments of bfct.bfct1
  for ibf in range(Bfp.nbf):
    # No bf coefficients for this set
    num_gi = Bfp.ngi_list[ibf]
    if num_gi == 0:
      continue

    # Skip inactive basis functions
    iD = jnp.array(Bfp.iDbf_list[ibf])
    is_active = jnp.all(jnp.abs(F1[iD - 1] - F0[iD - 1]) > 1e-13)

    # Define the row range for this basis function set
    ig_idx = jnp.arange(ig_start, min(ig_start + num_gi, Bfp.ng))
    ig_start += num_gi

    nP, nT = Bfp.bfp_list[ibf]
    this_Bfp = types.BfpData(nP=int(nP), nT=int(nT))

    for this_iD in Bfp.iDbf_list[ibf]:
      this_Tygi, this_TpDg, _, this_ITpDg = bfct.bfct1(
          Fx=Fx,
          F0=F0bf[ibf],
          F1=F1bf[ibf],
          Opy=(Opy == this_iD).astype(int),
          ry=ry,
          iry=iry,
          Bfp=this_Bfp,
      )
      Tyg = Tyg.at[ig_idx].add(is_active * this_Tygi)
      TpDg = TpDg.at[ig_idx, iD - 1].set(is_active * this_TpDg)
      ITpDg = ITpDg.at[ig_idx, iD - 1].set(is_active * this_ITpDg)

  return Tyg, TpDg, None, ITpDg


def bfct2_nD(
    FN: jt.Float[jt.Array, 'nr nz'],
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
    Bfp: types.BfpData,
) -> tuple[jt.Float[jt.Array, 'nQ nP+nT'], jt.Float[jt.Array, 'nQ nP+nT']]:
  """Multi-domain version of bfct2."""
  F0bf, _ = _process(F1, F0, Bfp.nbf, Bfp.iDbf_list)

  gQDg = jnp.zeros((Bfp.ng, Bfp.nD, FN.size))
  IgQDg = jnp.zeros((Bfp.ng, Bfp.nD, FN.size))
  ig_start = 0

  for ibf in range(Bfp.nbf):
    num_gi = Bfp.ngi_list[ibf]
    if num_gi == 0:
      continue

    # Skip inactive basis functions
    iD = jnp.array(Bfp.iDbf_list[ibf])
    is_active = jnp.all(jnp.abs(F1[iD - 1] - F0[iD - 1]) > 1e-13)

    # Define the row range for this basis function set
    ig_idx = jnp.arange(ig_start, min(ig_start + num_gi, Bfp.ng))
    ig_start += num_gi

    nP, nT = Bfp.bfp_list[ibf]
    this_Bfp = types.BfpData(nP=int(nP), nT=int(nT))

    idFbf = 1.0 / (F1[ibf] - F0[ibf] + 1e-10)

    for this_iD in Bfp.iDbf_list[ibf]:
      # Specific FN
      FN = (
          F0[iD - 1] + FN * (F1[this_iD - 1] - F0[this_iD - 1]) - F0bf[ibf]
      ) * idFbf

      this_gQDg, this_IgQDg = bfct.bfct2(FN, this_Bfp)

      gQDg = gQDg.at[ig_idx, this_iD - 1].set(is_active * this_gQDg)
      IgQDg = IgQDg.at[ig_idx, this_iD - 1].set(is_active * this_IgQDg)
  return gQDg, IgQDg


def bfct3_nD(
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
  """Multi-domain version of bfct3."""
  F0bf, F1bf = _process(F1, F0, Bfp.nbf, Bfp.iDbf_list)

  aPpg = jnp.zeros((Bfp.ng,))
  aTTpg = jnp.zeros((Bfp.ng,))
  aPg = jnp.zeros((Bfp.ng,))
  ahqTg = jnp.zeros((Bfp.ng,))
  ig_start = 0

  for ibf in range(Bfp.nbf):
    num_gi = Bfp.ngi_list[ibf]
    if num_gi == 0:
      continue

    # Skip inactive basis functions
    iD = jnp.array(Bfp.iDbf_list[ibf])
    is_active = jnp.all(jnp.abs(F1[iD - 1] - F0[iD - 1]) > 1e-13)

    # Define the row range for this basis function set
    ig_idx = jnp.arange(ig_start, min(ig_start + num_gi, Bfp.ng))
    ig_start += num_gi

    nP, nT = Bfp.bfp_list[ibf]
    this_Bfp = types.BfpData(nP=int(nP), nT=int(nT))

    this_aPpg, this_aTTpg, this_aPg, this_ahqTg = bfct.bfct3(
        ag=ag[ig_idx],
        F0=F0bf[ibf],
        F1=F1bf[ibf],
        ids=ids,
        Bfp=this_Bfp,
    )
    aPpg = aPpg.at[ig_idx].set(is_active * this_aPpg)
    aTTpg = aTTpg.at[ig_idx].set(is_active * this_aTTpg)
    aPg = aPg.at[ig_idx].set(is_active * this_aPg)
    ahqTg = ahqTg.at[ig_idx].set(is_active * this_ahqTg)

  return aPpg, aTTpg, aPg, ahqTg


def bfct5_nD(
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
    Bfp: types.BfpData,
) -> tuple[
    jt.Float[jt.Array, 'nP+nT'],
    jt.Float[jt.Array, 'nP+nT'],
]:
  """Multi-domain version of bfct5."""
  F0bf, F1bf = _process(F1, F0, Bfp.nbf, Bfp.iDbf_list)

  nD = len(Bfp.bfp_list)
  g0g = jnp.zeros((Bfp.ng, nD))
  Ig0g = jnp.zeros((Bfp.ng, nD))
  ig_start = 0

  for ibf in range(Bfp.nbf):
    num_gi = Bfp.ngi_list[ibf]
    if num_gi == 0:
      continue

    # Skip inactive basis functions
    iD = jnp.array(Bfp.iDbf_list[ibf])
    is_active = jnp.all(jnp.abs(F1[iD - 1] - F0[iD - 1]) > 1e-13)

    # Define the row range for this basis function set
    ig_idx = jnp.arange(ig_start, min(ig_start + num_gi, Bfp.ng))
    ig_start += num_gi

    nP, nT = Bfp.bfp_list[ibf]
    this_Bfp = types.BfpData(nP=int(nP), nT=int(nT))

    # NOTE: the Matlab code seems to call bfct91?
    this_gA, this_IgA = bfct.bfct5(F0=F0bf[ibf], F1=F1bf[ibf], Bfp=this_Bfp)

    iD = jnp.array(Bfp.iDbf_list[ibf])
    g0g = g0g.at[ig_idx, iD - 1].set(is_active * this_gA)
    Ig0g = Ig0g.at[ig_idx, iD - 1].set(is_active * this_IgA)

  return g0g, Ig0g


def bfct11_nD(
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
  """Multi-domain version of bfct11."""
  F0bf, F1bf = _process(F1, F0, Bfp.nbf, Bfp.iDbf_list)

  ny = Opy.size
  dgy = jnp.zeros((Bfp.ng, ny))
  dg0 = jnp.zeros((Bfp.nD, Bfp.ng, ny))
  dg1 = jnp.zeros((Bfp.nD, Bfp.ng, ny))
  dIg0 = jnp.zeros((Bfp.nD, Bfp.ng, ny))
  dIg1 = jnp.zeros((Bfp.nD, Bfp.ng, ny))

  ig_start = 0
  for ibf in range(Bfp.nbf):
    num_gi = Bfp.ngi_list[ibf]
    if num_gi == 0:
      continue

    # Skip inactive basis functions
    iD = jnp.array(Bfp.iDbf_list[ibf])
    is_active = jnp.all(jnp.abs(F1[iD - 1] - F0[iD - 1]) > 1e-13)

    # Define the row range for this basis function set
    ig_idx = jnp.arange(ig_start, min(ig_start + num_gi, Bfp.ng))
    ig_start += num_gi

    nP, nT = Bfp.bfp_list[ibf]
    this_Bfp = types.BfpData(nP=int(nP), nT=int(nT))

    arr = jnp.array([Opy == iD for iD in Bfp.iDbf_list[ibf]])
    this_Opy = jnp.sum(arr, axis=0, keepdims=False)

    this_dgy, this_dg0, this_dg1, this_dIg0, this_dIg1 = bfct.bfct11(
        Fx=Fx,
        F0=F0bf[ibf],
        F1=F1bf[ibf],
        Opy=this_Opy,
        ry=ry,
        iry=iry,
        Bfp=this_Bfp,
    )

    iD = jnp.array(Bfp.iDbf_list[ibf])
    dgy = dgy.at[ig_idx].set(is_active * this_dgy)
    dg0 = dg0.at[iD - 1, ig_idx].set(is_active * this_dg0)
    dg1 = dg1.at[iD - 1, ig_idx].set(is_active * this_dg1)
    dIg0 = dIg0.at[iD - 1, ig_idx].set(is_active * this_dIg0)
    dIg1 = dIg1.at[iD - 1, ig_idx].set(is_active * this_dIg1)

  return dgy, dg0, dg1, dIg0, dIg1


def bfct91_nD(
    F: jt.Float[jt.Array, 'nr nz'],
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
    Bfp: types.BfpData,
) -> tuple[
    jt.Float[jt.Array, 'nP+nT nr*nz'], jt.Float[jt.Array, 'nP+nT nr*nz']
]:
  """Multi-domain version of bfct91."""
  F0bf, F1bf = _process(F1, F0, Bfp.nbf, Bfp.iDbf_list)

  g = jnp.zeros((Bfp.ng, F.size))
  Ig = jnp.zeros((Bfp.ng, F.size))
  ig_start = 0

  for ibf in range(Bfp.nbf):
    num_gi = Bfp.ngi_list[ibf]
    if num_gi == 0:
      continue

    # Skip inactive basis functions
    iD = jnp.array(Bfp.iDbf_list[ibf])
    is_active = jnp.all(jnp.abs(F1[iD - 1] - F0[iD - 1]) > 1e-13)

    # Define the row range for this basis function set
    ig_idx = jnp.arange(ig_start, min(ig_start + num_gi, Bfp.ng))
    ig_start += num_gi

    nP, nT = Bfp.bfp_list[ibf]
    this_Bfp = types.BfpData(nP=int(nP), nT=int(nT))

    this_g, this_Ig = bfct.bfct91(
        F=F,
        F0=F0bf[ibf],
        F1=F1bf[ibf],
        Bfp=this_Bfp,
    )
    g = g.at[ig_idx].set(is_active * this_g)
    Ig = Ig.at[ig_idx].set(is_active * this_Ig)

  return g, Ig
