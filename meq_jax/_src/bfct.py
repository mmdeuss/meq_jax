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

"""JAX implementation of the C functions `dbfabN` where N is 1 & 5 (for now).

This is a direct port of the bfct.m (matlab) and bfabmex.c, bfctmex.h and bfct.c
(C) code.
"""

import jax
import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import bfab
from meq_jax._src import types

# pylint: disable=invalid-name


def bfct1(
    Fx: jt.Float[jt.Array, 'nry+2 nzy+2'],
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    Opy: jt.Int[jt.Array, 'nry nzy'],
    ry: jt.Float[jt.Array, 'nry'],
    iry: jt.Float[jt.Array, 'nry'],
    Bfp: types.BfpData,
) -> tuple[
    jt.Float[jt.Array, 'nP+nT nry*nzy'],
    jt.Float[jt.Array, 'nP+nT'],
    jt.Float[jt.Array, 'nP+nT nry*nzy'],
    jt.Float[jt.Array, 'nP+nT'],
]:
  """Computes the transfer matrix and integrals over the plasma domain.

  TODO(joeljennings): This defaults to the `bfab.f` function.

  Args:
      Fx: 2D array of flux values on the grid
      F0: Flux at the magnetic axis.
      F1: Flux at the plasma boundary.
      Opy: 2D boolean mask, 1 if in plasma, 0 otherwise.
      ry: 1D array of major radius values for each row.
      iry: 1D array of 1/R values for each row.
      Bfp: Bfp data

  Returns:
      Tyg: Transfer matrix
      Tpg: Integral of g over the domain
      ITyg: Inverse transfer matrix
      ITpg: Integral of Ig over the domain
  """
  if F0.ndim == 1:
    assert F0.shape[0] == 1
    assert F1.shape[0] == 1
    F0 = F0[0]
    F1 = F1[0]

  FBA = F1 - F0
  FxA = Fx[1:-1, 1:-1] - F0
  nP, nT = Bfp.nP, Bfp.nT

  # Vectorize the elementary basis function calculation over the (nry, nzy) grid
  vmap_f = jax.vmap(jax.vmap(lambda fxa: bfab.f(fxa, FBA, nP, nT)))
  g_grid, Ig_grid = vmap_f(FxA)

  # Define which basis functions are P' type and which are TT' type
  fPg, fTg = bfab.fPg(nP, nT)

  # Compute the geometric factor `fac` by broadcasting R and 1/R values
  # across the appropriate basis functions.
  # Note: r = nry, k = nP + nT
  fac = jnp.einsum('r,k->kr', ry, fPg) + jnp.einsum('r,k->kr', iry, fTg)

  # Apply the factor and the plasma mask (Opy).
  # Note: r = nry, z = nzy, k = nP + nT
  g_prime = jnp.einsum('rzk,kr,rz->krz', g_grid, fac, Opy)
  Tyg = g_prime.reshape(nP + nT, -1)  # (nP + nT, nry * nzy)
  Ig_prime = jnp.einsum('rzk,kr,rz->krz', Ig_grid, fac, Opy)
  ITyg = Ig_prime.reshape(nP + nT, -1)  # (nP + nT, nry * nzy)

  return Tyg, Tyg.sum(axis=-1), ITyg, ITyg.sum(axis=-1)


# pylint: disable=invalid-name
def bfct1_jac(
    Opy: jt.Float[jt.Array, 'Opy1 Opy2'],
    dYygdFy: jt.Float[jt.Array, 'nz ng'],
    dYygdF0: jt.Float[jt.Array, 'nz ng'],
    dYygdF1: jt.Float[jt.Array, 'nz ng'],
    dF0dFx: jt.Float[jt.Array, 'nD nx'],
    dF1dFx: jt.Float[jt.Array, 'nD nx'],
    lxy: jt.Bool[jt.Array, 'lxy1 lxy2 '],
    nB: int,
    nD: int,
    ng: int,
    nx: int,
    ny: int,
) -> tuple[
    jt.Float[jt.Array, 'ng nz'],
    jt.Float[jt.Array, 'nz ng'],
    jt.Float[jt.Array, 'nz ng'],
    jt.Float[jt.Array, 'nz nx'],
]:
  """Assembles analytical gradient of YDg, where YDg is any one of TpDg|ITpDg, with respect to Fy,F0,F1 and total Fx.

  Port of meqbfct1Jac.m

  Args:
    Opy: The Opy matrix.
    dYygdFy: The dYygdFy matrix.
    dYygdF0: The dYygdF0 matrix.
    dYygdF1: The dYygdF1 matrix.
    dF0dFx: The dF0dFx matrix.
    dF1dFx: The dF1dFx matrix.
    lxy: The lxy matrix.
    nB: Number of plasma domain boundaries found.
    nD: Number of plasma domains.
    ng:
    nx:
    ny:

  Returns:
    dYDgdFy: The dYDgdFy matrix.
    dYDgdF0: The dYDgdF0 matrix.
    dYDgdF1: The dYDgdF1 matrix.
    dYDgdFx: The dYDgdFx matrix.
  """
  dYDgdFy = jnp.zeros((nD, ng, ny))
  dYDgdF0 = jnp.zeros((nD, ng, nD))
  dYDgdF1 = jnp.zeros((nD, ng, nD))

  def body_fun(
      iD,
      vals,
  ):
    dYDgdFy, dYDgdF0, dYDgdF1 = vals
    maskD = Opy.T.flatten() == iD

    dYDgdFy = dYDgdFy.at[iD - 1].set((dYygdFy * maskD[:, None, None]).T[0])
    dYDgdF0 = dYDgdF0.at[iD - 1].set(
        (dYygdF0 * maskD[:, None, None]).sum(0)[None, :].reshape(ng, nD)
    )
    dYDgdF1 = dYDgdF1.at[iD - 1].set(
        (dYygdF1 * maskD[:, None, None]).sum(0)[None, :].reshape(ng, nD)
    )
    return dYDgdFy, dYDgdF0, dYDgdF1

  dYDgdFy, dYDgdF0, dYDgdF1 = jax.lax.fori_loop(
      1, nB + 1, body_fun, (dYDgdFy, dYDgdF0, dYDgdF1)
  )

  # Join first 2 dimensions for matrix product compatibility
  dYDgdFy = dYDgdFy.transpose((1, 0, 2)).reshape(nD * ng, ny)
  dYDgdF0 = dYDgdF0.transpose((1, 0, 2)).reshape(nD * ng, nD)
  dYDgdF1 = dYDgdF1.transpose((1, 0, 2)).reshape(nD * ng, nD)

  dYDgdFx = jnp.zeros((nD * ng, nx))

  # Avoid variable-shaped arrays. This is equivalent to:
  # dYDgdFx = dYDgdFx.at[:, lxy.T.flatten()].set(dYDgdFy)
  # lxy is boolean (equivalent to uint8) so we cast to avoid overflow.
  lxy_flat = lxy.T.flatten().astype(jnp.int32)
  lxy_ranks = jnp.cumsum(lxy_flat) - 1
  safe_lxy_ranks = jnp.where(lxy_flat, lxy_ranks, 0)
  dYDgdFx_upscaled = dYDgdFy[:, safe_lxy_ranks]
  dYDgdFx = jnp.where(lxy_flat[None], dYDgdFx_upscaled, dYDgdFx)

  dYDgdFx = dYDgdFx + dYDgdF0 @ dF0dFx + dYDgdF1 @ dF1dFx

  return dYDgdFy, dYDgdF0, dYDgdF1, dYDgdFx


def bfct2(
    FN: jt.Float[jt.Array, 'nQ'],
    Bfp: types.BfpData,
    F0: jt.Float[jt.Array, ''] | None = None,  # pylint: disable=unused-argument
    F1: jt.Float[jt.Array, ''] | None = None,  # pylint: disable=unused-argument
) -> tuple[jt.Float[jt.Array, 'nP+nT nQ'], jt.Float[jt.Array, 'nP+nT nQ']]:
  """Computes normalized basis functions.

  TODO(joeljennings): This defaults to the `bfab.fN` function.

  Args:
    FN: 1D array of normalized basis function values.
    Bfp: Bfp data
    F0: Flux at the magnetic axis.
    F1: Flux at the plasma boundary.

  Returns:
    gQg: Normalized basis functions
    IgQg: Integrated normalized basis functions
  """
  gQg, IgQg = jax.vmap(lambda fq: bfab.fN(fq, Bfp.nP, Bfp.nT))(FN)
  return gQg.T, IgQg.T


def bfct3(
    ag: jt.Float[jt.Array, 'nP+nT'],
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    ids: jt.Float[jt.Array, ''],
    Bfp: types.BfpData,
) -> tuple[
    jt.Float[jt.Array, 'nP+nT'],  # aPpg
    jt.Float[jt.Array, 'nP+nT'],  # aTTpg
    jt.Float[jt.Array, 'nP+nT'],  # aPg
    jt.Float[jt.Array, 'nP+nT'],  # ahqTg
]:
  """Computes the regularization constraints.

  TODO(joeljennings): This defaults to the `bfab` functions.

  Args:
    ag: 1D array of values for the basis functions.
    F0: Flux at the magnetic axis.
    F1: Flux at the plasma boundary.
    ids: 1D array of ids values.
    Bfp: Bfp data.

  Returns:
    aPpg: Coefficient for P'
    aTTpg: Coefficient for TT'
    aPg: Coefficient for P
    ahqTg: Coefficient for TT
  """
  nP, nT = Bfp.nP, Bfp.nT
  fPg, fTg = bfab.fPg(nP, nT)
  alphapg, alphag = bfab.fag(F1 - F0, nP, nT)
  P_coeff = 0.5 * ids * fPg * ag / jnp.pi  # [nP+nT]
  TT_coeff = 2e-7 * ids * fTg * ag  # [nP+nT]
  return (
      P_coeff * alphapg,  # aPpg
      TT_coeff * alphapg,  # aTTpg
      P_coeff * alphag,  # aPg
      TT_coeff * alphag,  # ahqTg
  )


def bfct5(
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    Bfp: types.BfpData,
) -> tuple[jt.Float[jt.Array, 'nP+nT'], jt.Float[jt.Array, 'nP+nT']]:
  """Computes physical basis functions at the magnetic axis.

  TODO(joeljennings): This defaults to the `bfab.fA` function.

  Args:
      F0: Flux at the magnetic axis.
      F1: Flux at the plasma boundary.
      Bfp: Bfp data

  Returns:
      gA: Basis functions `g` at the axis
      IgA: Integrated basis functions `Ig` at the axis
  """
  nP, nT = Bfp.nP, Bfp.nT
  FBA = F1 - F0
  if FBA.ndim == 1:
    assert FBA.shape[0] == 1
    FBA = FBA[0]
  return bfab.fA(FBA, nP, nT)


def bfct11(
    Fx: jt.Float[jt.Array, 'nry+2 nzy+2'],
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
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
  """Computes the derivatives of the transfer matrices.

  This function is a JAX implementation of mode 11 of the bfct function in
  bfct.m. It computes the derivatives of the transfer matrices Tyg and ITyg
  with respect to the flux values FA and FB.

  Args:
      Fx: 2D array of flux values on the grid
      F0: Flux at the magnetic axis.
      F1: Flux at the plasma boundary.
      Opy: 2D boolean mask, 1 if in plasma, 0 otherwise.
      ry: 1D array of major radius values for each row.
      iry: 1D array of 1/R values for each row.
      Bfp: Bfp data

  Returns:
      dTygdFy: d(Tyg)/d(Fy)
      dTygdF0: d(Tyg)/d(F0)
      dTygdF1: d(Tyg)/d(F1)
      dITygdF0: d(ITyg)/d(F0)
      dITygdF1: d(ITyg)/d(F1)
  """
  nP, nT = Bfp.nP, Bfp.nT
  if F0.ndim == 1:
    assert F0.shape[0] == 1
    assert F1.shape[0] == 1
    F0 = F0[0]
    F1 = F1[0]

  def f(fx, fa, fb):
    Tyg, _, ITyg, _ = bfct1(
        fx, fa, fb, Opy, ry, iry, types.BfpData(nP=nP, nT=nT)
    )
    return Tyg, ITyg

  (dTygdFy, dTygdF0, dTygdF1), (_, dITygdF0, dITygdF1) = jax.jacrev(
      f, argnums=(0, 1, 2)
  )(Fx, F0, F1)

  # dTygdFy is [nP+nT, nry * nzy, nry, nzy] so we sum over the last two dims
  return dTygdFy.sum(axis=(-1, -2)), dTygdF0, dTygdF1, dITygdF0, dITygdF1


def bfct15(
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    Bfp: types.BfpData,
) -> tuple[
    jt.Float[jt.Array, 'nP+nT'],
    jt.Float[jt.Array, 'nP+nT'],
    jt.Float[jt.Array, 'nP+nT'],
    jt.Float[jt.Array, 'nP+nT'],
]:
  """Computes the derivatives of the physical basis functions at the magnetic axis.

  Args:
    F0: Flux at the magnetic axis.
    F1: Flux at the plasma boundary.
    Bfp: Bfp data

  Returns:
    d(gA)/d(FA)
    d(gA)/d(FB)
    d(IgA)/d(FA)
    d(IgA)/d(FB)
  """
  (dgAdFA, dgAdFB), (dIgAdFA, dIgAdFB) = jax.jacrev(
      lambda fa, fb: bfct5(fa, fb, Bfp), argnums=(0, 1)
  )(F0, F1)
  return dgAdFA, dgAdFB, dIgAdFA, dIgAdFB


def bfct91(
    F: jt.Float[jt.Array, 'nr nz'],
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    Bfp: types.BfpData,
) -> tuple[
    jt.Float[jt.Array, 'nP+nT nr*nz'], jt.Float[jt.Array, 'nP+nT nr*nz']
]:
  """Computes the values of the physical basis functions.

  TODO(joeljennings): This defaults to the `bfab.f` function.

  Args:
    F: 1D array of flux values.
    F0: Flux at the magnetic axis.
    F1: Flux at the plasma boundary.
    Bfp: Bfp data

  Returns:
    gA: Basis functions `g` at the axis
    IgA: Integrated basis functions `Ig` at the axis
  """
  nP, nT = Bfp.nP, Bfp.nT
  if F0.ndim == 1:
    assert F0.shape[0] == 1
    assert F1.shape[0] == 1
    F0 = F0[0]
    F1 = F1[0]
  gA, IgA = jax.vmap(jax.vmap(lambda fxa: bfab.f(fxa, F1 - F0, nP, nT)))(F - F0)
  return gA.reshape(-1, nP + nT).T, IgA.reshape(-1, nP + nT).T
