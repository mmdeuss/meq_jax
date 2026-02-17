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

"""meqpost JAX implementation.

This overrides the Matlab implementation in
  third_party/meq/meqpost.m
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import bfct_router
from meq_jax._src import meqfx
from meq_jax._src import meqintQ
from meq_jax._src import meqpostq
from meq_jax._src import meqprof
from meq_jax._src import types


# pylint: disable=invalid-name
def meqpost(
    # NOTE: L is so large that executing pytypecheck runs out of memory.
    # L: types.StaticData,
    # LY: types.OutputData,
    L: Any,
    LY: Any,
    # TODO(adedieu): add missing sizes
    ag,
    Fx: jt.Float[jt.Array, 'nrx nzx'],
    FA: jt.Float[jt.Array, ''],
    FB: jt.Float[jt.Array, ''],
    rA: jt.Float[jt.Array, ''],
    zA: jt.Float[jt.Array, ''],
    dr2FA: jt.Float[jt.Array, ''],
    dz2FA: jt.Float[jt.Array, ''],
    drzFA: jt.Float[jt.Array, ''],
    rB: jt.Float[jt.Array, ''],
    zB: jt.Float[jt.Array, ''],
    lB: jt.Float[jt.Array, ''],
    lX: jt.Float[jt.Array, ''],
    rX: jt.Float[jt.Array, ''],
    zX: jt.Float[jt.Array, ''],
    FX,
    dr2FX: jt.Float[jt.Array, ''],
    dz2FX: jt.Float[jt.Array, ''],
    drzFX: jt.Float[jt.Array, ''],
    rBt: jt.Float[jt.Array, ''],
    Ia: jt.Float[jt.Array, 'ne'],
    Iu,
    Iy: jt.Float[jt.Array, 'nrx-2 nzx-2'],
    Opy: jt.Float[jt.Array, 'nrx-2 nzx-2'],
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
) -> types.OutputData:
  """Returns post-processing for various meq functions.

  Computes integral quantities, estimated magnetic measurements,
  profiles, vessel currents. Given:

  Arguments:
    L: MEQ structure
    LY: Equilibrium with shot, t, and containing optional initial aq,aW guesses.
    ag: initial guess
    Fx: flux map
    FA: boundary flux
    FB: boundary flux
    rA: positions flux
    zA: positions flux
    dr2FA: flux hessians
    dz2FA: flux hessians
    drzFA: flux hessians
    rB: boundary flux position
    zB: boundary flux position
    lB: logical indicating whether boundary was found
    lX: logical indicating whether the plasma is diverted.
    rX: positions of X points
    zX: positions of X points
    FX: fluxes of X points
    dr2FX: flux hessians of X points
    dz2FX: flux hessians of X points
    drzFX: flux hessians of X points
    rBt: the vacuum r*Bt
    Ia: circuit currents
    Iu: vessel currents
    Iy: plasma currents
    Opy: plasma current mask.
    F0: limiting fluxes for each domain.
    F1: limiting fluxes for each domain.
  """
  # bfct
  fPg = L.fPg
  fTg = L.fTg
  Bfp = L.bfp

  # assert FA.shape[0] <= L.nD
  # assert FB.shape[0] <= L.nD

  FR = calcFR(
      F0=F0, F1=F1, FX=FX, lX=lX
  )  # X-point/boundary normalized flux ratio

  # cumulative integrals per domain
  assert ag.ndim == 1
  IpQ, WkQ, WpQ, VpQ, FtQ, Ft0Q, _ = meqintQ.meqintQ(
      L=L,
      F0=F0,
      F1=F1,
      rBt=rBt,
      ag=ag[..., None],
      Fx=Fx,
      Opy=Opy,
      L_ny=L.ny,
      Bfp=Bfp,
  )
  FtPQ = FtQ + Ft0Q

  # Integrals per domain
  IpD = IpQ[-1]
  WkD = WkQ[-1]
  WpD = WpQ[-1]
  VpD = VpQ[-1]
  FtD = FtQ[-1]
  Ft0D = Ft0Q[-1]

  WtD = 5.0e6 * rBt * FtD
  Wt0D = 2.5e6 * rBt * Ft0D
  assert L.P is not None
  WND = 1e-7 * jnp.pi * L.P.r0 * IpD**2

  iWND = jnp.where(IpD == 0, 0.0, 1.0 / WND)  # Safe division
  bpD = 2 / 3 * WkD * iWND

  iWt0D = jnp.where(Ft0D == 0, 0.0, 1.0 / Wt0D)  # Safe division
  btD = 2 / 3 * WkD * iWt0D
  liD = WpD * iWND

  # Centroids per domain
  # Use lax.fori_loop for dynamic loop limit nB
  rIpD_init = jnp.zeros((L.nD,))
  zIpD_init = jnp.zeros((L.nD,))

  Opy_flat = Opy.ravel()
  Iy_flat = Iy.ravel()

  assert L.rry is not None
  assert L.zzy is not None

  def centroid_body_fun(iD, val):
    rIpD, zIpD = val
    mask = Opy_flat == iD + 1  # iD is 0-indexed, domain is 1-indexed
    L_rry_masked = jnp.where(mask, L.rry.ravel(), 0.0)
    L_zzy_masked = jnp.where(mask, L.zzy.ravel(), 0.0)
    IY_flat_masked = jnp.where(mask, Iy_flat, 0.0)
    rIp_val = jnp.sum(L_rry_masked * IY_flat_masked)
    zIp_val = jnp.sum(L_zzy_masked * IY_flat_masked)
    rIpD = rIpD.at[iD].set(rIp_val)
    zIpD = zIpD.at[iD].set(zIp_val)
    return rIpD, zIpD

  rIpD, zIpD = jax.lax.fori_loop(
      0, L.nD, centroid_body_fun, (rIpD_init, zIpD_init)
  )

  rYD = jnp.where(IpD == 0, 0.0, rIpD / IpD)  # Safe division
  zYD = jnp.where(IpD == 0, 0.0, zIpD / IpD)

  # Total integrals
  Ip = jnp.sum(IpD).astype(float)
  Wk = jnp.sum(WkD).astype(float)
  Wp = jnp.sum(WpD).astype(float)
  Vp = jnp.sum(VpD).astype(float)
  Ft = jnp.sum(FtD).astype(float)
  Ft0 = jnp.sum(Ft0D).astype(float)
  Wt = jnp.sum(WtD).astype(float)
  Wt0 = jnp.sum(Wt0D).astype(float)
  WN = 1e-7 * jnp.pi * L.P.r0 * Ip**2

  iWN = jnp.where(Ip == 0, 0.0, 1.0 / WN)  # Safe division

  bp = 2 / 3 * Wk * iWN
  bt = jnp.where(Wt0 == 0, 0.0, 2 / 3 * Wk / Wt0)  # Safe division
  li = Wp * iWN
  mu = jnp.where(WN == 0, 0.0, Wt / WN)  # Safe division
  bpli2 = bp + li / 2

  # for the Ipmin case, recompute Ip from Iy
  # Use lax.cond for data-dependent conditional
  Ip = jax.lax.cond(
      jnp.logical_or(Ip == 0, jnp.all(ag == 0)),
      lambda _: jnp.sum(Iy),
      lambda _: Ip,
      None,
  )

  # Centroids
  rIp = jnp.sum(L.rry.ravel() * Iy.ravel())
  zIp = jnp.sum(L.zzy.ravel() * Iy.ravel())

  rY = jnp.where(Ip == 0, 0.0, rIp / Ip)  # Safe division
  zY = jnp.where(Ip == 0, 0.0, zIp / Ip)

  # q axis using small diamagnetic approximation (LIUQE 2015 paper eq. 88)
  qA_init = jnp.zeros((L.nD))

  gAg, IgAg = bfct_router.bfct5(F0=F0, F1=F1, Bfp=Bfp)
  if gAg.ndim == 1:
    gAg = gAg[:, None]
  if IgAg.ndim == 1:
    IgAg = IgAg[:, None]

  IgAg = 2e-7 * L.idsx * IgAg

  def qA_body_fun(iA, qA):
    TyAg = gAg[:, iA] * (rA[iA] * fPg + fTg / rA[iA])
    cqA_num = (
        4e-7
        * jnp.pi
        * rA[iA] ** 2
        * rBt
        * jnp.sqrt(dr2FA[iA] * dz2FA[iA] - drzFA[iA] ** 2)
    )
    cqA_den = jnp.abs(dr2FA[iA] + dz2FA[iA]) * L.dsx
    cqA = jnp.where(cqA_den == 0, jnp.inf, cqA_num / cqA_den)  # Safe division

    numerator = rBt**2 + (IgAg[:, iA] * fTg) @ ag
    denominator = cqA * (TyAg @ ag)

    q_val = jnp.where(
        denominator == 0, jnp.inf, numerator / denominator
    )  # Safe division
    qA = qA.at[iA].set(q_val)
    return qA

  qA = jax.lax.fori_loop(0, L.nD, qA_body_fun, qA_init)

  # Profiles
  assert L.pQ is not None
  PpQ, TTpQ, PQ, TQ, iTQ, PpQg, _ = meqprof.meqprof(
      ag=ag,
      FN=L.pQ.ravel() ** 2,
      F0=F0,
      F1=F1,
      rBt=rBt,
      idsx=L.idsx,
      Bfp=Bfp,
      smalldia=L.smalldia,
  )
  # Meprof outputs are not correctly transposed
  TQ = TQ.T

  # Fictional measurements
  assert L.G is not None
  assert L.G.Bmx is not None
  assert L.G.Mfx is not None
  assert L.lxy is not None
  Bmx = L.G.Bmx.reshape((L.lxy.shape[0], L.lxy.shape[1], L.G.Bmx.shape[1]))
  Bmx = Bmx[1:-1, 1:-1, :]
  Bmx = Bmx.reshape((-1, L.G.Bmx.shape[1]))

  Mfx = L.G.Mfx.reshape((L.lxy.shape[0], L.lxy.shape[1], L.G.Bmx.shape[1]))
  Mfx = Mfx[1:-1, 1:-1, :]
  Mfx = Mfx.reshape((-1, L.G.Mfx.shape[1]))

  Bm = Ia @ L.G.Bma + Iy.ravel() @ Bmx
  if L.G.Bmu is not None:
    Bm += Iu @ L.G.Bmu

  Ff = Ia @ L.G.Mfa + Iy.ravel() @ Mfx
  if L.G.Mfu is not None:
    Ff += Iu @ L.G.Mfu

  # Vessel current from generic representation
  Iv = None
  if L.G.Tvu is not None:
    Iv = Iu @ L.G.Tvu
  Is = None
  if L.G.Tius is not None:
    Is = Iu @ L.G.Tius

  nA = jnp.isfinite(FA).sum()
  nB = jnp.isfinite(FB).sum()
  nX = jnp.isfinite(FX).sum()

  def _process_1D(attr):
    attr = attr.squeeze().T
    return jnp.atleast_1d(attr)

  def _process_2D(attr):
    attr = attr.squeeze().T
    if attr.ndim == 1:
      return jnp.array(attr[None])
    else:
      return jnp.array(attr)

  new_ly_fields = {
      # Flux maps
      'Fx': Fx,
      'rB': rB,
      'zB': zB,
      'FB': FB,
      'rA': rA,
      'zA': zA,
      'FA': FA,
      'lB': lB,
      'lX': lX,
      'dr2FA': dr2FA,
      'dz2FA': dz2FA,
      'drzFA': drzFA,
      'rX': rX,
      'zX': zX,
      'FX': FX,
      'nX': nX,
      'Opy': Opy,
      'nA': nA,
      'nB': nB,
      'FR': FR,
      'dr2FX': dr2FX,
      'dz2FX': dz2FX,
      'drzFX': drzFX,
      'F0': F0,
      'F1': F1,
      # Current, measurements
      'Iy': Iy,
      'Ia': Ia,
      'Iu': Iu,
      'Iv': Iv,
      'Is': Is,
      'Bm': Bm,
      'Ff': Ff,
      'ag': ag,
      # Integrals per Q
      'IpQ': IpQ.T,  # meqintQ outputs are not transposed
      'VpQ': VpQ.T,  # meqintQ outputs are not transposed
      'FtPQ': FtPQ.T,  # meqintQ outputs are not transposed
      'PpQg': PpQg.T,  # meqintQ outputs are not transposed
      'PpQ': PpQ.T,  # meqintQ outputs are not transposed
      'TTpQ': TTpQ.T,  # meqintQ outputs are not transposed
      'iTQ': iTQ.T,  # meqintQ outputs are not transposed
      'PQ': PQ.T,  # meqintQ outputs are not transposed
      'TQ': TQ,
      # Integrals per domain
      'IpD': IpD,
      'zYD': zYD.T,
      'rYD': rYD.T,
      'zIpD': zIpD,
      'rIpD': rIpD,
      'WkD': WkD,
      'WpD': WpD,
      'WtD': WtD,
      'Wt0D': Wt0D,
      'WND': WND,
      'VpD': VpD,
      'FtD': FtD,
      'Ft0D': Ft0D,
      'bpD': bpD,
      'btD': btD,
      'liD': liD,
      # Integrals
      'Ip': Ip,
      'zY': zY,
      'rY': rY,
      'zIp': zIp,
      'rIp': rIp,
      'rBt': rBt,
      'Wk': Wk,
      'Wp': Wp,
      'Wt': Wt,
      'Wt0': Wt0,
      'WN': WN,
      'Vp': Vp,
      'Ft': Ft,
      'Ft0': Ft0,
      'bp': bp,
      'bt': bt,
      'mu': mu,
      'li': li,
      'bpli2': bpli2,
      'qA': qA,
  }
  # Unexpected arguments
  LY = dataclasses.replace(LY, **new_ly_fields)

  if L.P.iterq:  # Static conditional
    LY = meqpostq.meqpostq(L, LY)

  # Optional interpolation of flux, fields in desired points
  assert not L.nn, 'Code not tested with nn is True'
  if L.nn:
    Fn, Brn, Bzn, Brrn, Brzn, Bzrn, Bzzn = L.P.infct(
        L.rx, L.zx, Fx, L.P.rn, L.P.zn, L.inM
    )
  else:
    Fn = None
    Brn = None
    Bzn = None
    Brrn = None
    Brzn = None
    Bzrn = None
    Bzzn = None

  new_ly_fields = {
      'Fn': Fn,
      'Brn': Brn,
      'Bzn': Bzn,
      'Brrn': Brrn,
      'Brzn': Brzn,
      'Bzrn': Bzrn,
      'Bzzn': Bzzn,
  }
  LY = dataclasses.replace(LY, **new_ly_fields)

  # optional computation of Br,Bz fields on x grid
  Brx, Bzx = None, None
  if L.P.ifield:  # Static conditional
    Brx, Bzx = meqBrBz(Fx, L.i4pirxdzx, L.i4pirxdrx, L.nzx, L.nrx)
    Btx = meqBt(L, Fx, Opy, ag, rBt, F0, F1, TQ)
    LY = dataclasses.replace(LY, Brx=Brx, Bzx=Bzx, Btx=Btx)

  # Vacuum flux
  F0x = None
  if L.P.ivacuum:
    I_concat = jnp.concatenate([Ia, Iu]) if Iu is not None else Ia
    if I_concat.ndim == 1:
      I_concat = I_concat[None]

    F0x = jax.lax.cond(
        jnp.any(Iy),
        # recompute flux with Iy=0
        lambda _: meqfx.meqFx(L, jnp.zeros((1, L.nry, L.nzy)), I_concat),
        # no plasma, Fx is already vacuum case
        lambda _: Fx,
        None,
    )

  # Optional computation of fields/fluxes on z grid
  Fz = None
  F0z = None
  Brz, Bzz = None, None
  if L.P.izgrid:
    assert L.G is not None
    F0z = LY.Ia @ L.G.Mza
    if LY.Iu is not None and L.G.Mzu is not None:
      F0z += LY.Iu @ L.G.Mzu
    Fz = F0z + Iy.ravel() @ L.G.Mzy

    if L.P.izxoverlap:
      assert L.G.lzx is not None
      lzx = L.G.lzx.ravel()
      assert F0x is not None
      update_indices = jnp.where(lzx, size=int(Fx.ravel().shape[0]))[0]

      Fz = Fz.at[update_indices].set(Fx.ravel())
      # Fz = jnp.where(lzx, Fz, Fx.ravel())
      if L.P.ivacuum:
        F0z = F0z.at[update_indices].set(F0x.ravel())

    if Fz is not None:
      Fz = Fz.reshape(L.nrz, L.nzz)
    if F0z is not None:
      F0z = F0z.reshape(L.nrz, L.nzz)
    LY = dataclasses.replace(LY, F0x=F0x, Fz=Fz)

    # Fields
    if L.P.ifield:
      Brz, Bzz = meqBrBz(Fz, L.i4pirzdzz, L.i4pirzdrz, L.nzz, L.nrz)
      LY = dataclasses.replace(LY, Brz=Brz, Bzz=Bzz)

  # optional vacuum field/flux computation
  if L.P.ivacuum:
    if L.P.izgrid:
      new_F0z = jax.lax.cond(
          jnp.any(Iy),
          lambda _: F0z,
          lambda _: Fz,
          None,
      )
      LY = dataclasses.replace(LY, F0z=new_F0z)

    if L.P.ifield:
      new_Br0x, new_Bz0x = jax.lax.cond(
          jnp.any(Iy),
          lambda _: meqBrBz(LY.F0x, L.i4pirxdzx, L.i4pirxdrx, L.nzx, L.nrx),
          lambda _: (Brx, Bzx),
          None,
      )
      LY = dataclasses.replace(LY, Br0x=new_Br0x, Bz0x=new_Bz0x)

    if L.P.ifield and L.P.izgrid:
      new_Br0z, new_Bz0z = jax.lax.cond(
          jnp.any(Iy),
          lambda _: meqBrBz(LY.F0z, L.i4pirzdzz, L.i4pirzdrz, L.nzz, L.nrz),
          lambda _: (Brz, Bzz),
          None,
      )
      LY = dataclasses.replace(LY, Br0z=new_Br0z, Bz0z=new_Bz0z)

  # NOTE: The final loop from MATLAB that uses `eval` is omitted.
  return LY


def calcFR(F0, F1, FX, lX):
  """Returns ratio of normalized flux at x-point vs normalized flux at boundary."""
  sFBA = jnp.sign(F1[0] - F0[0])
  nD = F1.shape[0]
  FR_init = jnp.zeros((nD,))

  def body_fun(iD, FR):
    F0_i, F1_i, lX_i = F0[iD], F1[iD], lX[iD]

    # Define all possible outcomes as pure functions
    def case_empty():
      return jnp.nan

    def case_diverted():
      return 1.0

    def case_main():
      FB = F1_i

      def case_mantle():
        FA = sFBA * jnp.min(F0 * sFBA)
        FXi = F0_i
        return jnp.where(FB == FA, jnp.inf, (FXi - FA) / (FB - FA))

      def case_axis():
        FA = F0_i

        # This guarantees that, when FX is of size 0, case_axis return jnp.inf
        nX = FX.shape[0]
        FX_extended = jnp.full((nX + 1,), fill_value=(FB - FA) * jnp.inf)
        FX_extended = FX_extended.at[:nX].set(FX[:nX])

        iX = jnp.argmin(jnp.abs(FX_extended - FB))
        FXi = FX_extended[iX]
        return jnp.where(FB == FA, jnp.inf, (FXi - FA) / (FB - FA))

      return jax.lax.cond(jnp.any(F0_i == FX), case_mantle, case_axis)

    # Chain the conditions
    fr_val = jax.lax.cond(
        F0_i == F1_i,
        case_empty,
        lambda: jax.lax.cond(
            lX_i,
            case_diverted,
            case_main,
        ),
    )
    return FR.at[iD].set(fr_val)

  FR = jax.lax.fori_loop(0, nD, body_fun, FR_init)
  return FR


def meqBrBz(
    Fx: jt.Float[jt.Array, 'nrx nzx'],
    i4pirxdzx: jt.Float[jt.Array, 'nrx'],
    i4pirxdrx: jt.Float[jt.Array, 'nrx'],
    nzx: jt.Float[jt.Array, ''],
    nrx: jt.Float[jt.Array, ''],
) -> tuple[jt.Float[jt.Array, 'nrx nzx'], jt.Float[jt.Array, 'nrx nzx']]:
  """Returns Br,Bz fields."""
  assert Fx.ndim == 2
  shape = (nrx, nzx)
  Br, Bz = jnp.zeros(shape), jnp.zeros(shape)

  # Br = -1/(2*pi*R)* dF/dz
  # Central differences
  Br = Br.at[:, 1:-1].set(-i4pirxdzx[:, None] * (Fx[:, 2:] - Fx[:, :-2]))
  # At grid boundary
  Br = Br.at[:, -1].set(
      -i4pirxdzx * (Fx[:, -3] - 4 * Fx[:, -2] + 3 * Fx[:, -1])
  )
  Br = Br.at[:, 0].set(-i4pirxdzx * (-Fx[:, 2] + 4 * Fx[:, 1] - 3 * Fx[:, 0]))

  # Bz = 1/(2*pi*R)* dF/dr
  Bz = Bz.at[1:-1, :].set(i4pirxdrx[1:-1][:, None] * (Fx[2:, :] - Fx[:-2, :]))
  # Same as for Br
  Bz = Bz.at[-1, :].set(
      i4pirxdrx[-1] * (Fx[-3, :] - 4 * Fx[-2, :] + 3 * Fx[-1, :])
  )
  Bz = Bz.at[0, :].set(i4pirxdrx[0] * (-Fx[2, :] + 4 * Fx[1, :] - 3 * Fx[0, :]))
  return Br, Bz


def meqBt(L, Fx, Opy, _, rBt, F0, F1, TQ):
  """Computes toroidal field on x grid."""
  Btx = rBt * jnp.tile(1.0 / L.rx[:, None], (1, L.nzx))
  Bty_init = Btx[1:-1, 1:-1]  # 1D vector

  def bt_body_fun(iB, Bty):
    # Use static `if` since function name is static
    assert (
        L.bfct != 'bf3pmex'
    ), "Code not tested with bfct == 'bf3pmex. Also bfct8 is not supported yet'"

    den = F1[iB] - F0[iB]
    FyN_1d = (Fx[1:-1, 1:-1] - F0[iB]) / jnp.where(den == 0, 1.0, den)
    FyN_1d = jnp.where(den == 0, 0.0, FyN_1d)  # Handle 0/0

    # jnp.interp is the JAX version of `interp1`
    Bty_interp = jnp.interp(FyN_1d, L.pQ**2, TQ[iB])
    assert Bty_interp.shape == Opy.shape
    assert L.rry.shape == Opy.shape

    Bty = jnp.where(Opy == iB + 1, Bty_interp / (1e-9 + L.rry), Bty)
    return Bty

  Bty_final = jax.lax.fori_loop(0, L.nD, bt_body_fun, Bty_init)
  Btx = Btx.at[1:-1, 1:-1].set(Bty_final)
  return Btx


# pylint: enable=invalid-name
