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

"""fgeF JAX implementation.

This overrides the Matlab implementation in
  third_party/meq/fgeF.m
"""

import dataclasses
import functools
from typing import Any, List

import jax
import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import asxy
from meq_jax._src import asxycs
from meq_jax._src import bfct
from meq_jax._src import bfct_router
from meq_jax._src import meqagcon
from meq_jax._src import meqcde
from meq_jax._src import meqfbp
from meq_jax._src import meqfx
from meq_jax._src import meqIyJac
from meq_jax._src import meqpdom
from meq_jax._src import meqpost
from meq_jax._src import types


# pylint: disable=invalid-name
def co2lx(
    L: types.StaticData,
    LX: types.InputData,
    agconc: list[types.ConcData],
    u: jt.Float[jt.Array, 'n'],
):
  """Implementation of Co2LX.m. One time update of LX based on u and agconc."""
  assert L.ind is not None
  Co = u[L.ind.iuC - 1]

  # Update LX
  assert len(agconc) == L.nC

  # Implementation of Co2LX.m
  for idx, ag_data in enumerate(agconc):
    field = ag_data.lx_name
    index = ag_data.index
    if field == 'fbtlegacy':
      continue  # Skip constraints without LX field

    val = getattr(LX, field)
    # Shift index by 1 to match Matlab indexing
    val_arr = jnp.atleast_1d(val)
    if index - 1 < val_arr.shape[0]:
      val_arr = val_arr.at[index - 1].set(Co[idx])
      val = val_arr[0] if jnp.ndim(val) == 0 else val_arr
    else:
      val = jnp.concatenate([val_arr, Co[idx][None]])
    setattr(LX, field, val)

  assert L.nD is not None
  if L.nD > 1:
    assert LX.IpD is not None
    setattr(LX, 'Ip', LX.IpD.sum())
  return LX


def _fgeF_u2LXLY(
    u: jt.Float[jt.Array, 'n'],
    xdot: jt.Float[jt.Array, 'n'],
    L: types.StaticData,
    LX: types.InputData,
    LYp: types.OutputData | types.InputData,  # InputData for fgeF_test
    xHasIy: bool,
) -> tuple[types.InputData, types.OutputData | types.InputData]:
  """Update LX and LYp based on u and xdot input for fgeF."""
  assert L.ind is not None and L.P is not None

  Va = None
  Ini = None
  Rp = None
  dIy = None
  dFx = None
  dag = None
  dIa = None
  dIu = None
  Ia = None
  Iu = None

  # Remember that Matlab and Python indices differ by 1
  if L.isEvolutive:
    # u = [Va;Co;Ini;Rp]
    Va = u[L.ind.iua - 1]
    Ini = u[L.ind.iuni - 1]
    Rp = u[L.ind.iurp - 1]
    # x = [GS;ag;Ia;Iu]
    dt = LX.t - LYp.t
    # if jnp.any(xdot) and dt == 0:
    #   raise ValueError('fgeF with xdot as argument requires dt>0')
    dxSI = xdot * (L.xscal * dt)
    if xHasIy:
      dIy = dxSI[L.ind.ixGS - 1]
    else:
      dFx = dxSI[L.ind.ixGS - 1]
    dag = dxSI[L.ind.ixg - 1]
    dIa = dxSI[L.ind.ixa - 1]
    dIu = dxSI[L.ind.ixu - 1]
  else:
    # u = [Ia;Iu;Co]
    Ia = u[L.ind.iua - 1]
    Iu = u[L.ind.iuu - 1]

  if L.isEvolutive:
    LX = dataclasses.replace(LX, Va=Va, IniD=Ini, Ini=jnp.sum(Ini), Rp=Rp)

    # Update LYp to compute dFdxdot = -dt*dFdx0 (x0 is the previous state)
    lyp_dict = {}
    if xHasIy:
      assert dIy is not None
      assert LYp.Iy is not None
      lyp_dict['Iy'] = LYp.Iy - dIy.reshape(LYp.Iy.shape)
    else:
      assert dFx is not None
      assert LYp.Fx is not None
      lyp_dict['Fx'] = LYp.Fx - dFx.reshape(LYp.Fx.shape)
    lyp_dict['ag'] = LYp.ag - dag
    lyp_dict['Ia'] = LYp.Ia - dIa
    lyp_dict['Iu'] = LYp.Iu - dIu
    LYp = dataclasses.replace(LYp, **lyp_dict)

    Ia_Iu = jnp.concatenate([LYp.Ia, LYp.Iu])

    # Update Iy for circuit equations
    if not xHasIy:
      Iy_new = -(LYp.Fx.flatten() @ L.dlst) / L.rhsf + Ia_Iu @ L.Tye
      Iy_new = Iy_new.reshape(L.nry, L.nzy)
      LYp = dataclasses.replace(LYp, Iy=Iy_new)

    # Update fields necessary for meqcde time derivatives
    assert L.np is not None
    if L.np > 0:
      if xHasIy:
        new_IyD = LYp.Iy
        if new_IyD.ndim == 2:
          new_IyD = new_IyD[None]
        if Ia_Iu.ndim == 1:
          Ia_Iu = Ia_Iu[None]
        Fx_new = meqfx.meqFx(L=L, IyD=new_IyD, Ie=Ia_Iu)
        LYp = dataclasses.replace(LYp, Fx=Fx_new)

      IpD_new = jnp.zeros(L.nD)
      for iD in range(L.nD):
        new_val = jnp.sum(LYp.Iy.flatten() * (LYp.Opy.flatten() == iD + 1))
        IpD_new = IpD_new.at[iD].set(new_val)
      LYp = dataclasses.replace(LYp, IpD=IpD_new)

      assert isinstance(LYp, types.OutputData)
      assert isinstance(L.P, types.GeometryData)
      res_meqpdom = meqpdom.meqpdom(
          fx=LYp.Fx, Ip=LYp.Ip, isaddl=L.P.isaddl, L=L
      )
      FA = res_meqpdom[2]
      FB = res_meqpdom[14]
      FA = FA[jnp.isfinite(FA)]
      FB = FB[jnp.isfinite(FB)]
      F0 = res_meqpdom[18]
      F1 = res_meqpdom[19]
      LYp = dataclasses.replace(LYp, FA=FA, FB=FB, F0=F0, F1=F1)
  else:
    LX = dataclasses.replace(LX, Ia=Ia, Iu=Iu)
  return LX, LYp


def _fgepost(
    L: types.StaticData,
    LX: types.InputData,
    LYp: types.OutputData,
    ag,
    Fx,
    FA,
    FB,
    rA,
    zA,
    dr2FA,
    dz2FA,
    drzFA,
    rB,
    zB,
    lB,
    lX,
    rX,
    zX,
    FX,
    dr2FX,
    dz2FX,
    drzFX,
    rBt,
    Ia,
    Iu,
    Iy,
    Opy,
    F0,
    F1,
    resy,
    resC,
    resp,
    rese,
    resFx,
):
  """Post-processing for fgeF."""
  LY = types.OutputData(shot=LX.shot, t=LX.t, aq=LYp.aq, aW=LYp.aW)
  LY = meqpost.meqpost(
      L=L,
      LY=LY,
      ag=ag,
      Fx=Fx,
      FA=FA,
      FB=FB,
      rA=rA,
      zA=zA,
      dr2FA=dr2FA,
      dz2FA=dz2FA,
      drzFA=drzFA,
      rB=rB,
      zB=zB,
      lB=lB,
      lX=lX,
      rX=rX,
      zX=zX,
      FX=FX,
      dr2FX=dr2FX,
      dz2FX=dz2FX,
      drzFX=drzFX,
      rBt=rBt,
      Ia=Ia,
      Iu=Iu,
      Iy=Iy,
      Opy=Opy,
      F0=F0,
      F1=F1,
  )

  assert L.G is not None
  if L.isEvolutive:
    Va = LX.Va
    IniD = LX.IniD
  else:
    Va = jnp.zeros(L.G.na)
    IniD = jnp.zeros(L.nD)

  LY = dataclasses.replace(
      LY, resy=resy, resC=resC, rese=rese, resFx=resFx, Va=Va, IniD=IniD
  )
  return LY


def fgeF(
    x: jt.Float[jt.Array, 'n'],
    L: types.StaticData,
    LX: types.InputData,
    LYp: types.OutputData | types.InputData,  # InputData for fgeF_test
    opts: types.Opts,
    aux: List[Any],
    u: jt.Float[jt.Array, 'n'],
    xdot: jt.Float[jt.Array, 'n'] | None,
    agconc: list[types.ConcData],
    cdeconc: list[types.ConcData],
):
  """FGE residual function."""

  assert L.ind is not None and L.P is not None
  auxHasFbe = False
  auxHasOpy = False
  auxHasdz = False
  picardJac = False
  if aux is not None and len(aux) > 1:
    auxHasFbe = aux[0] is not None and aux[1] is not None
    auxHasOpy = len(aux) > 2 and aux[2] is not None
    auxHasdz = len(aux) > 3 and aux[3] is not None
    picardJac = len(aux) > 5 and not aux[5]

  # Identify algoNL variant
  # xHasIy: state has Iy, rHasFx: residual has Fx
  assert L.P.algoNL is not None
  if L.P.algoNL.endswith('l'):  # all-nl
    xHasIy = True
    rHasFx = False
  elif L.P.algoNL.endswith('x'):  # all-nl-Fx
    xHasIy = False
    rHasFx = True
  elif L.P.algoNL.endswith('S'):  # Newton-GS
    raise ValueError('Newton-GS has not been tested yet')
    xHasIy = False
    rHasFx = False
  else:
    raise ValueError(f'Unknown algoNL: {L.P.algoNL}')

  if u is not None:
    if xdot is None:
      xdot = jnp.zeros(((L.nN)))
    LX, LYp = _fgeF_u2LXLY(u=u, xdot=xdot, L=L, LX=LX, LYp=LYp, xHasIy=xHasIy)

  # assert L.np == 0, 'Tested code only supports L.np == 0'

  # Indicates whether jacobian of static equations will be computed
  dojacstat = opts.dojacx or opts.dojacu or opts.dojacF
  # Indicates whether any form of jacobian will be computed
  dojac = dojacstat or opts.dojacxdot

  res = jnp.zeros(L.nN)
  LY = LYp if opts.dopost else None
  Jx = None
  Ju = None
  Jxdot = None
  # Use pre-filled jacobians
  if opts.dojacx and L.Jx is not None:
    Jx = jnp.copy(L.Jx)
  if opts.dojacu and L.Ju is not None:
    Ju = jnp.copy(L.Ju)
  if opts.dojacxdot and L.Jxdot is not None:
    Jxdot = jnp.copy(L.Jxdot)
  rowmask = jnp.zeros((L.nN,))

  # Time step
  if L.isEvolutive:
    dt = LX.t - LYp.t
    idt = jnp.where(dt == 0, 0, 1.0 / (dt + 1e-9))
  else:
    dt = 0
    idt = 0

  # Scales
  ire = None
  iue = None
  ixe = None
  xscalGS = L.xscal[L.ind.ixGS - 1]
  xscalg = L.xscal[L.ind.ixg - 1]
  resscalc = None
  if L.isEvolutive:
    ixe = jnp.concatenate([L.ind.ixa - 1, L.ind.ixu - 1])
    ire = jnp.concatenate([L.ind.ira - 1, L.ind.iru - 1])
    xscale = L.xscal[ixe]
  else:
    iue = jnp.concatenate([L.ind.iua - 1, L.ind.iuu - 1])
    xscale = 1.0

  resscalGS = L.resscal[L.ind.irGS - 1]
  assert L.np is not None
  resscalp = None
  if L.np > 0:
    resscalp = L.resscal[L.ind.irp - 1]
  if L.isEvolutive:
    assert ire is not None
    resscalc = L.resscal[ire]

  # Extract NL unknowns
  Iy = None
  Fx = None
  xSI = L.xscal * x
  if xHasIy:
    Iy = xSI[L.ind.ixGS - 1].reshape(L.nry, L.nzy)
  else:
    Fx = xSI[L.ind.ixGS - 1].reshape(L.nrx, L.nzx)
  ag = xSI[L.ind.ixg - 1]
  assert ag.shape == (L.ng,)

  if L.isEvolutive:
    Ia = xSI[L.ind.ixa - 1]
    Iu = xSI[L.ind.ixu - 1]
  else:
    Ia = LX.Ia
    Iu = LX.Iu
  Ie = jnp.concatenate([Ia, Iu])

  # Sign of Ip
  Ip = LX.Ip
  assert Ip is not None
  sIp = jnp.sign(Ip)
  rBt = LX.rBt

  # smaller than Ipmin flag
  # assign_LXIy = abs(Ip) < L.P.Ipmin  # Ip is static

  # NOTE: The code has not been tested with assign_LXIy
  # if assign_LXIy and L.np > 0:
  #   raise ValueError('LX.Iy can only be assigned if no CDE is selected')
  # if assign_LXIy:
  #   raise ValueError('The code has not been tested with assign_LXIy')
  # TODO(pfau): implement assign_LXIy in a jit-compatible way,
  # since Ip is not static
  assign_LXIy = False

  # External currents contributions to meqFx
  if L.isEvolutive or not auxHasFbe:
    Fbe = Ie @ L.Mbe
    Iyie = (Ie @ L.Tye).reshape(L.nry, L.nzy)
  else:
    Fbe = aux[0]
    Iyie = aux[1]

  # Get missing Fx or Iy
  if xHasIy:
    assert Iy is not None
    if auxHasdz:
      dz = aux[3]
      assert L.nD is not None
      if L.nD > 1 and aux[4] is not None:
        Opy_aux = aux[4]
        IyD = jnp.zeros((L.nD, L.nry, L.nzy))
        for iD in range(L.nD):
          IyD = IyD.at[iD].set(Iy * (Opy_aux == iD + 1))
      else:
        IyD = jnp.zeros((L.nD, L.nry, L.nzy))
        IyD = IyD.at[0].set(Iy)
      Fx = meqfx.meqFx(
          L=L,
          IyD=IyD,
          Ie=Ie,
          Fb_ext=(Fbe, Iyie),
          dz=dz,
          rst=False,
          Jh=jnp.zeros(L.nh),
      )
    else:
      new_Iy = Iy.copy()
      if new_Iy.ndim == 2:
        new_Iy = new_Iy[None]
      new_Ie = Ie.copy()
      if new_Ie.ndim == 1:
        new_Ie = new_Ie[None]
      new_Iyie = Iyie.copy()
      if new_Iyie.ndim == 2:
        new_Iyie = new_Iyie[None]
      Fx = meqfx.meqFx(L=L, IyD=new_Iy, Ie=new_Ie, Fb_ext=(Fbe, new_Iyie))
  else:
    assert Fx is not None
    Iy = -((Fx.flatten() @ L.dlst) / L.rhsf).reshape(L.nry, L.nzy) - Iyie

  # Plasma domain and current distribution
  (
      rA,
      zA,
      FA,
      dr2FA,
      dz2FA,
      drzFA,
      rX,
      zX,
      FX,
      dr2FX,
      dz2FX,
      drzFX,
      rB,
      zB,
      FB,
      lB,
      lX,
      Opy,
      F0,
      F1,
      stat,
      dF0dFx,
      dF1dFx,
      ixI,
  ) = meqpdom.meqpdom(fx=Fx, Ip=sIp, isaddl=L.P.isaddl, L=L)

  if opts.dojacx and picardJac:
    dF0dFx = jnp.zeros((L.nx, L.nD))
    dF1dFx = jnp.zeros((L.nx, L.nD))
    ixI = jnp.full((L.nD,), fill_value=L.nzx + 1)
  else:
    assert dF0dFx.shape == (L.nD, L.nx)
    dF0dFx = dF0dFx.T
    dF1dFx = dF1dFx.T

  # Enforce Opy value
  if auxHasOpy:
    Opy = aux[2]

  nA = FA.size
  nB = FB.size

  # NOTE: rather than having a giant jax.lax.cond, we ignore the failure case
  # if stat != 1:
  #   warnings.warn(f'meqpdom failed for shot {LX.shot} at t={LX.t}')
  #   res = jnp.full_like(res, jnp.nan)
  #   # TODO(pfau): implement fgepost call for failed pdomstat
  #   return res, LY, Jx, Ju, Jxdot, rowmask

  if L.code == 'liu':
    raise ValueError('liuNLmeas has not been ported yet')

  # Plasma current distribution from basis function coefficients
  if not assign_LXIy:
    Tyg, TpDg, _, ITpDg = bfct_router.bfct1(
        Fx=Fx, F0=F0, F1=F1, Opy=Opy, ry=L.ry, iry=L.iry, Bfp=L.bfp
    )
    Iy1 = (ag @ Tyg).reshape(L.nry, L.nzy)
  else:
    Iy1 = LX.Iy  # Prescribe Iy from LX
    Tyg = jnp.zeros((L.ng, L.ny))
    TpDg = jnp.zeros((L.ng, L.nD))
    ITpDg = jnp.zeros((L.ng, L.nD))

  # Assemble dIy/dFx
  dTygdFy = None
  dTygdF0 = None
  dTygdF1 = None
  dITygdF0 = None
  dITygdF1 = None
  dIydIy = None
  dIydIe = None
  dIypdFx = None

  if dojacstat:
    # Option 1
    if not assign_LXIy and not picardJac:
      # Iy depends non-linearly on Fx
      dTygdFy, dTygdF0, dTygdF1, dITygdF0, dITygdF1 = bfct_router.bfct11(
          Fx=Fx, F0=F0, F1=F1, Opy=Opy, ry=L.ry, iry=L.iry, Bfp=L.bfp
      )
      if dTygdF0.ndim == 2:
        dTygdF0 = dTygdF0[None]
        assert dTygdF1.ndim == 2
        dTygdF1 = dTygdF1[None]
        assert dITygdF0.ndim == 2
        dITygdF0 = dITygdF0[None]
        assert dITygdF1.ndim == 2
        dITygdF1 = dITygdF1[None]

      # Domains where there is some current
      mask = jnp.zeros((L.ny,), dtype=bool)
      assert L.TDg is not None
      active_domains = L.TDg.shape[0]
      for iD in range(active_domains):
        mask = jnp.logical_or(mask, (Opy.flatten() == iD + 1))

      # Derivatives of Iyp
      # In the Matlab version, dIypdFx may not be computed and the flag
      # hasdIypdFx is used to indicate this. Here, it is always computed, 
      # and we can skip the flag.
      dIypdFy, dIypdF0, dIypdF1, dIypdFx = meqIyJac.meqIyJac(
          lxy=L.lxy,
          ag=ag,
          mask=mask,
          dTygdFy=dTygdFy,
          dTygdF0=dTygdF0,
          dTygdF1=dTygdF1,
          dF0dFx=dF0dFx,
          dF1dFx=dF1dFx,
      )

      if xHasIy:
        # Assemble dIydIy/dIydIe when needed
        hasdIydIy = opts.dojacx
        hasdIydIe = (opts.dojacx and L.isEvolutive) or (
            opts.dojacu and not L.isEvolutive
        )

        # dIydIy
        if hasdIydIy:
          dIydIy = L.Mxy @ dIypdFx

        # dIydIe
        if hasdIydIe:
          dIydIe = L.Mxe @ dIypdFx

    # Option 2
    elif assign_LXIy:
      # Iy is independent of Fx
      # Iyp derivatives only needed for function handle
      if ~opts.dojacF:
        # Neglecting dependence of Iy on Fx
        dITygdF0 = jnp.zeros((L.nD, L.ng, L.ny))
        dITygdF1 = jnp.zeros((L.nD, L.ng, L.ny))

  # GS residuals
  Fx1 = None
  if xHasIy:
    # all-nl: residual = Iy-Iy(previous)
    assert Iy1 is not None
    residualGS = resscalGS * (Iy1.flatten() - Iy.flatten())
  elif rHasFx:
    # all-nl-Fx: residual = Fx-Fx(previous)
    assert Iy1 is not None
    new_Iy1 = Iy1.copy()
    if new_Iy1.ndim == 2:
      new_Iy1 = new_Iy1[None]
    new_Ie = Ie.copy()
    if new_Ie.ndim == 1:
      new_Ie = new_Ie[None]
    Fx1 = meqfx.meqFx(L=L, IyD=new_Iy1, Ie=new_Ie, Fb_ext=(Fbe, Iyie))
    residualGS = resscalGS * (Fx1.flatten() - Fx.flatten())
  else:
    #  Newton-GS: residual = [Fb-Fb(previous),Iy-Iy(previous)]
    # NOTE: Code not tested

    residualGS_true = L.rhsf * (Iy1.flatten() - Iy.flatten())
    residualGS_false = (
        Fx.flatten()[~L.lxy] - meqfbp.meqfbp(Iy1, L, L.P.ilackner) - Fbe
    )
    residualGS = jnp.where(L.lxy, residualGS_true, residualGS_false)
    residualGS = resscalGS * residualGS

  # Evaluate meqagcon residuals and all their derivatives
  # Copied from meqagcon_test.py
  argnums_to_diff = (0, 1, 6, 7, 9, 10, 2, 3, 4, 5)
  fun = functools.partial(meqagcon.meqagcon, L, LX, agconc, L.nD)

  args = (F0, F1, rA, dr2FA, dz2FA, drzFA, ag, Fx, Opy, TpDg, ITpDg)

  grad_fn = jax.jacfwd(fun, argnums=argnums_to_diff, has_aux=True)
  residualag, dresdCo = fun(*args)
  jacs, _ = grad_fn(*args)
  jacs_reshaped = []
  for j in jacs:
    j = j.reshape((len(agconc), -1))
    jacs_reshaped.append(j)
  (
      dres_dF0,
      dres_dF1,
      dresdag,
      dres_dFx,
      dres_dTpDg,
      dres_dITpDg,
      dres_drA,
      dres_ddr2FA,
      dres_ddz2FA,
      dres_ddrzFA,
  ) = jacs_reshaped

  dresdFx = None
  if dojacstat and not picardJac:
    # Derivatives of flux hessian at magnetic axis
    if L.P.icsint:
      _, _, drAdFx, ddz2FAdFx, ddr2FAdFx, ddrzFAdFx = asxycs.jac(
          Fx=Fx,
          zs=zA,
          rs=rA,
          dz2Fs=dz2FA,
          dr2Fs=dr2FA,
          drzFs=drzFA,
          Mr=L.Mr,
          Mz=L.Mz,
          rk=L.taur,
          zk=L.tauz,
          idrx=L.idrx,
          idzx=L.idzx,
          dmax=2,
      )
    else:
      assert L.G is not None
      _, _, drAdFx, ddz2FAdFx, ddr2FAdFx, ddrzFAdFx = asxy.jac(
          Fxy=Fx,
          x=L.G.zx,
          y=L.G.rx,
          Dx=L.dzx,
          Dy=L.drx,
          IDx=L.idzx,
          IDy=L.idrx,
          ixs=ixI[:nA].astype(jnp.int32),
      )

    # Group Fx derivatives
    dresdFx = dres_dFx
    # F0
    dresdFx += dres_dF0 @ dF0dFx.T
    # F1
    dresdFx += dres_dF1 @ dF1dFx.T
    # TpDg
    assert L.lxy is not None
    if dTygdFy is not None and dTygdF0 is not None and dTygdF1 is not None:
      # TODO(adedieu): modify bfct1_jac to have inputs transposed
      _, _, _, dTpDgdFx = bfct.bfct1_jac(
          Opy=Opy.T,
          dYygdFy=dTygdFy.T[..., None],
          dYygdF0=dTygdF0.T,
          dYygdF1=dTygdF1.T,
          dF0dFx=dF0dFx.T,
          dF1dFx=dF1dFx.T,
          lxy=L.lxy.T,
          nB=nB,
          nD=L.nD,
          ng=L.ng,
          nx=L.nx,
          ny=L.ny,
      )
      dresdFx += dres_dTpDg @ dTpDgdFx
    # TpDg
    if dITygdF0 is not None and dITygdF1 is not None:
      assert Tyg is not None
      _, _, _, dITpDgdFx = bfct.bfct1_jac(
          Opy=Opy.T,
          dYygdFy=Tyg.T[..., None],
          dYygdF0=dITygdF0.T,
          dYygdF1=dITygdF1.T,
          dF0dFx=dF0dFx.T,
          dF1dFx=dF1dFx.T,
          lxy=L.lxy.T,
          nB=nB,
          nD=L.nD,
          ng=L.ng,
          nx=L.nx,
          ny=L.ny,
      )
      dresdFx += dres_dITpDg @ dITpDgdFx

    def _mask_nan_and_multiply(A, B):
      """A@B where B can have NaN values."""
      column_mask = jnp.any(jnp.isfinite(B), axis=1)
      if column_mask.shape[0] == 0:
        return 0.0
      A_masked = jnp.where(column_mask[None, :], A, 0.0)
      B_masked = jnp.where(column_mask[:, None], B, 0.0)
      return A_masked @ B_masked

    if drAdFx.shape[0] > 0:
      dresdFx += _mask_nan_and_multiply(dres_drA, drAdFx)

    if ddr2FAdFx.shape[0] > 0:
      dresdFx += _mask_nan_and_multiply(dres_ddr2FA, ddr2FAdFx)
      dresdFx += _mask_nan_and_multiply(dres_ddrzFA, ddrzFAdFx)
      dresdFx += _mask_nan_and_multiply(dres_ddz2FA, ddz2FAdFx)

  # CDE constraint residuals
  residualcde = None
  dcdedIni = None
  dcdedRp = None
  dcdedFx = None
  dcdedF0 = None
  dcdedF1 = None
  dcdedIy = None
  dcdedIe = None
  dcdedag = None
  dcdedFxdot = None
  dcdedF0dot = None
  dcdedF1dot = None
  dcdedIydot = None
  dcdedIedot = None
  dcdedagdot = None
  if dojac:
    # NOTE: code not tested
    (
        residualcde,
        dcdedFx,
        dcdedag,
        dcdedIy,
        dcdedF0,
        dcdedF1,
        dcdedIe,
        dcdedIni,
        dcdedRp,
        dcdedFxdot,
        dcdedagdot,
        dcdedIydot,
        dcdedF0dot,
        dcdedF1dot,
        dcdedIedot,
    ) = meqcde.meqcde(
        L=L,
        LX=LX,
        LY=LYp,
        cdeconc=cdeconc,
        Fx=Fx,
        ag=ag,
        Iy=Iy,
        F0=F0,
        F1=F1,
        Ie=Ie,
        Opy=Opy,
    )

    if L.np > 0:
      residualcde = resscalp * residualcde
  else:
    residualcde = jnp.array(meqcde.meqcde(
        L=L,
        LX=LX,
        LY=LYp,
        cdeconc=cdeconc,
        Fx=Fx,
        ag=ag,
        Iy=Iy,
        F0=F0,
        F1=F1,
        Ie=Ie,
        Opy=Opy,
    ))

  # Circuit equation residuals
  residualcirc = None
  if L.isEvolutive:
    assert L.G is not None
    assert LYp.Iy is not None
    Ve = jnp.zeros(L.ne)
    Ve = Ve.at[: L.G.na].set(LX.Va)
    Fedot = idt * (
        (Ie - jnp.concatenate([LYp.Ia, LYp.Iu])) @ L.Mee
        + (Iy.flatten() - LYp.Iy.flatten()) @ L.Mey
    )
    # Residual for conductor equation
    residualcirc = Fedot + L.Re * Ie - Ve
    residualcirc = resscalc * residualcirc

  # Collect residuals
  res = []
  for this_res in [residualGS, residualag, residualcde, residualcirc]:
    if this_res is not None:
      res.append(this_res.flatten())
  res = jnp.concatenate(res)

  ###### Assemble Jx jacobian ######
  if opts.dojacx and not assign_LXIy:
    assert Jx is not None

    # GS residuals
    if xHasIy:  # 'all-nl'
      assert dIydIy is not None
      # Subtract identity matrix to get jacobian of residual
      dIydIy -= jnp.eye(L.ny)

      Jx = Jx.at[L.ind.ixGS[:, None] - 1, L.ind.irGS[None, :] - 1].set(dIydIy)

      Jx = Jx.at[
          L.ind.ixg[:, None] - 1,
          L.ind.irGS[None, :] - 1,
      ].set(xscalg[:, None] * resscalGS[None] * Tyg)

      if L.isEvolutive:
        Jx = Jx.at[ixe[None, :], L.ind.irGS[:, None] - 1].set(
            (xscale[:, None] * resscalGS[None] * dIydIe).T
        )

    elif rHasFx:  # 'all-nl-Fx'
      # Compute dFxdFx from dIydFx
      assert dIypdFx is not None

      dFxdFx = dIypdFx @ L.Mxy

      dFxdag = Tyg @ L.Mxy

      # Subtract identity matrix to get jacobian of residual
      dFxdFx -= jnp.eye(L.nx)

      Jx = Jx.at[L.ind.ixGS[:, None] - 1, L.ind.irGS[None, :] - 1].set(dFxdFx)

      Jx = Jx.at[L.ind.ixg[:, None] - 1, L.ind.irGS[None, :] - 1].set(
          xscalg[:, None] * resscalGS[None] * dFxdag
      )
    else:  # Newton-GS
      # NOTE: code not tested

      assert dIypdFx is not None
      dIy_dFx = dIypdFx

      # compute reference boundary jacobians with Mby (for Rb = Fb - Fb_ref)
      # keep the Iy Jacobians in sparse format although Mby is dense
      dFb_dFx = meqfbp.meqfbp(dIy_dFx, L, L.P.ilackner)
      dFb_dag = meqfbp.meqfbp(Tyg, L, L.P.ilackner)

      # this is the fastest way to apply the rhs factor to the Iy Jacobian
      drhs_Iy_dFx = dIy_dFx @ jnp.diag(L.rhsf)

      # inner part of dRx_dFx is given by dlst contribution and
      # rhs_factor * dIy_dFx contribution
      dRx_dFx = drhs_Iy_dFx @ L.Txy  - dFb_dFx @ L.Txb
      dRx_dag = (L.rhsf * Tyg) @ L.Txy  - dFb_dag @ L.Txb

      Jx = Jx.at[L.ind.ixGS[None, :]-1, L.ind.irGS[:, None]].add(
          xscalg[:, None] * resscalGS[None] * dRx_dFx
      )
      Jx = Jx.at[L.ind.ixg[None, :] - 1, L.ind.irGS[:, None]].set(
          xscalg[:, None] * resscalGS[None] * dRx_dag
      )

    # ag residual
    if xHasIy:
      # Assemble full residual jacobian w.r.t. Iy
      Jx = Jx.at[L.ind.ixGS[:, None] - 1, L.ind.irC[None, :] - 1].set(
          xscalGS[:, None] * (L.Mxy @ dresdFx.T)
      )
      if L.isEvolutive:
        # Assemble full residual jacobian w.r.t. Ie
        Jx = Jx.at[ixe[:, None], L.ind.irC[None, :] - 1].set(
            xscale[:, None] * (L.Mxe @ dresdFx.T)
        )
    else:
      Jx = Jx.at[L.ind.ixGS[:, None] - 1, L.ind.irC[None, :] - 1].set(
          xscalGS[:, None] * dresdFx.T
      )

    # Residuals are already rescaled
    Jx = Jx.at[L.ind.ixg[:, None] - 1, L.ind.irC[None, :] - 1].set(
        xscalg[:, None] * dresdag.T
    )

    # CDE residual
    if L.np > 0:
      if dcdedFx.shape[0] > 1:
        raise NotImplementedError('dcdedFx.shape[0] > 1 is not tested.')

      # Group Fx derivatives
      dcdedFx_cde = dcdedFx.ravel() + (
          dF0dFx @ dcdedF0 + dF1dFx @ dcdedF1).squeeze()
      if xHasIy:
        drescdedGS = dcdedIy.ravel() + L.Mxy @ dcdedFx_cde
        drescdedIe = dcdedIe.ravel() + L.Mxe @ dcdedFx_cde
      else:
        drescdedGS = dcdedFx_cde - L.dlst @ (dcdedIy / L.rhsf)
        drescdedIe = dcdedIe - L.Tye @ dcdedIy
      drescdedag = dcdedag
      Jx = Jx.at[L.ind.ixGS - 1, L.ind.irp - 1].set(
          xscalGS * (resscalp * drescdedGS)  # do we need to transpose?
      )
      Jx = Jx.at[L.ind.ixg - 1, L.ind.ixg - 1].set(
          ((resscalp * drescdedag) * xscalg).squeeze()
      )
      if L.isEvolutive:
        Jx = Jx.at[ixe, L.ind.irp - 1].set(
            (resscalp * drescdedIe) * xscale
        )

    # Circuit equations
    if L.isEvolutive:
      irDe = ire - (L.nN - L.nrD)
      Jx = Jx.at[:, ire].add(idt * L.Jxdot[:, irDe])

  ###### Ju ######
  if opts.dojacu and not assign_LXIy:
    assert Ju is not None
    # Only derivatives of non-linear terms need be updated
    if xHasIy and not L.isEvolutive:
      # GS residuals
      assert dIydIe is not None
      # indices already substracted in iue
      Ju = Ju.at[iue[:, None], L.ind.irGS[None, :] - 1].set(
          xscale * (dIydIe * resscalGS[None])
      )

      # ag residuals
      assert dresdFx is not None
      Ju = Ju.at[iue[:, None], L.ind.irC[None, :] - 1].set(
          xscale * (L.Mxe @ dresdFx.T)
      )

    Ju = Ju.at[L.ind.iuC[:, None] - 1, L.ind.irC[None, :] - 1].set(
        jnp.diag(dresdCo)
    )

    # CDE residual
    if L.np > 0 and L.isEvolutive:
      # NOTE: code not tested

      assert dcdedIni is not None
      Ju = Ju.at[
          L.ind.iuni[:, None] - 1,
          L.ind.irp[None, :] - 1,
      ].set(resscalp[:, None] * dcdedIni)

      assert dcdedRp is not None
      Ju = Ju.at[L.ind.iurp[:, None] - 1, L.ind.irp[None, :] - 1].set(
          resscalp[:, None] * dcdedRp
      )

  # Jxdot
  if opts.dojacxdot and not assign_LXIy and L.isEvolutive:
    # NOTE: code not tested

    assert Jxdot is not None
    if L.np > 0:
      dcdedFxdot_cde = dcdedFxdot + dcdedF0dot @ dF0dFx + dcdedF1dot @ dF1dFx
      if xHasIy:
        drescdedGSdot = dcdedIydot + L.Mxy @ dcdedFxdot_cde
        drescdedIedot = dcdedIedot + L.Mxe @ dcdedFxdot_cde
      else:
        drescdedGSdot = dcdedFxdot_cde - (dcdedIydot / L.rhsf) @ L.dlst
        drescdedIedot = dcdedIedot - L.Tye @ dcdedIydot
      drescdedagdot = dcdedagdot
      irDp = L.ind.irp - (L.nN - L.nrD)
      Jxdot = Jxdot.at[L.ind.ixGS[:, None] - 1, irDp[None, :] - 1].set(
          xscalGS * (resscalp[None] * drescdedGSdot)
      )
      Jxdot = Jxdot.at[L.ind.ixg[:, None], irDp[None, :]].set(
          xscalg * (resscalp[None] * drescdedagdot)
      )
      Jxdot = Jxdot.at[ixe[:, None], irDp[None, :]].set(
          xscale * (resscalp[None] * drescdedIedot)
      )

  # Jacobian function
  if opts.dojacF:
    raise ValueError('fgeFJac has not been ported to JAX.')

  ###### rowmask ######
  if assign_LXIy:
    if xHasIy or rHasFx:  # % all-nl, all-nl-Fx
      rowmask = rowmask.at[L.ind.irGS - 1].set(-1)
    # [resC,resp] = ag./ag0
    rowmask = rowmask.at[L.ind.irC - 1].set(1)
    rowmask = rowmask.at[L.ind.irp - 1].set(1)
  else:
    # GS equation
    if xHasIy:  # algoNL=all-nl
      # Identify grid points where there cannot be any Iy
      init_ymask = Opy.flatten() == 0

      def main_fun(iD, this_ymask):
        this_ymask = jax.lax.cond(
            jnp.any(L.TDg[iD]),
            lambda: this_ymask,
            lambda: jnp.logical_or(this_ymask, (Opy.flatten() == iD + 1)),
        )
        return this_ymask

      ymask = jax.lax.fori_loop(0, L.nD, main_fun, init_ymask)

      rowmask = rowmask.at[L.ind.irGS - 1].set(-ymask.astype(float))
    # Direct constraint for ag(i) which is also the i-th constraint
    # (not necessarily the case with CDEs)
    mask_ag = [
        c.fun_name == 'ag' and c.index == i + 1 for i, c in enumerate(agconc)
    ]
    rowmask = rowmask.at[L.ind.irC - 1].set(jnp.array(mask_ag))

  if L.isEvolutive:
    # If dt==0, circuit equations are diagonal Re*Ie - Ve = 0
    rowmask_new = rowmask.at[ire].set(resscalc * L.Re * xscale)
    rowmask = jax.lax.cond(dt == 0, lambda: rowmask_new, lambda: rowmask)

  ###### LY ######
  if opts.dopost or opts.doplot or opts.dodisp:
    resy = jnp.linalg.norm(Iy1 - Iy, 2)
    if residualag is not None:
      resC = jnp.linalg.norm(residualag, 2)
    else:
      resC = 0.0
    if residualcde is not None:
      resp = jnp.linalg.norm(residualcde, 2)
    else:
      resp = 0.0
    if residualcirc is not None:
      rese = jnp.linalg.norm(residualcirc, 2)
    else:
      rese = 0.0
    if not rHasFx:
      assert Iy1 is not None
      new_Iy1 = Iy1.copy()
      if new_Iy1.ndim == 2:
        new_Iy1 = new_Iy1[None]
      new_Ie = Ie.copy()
      if new_Ie.ndim == 1:
        new_Ie = new_Ie[None]
      Fx1 = meqfx.meqFx(L=L, IyD=new_Iy1, Ie=new_Ie, Fb_ext=(Fbe, Iyie))
    resFx = jnp.linalg.norm(Fx1 - Fx, 2)

    LY = _fgepost(
        L=L,
        LX=LX,
        LYp=LYp,
        ag=ag,
        Fx=Fx,
        FA=FA,
        FB=FB,
        rA=rA,
        zA=zA,
        dr2FA=dr2FA,
        dz2FA=dz2FA,
        drzFA=drzFA,
        rB=rB,
        zB=zB,
        lB=lB,
        lX=lX,
        rX=rX,
        zX=zX,
        FX=FX,
        dr2FX=dr2FX,
        dz2FX=dz2FX,
        drzFX=drzFX,
        rBt=rBt,
        Ia=Ia,
        Iu=Iu,
        Iy=Iy,
        Opy=Opy,
        F0=F0,
        F1=F1,
        resy=resy,
        resC=resC,
        resp=resp,
        rese=rese,
        resFx=resFx,
    )
  else:
    LY = types.OutputData(shot=LX.shot, t=LX.t, Opy=Opy)

  # Value and Jacobian of NL measurements
  if L.code == 'liu':
    raise ValueError(
        'liuNLmeas has not been ported to JAX. There is some work to do here.'
    )
  return res, LY, Jx, Ju, Jxdot, rowmask
