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

"""meqpostq JAX implementation.

This overrides the Matlab implementation in
  third_party/meq/meqpostq.m
"""

import dataclasses

import jax
import jax.numpy as jnp

from meq_jax._src import fsgi
from meq_jax._src import locr
from meq_jax._src import locs
from meq_jax._src import minq
from meq_jax._src import rtciplasma
from meq_jax._src import rtciwall
from meq_jax._src import types


# pylint: disable=invalid-name
def meqpostq(L: types.StaticData, LY: types.OutputData):
  """Updates LY with quantities related to contour integrals and wall gaps."""

  # Plasma flux surfaces, rho-ray origin points and directions returned by rtci
  rq, zq, aq, rO, zO, crq, czq, _ = rtciplasma.rtciplasma(L, LY)

  # Get flux surface quantities and shape parameters
  res_fsgi = fsgi.fsgi(L, LY, rq, zq, aq, rO, zO)
  assert len(res_fsgi) == 31
  (
      Q0Q,
      Q1Q,
      Q2Q,
      Q3Q,
      Q4Q,
      iqQ,
      ItQ,
      LpQ,
      rbQ,
      Q5Q,
      SlQ,
      VQ,
      AQ,
      IpVQ,
      FtPVQ,
      rgeom,
      zgeom,
      aminor,
      epsilon,
      kappa,
      delta,
      deltal,
      deltau,
      rrmax,
      zrmax,
      rrmin,
      zrmin,
      rzmax,
      zzmax,
      rzmin,
      zzmin,
  ) = res_fsgi

  # wall gaps are computed if needed
  aW, FW = rtciwall.rtciwall(L, LY)

  # q95. Note: MATLAB's 1-based L.i95 is converted to 0-based index
  assert iqQ.ndim == 2
  q95 = 1.0 / jnp.dot(L.c95, iqQ[:, L.i95 - 1].T)

  assert L.P is not None
  jtorQ = 2 * jnp.pi * L.P.r0 * (LY.PpQ + (LY.TTpQ * Q2Q) / fsgi.MU0)

  # domain outer surface length and r-barycenter
  lp = LpQ[:, -1]
  rbary = rbQ[:, -1]

  # rhotornorm
  rhotornorm = jnp.sqrt(
      (FtPVQ - FtPVQ[:, 0][:, None])
      / (FtPVQ[:, -1][:, None] - FtPVQ[:, 0][:, None])
  )

  # outboard radius and q surface location
  sq = jnp.sign(LY.Ip) * jnp.sign(LY.rBt)

  def call_minq(zq_, rq_, aq_, iqQ_, iD):
    # Branch 1: iD > LY.nA (mantle domain)
    icr_mantle = jnp.argmin(jnp.abs(zq_[0, :] - LY.zA[0]) - rq_[0, :])

    # Branch 2: else (axis domain)
    icr_axis = jnp.argmax(crq[iD])

    icr = jnp.where(iD + 1 > LY.nA, icr_mantle, icr_axis)

    # Update raQ for the current domain
    current_ra = jnp.concatenate([jnp.array([0]), aq_[:, icr]])
    raQ_ = current_ra / current_ra[-1]
    # In MATLAB, raQ(end) = x/x is exactly 1.0 (IEEE division) for finite
    # nonzero x. XLA on CPU may rewrite the division as a
    # multiply-by-reciprocal, giving e.g. 0.9999999999999999. locQ/locS/locR
    # rely on exact equality with the requested value 1.0 (e.g. L.P.raS(end))
    # to locate the boundary surface, so pin the endpoint to exactly 1.0 in
    # that case. For x = 0/inf/NaN (e.g. unused mantle domains where aq is
    # NaN), keep the IEEE result (NaN) as MATLAB does.
    last = current_ra[-1]
    raQ_ = raQ_.at[-1].set(
        jnp.where(jnp.isfinite(last) & (last != 0.0), 1.0, raQ_[-1])
    )

    # Find qmin for the domain using our helper function
    raqmin_iD, iqmin_val = minq.minQ(raQ_, iqQ_, -sq, n=1)
    qmin_iD = 1.0 / iqmin_val

    return raQ_, raqmin_iD[0], qmin_iD[0]

  raQ, raqmin_vals, qmin_vals = jax.vmap(call_minq, in_axes=(0, 0, 0, 0, 0))(
      zq, rq, aq, iqQ, jnp.arange(L.nD)
  )

  new_ly_fields = {
      'aq': aq,  # 3D
      'aW': aW,  # 2D
      'iqQ': iqQ.squeeze(),
      'jtorQ': jtorQ.squeeze(),
      'q95': q95.squeeze(),
      'lp': lp.squeeze(),
      'rbary': rbary.squeeze(),
      'IpVQ': IpVQ.squeeze(),
      'FtPVQ': FtPVQ.squeeze(),
      'rhotornorm': rhotornorm.squeeze(),
      'rgeom': rgeom.squeeze(),
      'zgeom': zgeom.squeeze(),
      'aminor': aminor.squeeze(),
      'epsilon': epsilon.squeeze(),
      'kappa': kappa.squeeze(),
      'delta': delta.squeeze(),
      'deltal': deltal.squeeze(),
      'deltau': deltau.squeeze(),
      'rrmax': rrmax.squeeze(),
      'zrmax': zrmax.squeeze(),
      'rrmin': rrmin.squeeze(),
      'zrmin': zrmin.squeeze(),
      'rzmax': rzmax.squeeze(),
      'zzmax': zzmax.squeeze(),
      'rzmin': rzmin.squeeze(),
      'zzmin': zzmin.squeeze(),
      'Q0Q': Q0Q.squeeze(),
      'Q1Q': Q1Q.squeeze(),
      'Q2Q': Q2Q.squeeze(),
      'Q3Q': Q3Q.squeeze(),
      'Q4Q': Q4Q.squeeze(),
      'ItQ': ItQ.squeeze(),
      'LpQ': LpQ.squeeze(),
      'rbQ': rbQ.squeeze(),
      'Q5Q': Q5Q.squeeze(),
      'VQ': VQ.squeeze(),
      'AQ': AQ.squeeze(),
      'SlQ': SlQ.squeeze(),
      'rq': rq.squeeze(),
      'zq': zq.squeeze(),
      'FW': FW.squeeze(),
      'raqmin': jnp.atleast_1d(raqmin_vals.squeeze()),
      'qmin': jnp.atleast_1d(qmin_vals.squeeze()),
      'raQ': raQ.squeeze(),
  }

  # raR computation if needed
  if L.nR:

    def call_locR(aQ_, qQ_):
      return locr.locR(aQ=aQ_, qQ=qQ_, qR=sq * L.P.iqR, naR=L.P.naR, NF=L.raN)

    raR = jax.vmap(call_locR, in_axes=(0, 0), out_axes=0)(raQ, iqQ)
    new_ly_fields['raR'] = raR

  # rS and zS computations if needed
  if L.nS:

    def call_locS(aQ_, qQ_, crq_, rO_, czq_, zO_):
      aQ_aug = jnp.concatenate([jnp.zeros((1, L.noq)), aQ_], axis=0)
      aS_ = locs.locS(aQ=aQ_aug, qQ=qQ_, qS=L.P.raS, NF=L.raN)
      rS_ = crq_ * aS_ + rO_
      zS_ = czq_ * aS_ + zO_
      return rS_, zS_

    rS, zS = jax.vmap(call_locS, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)(
        aq, raQ, crq, rO, czq, zO
    )
    new_ly_fields['rS'] = rS
    new_ly_fields['zS'] = zS

  LY = dataclasses.replace(LY, **new_ly_fields)
  return LY
