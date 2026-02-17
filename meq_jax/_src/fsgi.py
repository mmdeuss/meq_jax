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

"""fsgi JAX implementation.

This overrides the implementation in
  third_party/meq/libmeq/fsgi.c
whose pure Matlab implementation is in
  third_party/meq/fsgi.m
"""

import math

import jax
import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import fsgi_lib
from meq_jax._src import shapmex_lib
from meq_jax._src import types

eps = jnp.finfo(float).eps

MU0 = 4.0 * math.pi * 1e-7


# pylint: disable=invalid-name
def fsgi(
    L: types.StaticData,
    LY: types.OutputData,
    rq: jt.Float[jt.Array, 'nD npq noq'],
    zq: jt.Float[jt.Array, 'nD npq noq'],
    aq: jt.Float[jt.Array, 'nD npq noq'],
    rO: jt.Float[jt.Array, 'nD noq'],
    zO: jt.Float[jt.Array, 'nD noq'],
) -> tuple[jt.Float[jt.Array, 'nD nQ'], ...]:  # 31 outputs
  """Returns flux surfaces averaged quantities and geometric quantities.

  Computes a multitude of flux surface averaged quantities as well as
  geometrical quantities of the plasma. Uses flux surfaces as computed by
  rtci.m. Outputs are the combined outputs of fsgimex, fsg2mex and shapmex.

  Arguments:
    L: standard L structure
    LY: standard LY structure containing converged GS-equilibrium
    rq: r-positions of discreitzed flux surfaces
    zq: z-positions of discreitzed flux surfaces
    aq: radial positions (how far away from origin points) for the discreitzed
      flux surfaces
    rO: double(noq,nD), r-positions of constant-theta ray origins
    zO: double(noq,nD), z-positions of constant-theta ray origins

  Returns: (inside varargout, in this order)
    Q0Q: <1/R>
    Q1Q: -dpsi/dV = -1/int(dl/|Bp|)
    Q2Q: <1/R^2>
    Q3Q: <|grad psi|^2/R^2> = (2pi)^2<Bp^2>
    Q4Q: <|grad psi|^2> = (2pi)^2<(R*Bp)^2>
    iqQ: 1/q
    ItQ: Ip contained inside FS = int(Bp dl)
    LpQ: length of flux surface = int(dl)
    rbQ: R-barycenter = int(R dl) / int(dl)
    Q5Q: <|grad psi|/(2*pi)> = <R*Bp>
    SlQ: 2*pi*int(R dl)
    VQ: Volume inside FS = int(2*pi*R dA)
    AQ: Surface inside contour = int(dA)
    IpVQ: Ip contained inside FS = int(p' + <1/R^2> TT' / mu0 dV)
    FtPVQ: Toroidal flux contained inside FS = int(<1/R^2> T / 2pi dV)
    rgeom: 0.5 * (rmin + rmax)
    zgeom: 0.5 * (zmin + zmax)
    aminor: minor radius = 0.5 * (rmax - rmin)
    epsilon: inverse aspect ratio = aminor./rgeom
    kappa: elongation = (zmax - zmin) / (rmax - rmin)
    delta: triangularity, see shapmexm.m
    deltal: lower triangularity, see shapmexm.m
    deltau: upper triangularity, see shapmexm.m
    rrmax: rmax
    zrmax: z at rmax point
    rrmin: rmin
    zrmin: z at rmin point
    rzmax: r at zmax point
    zzmax: zmax
    rzmin: r at zmin point
    zzmin: zmin
  """
  assert rq.ndim == 3
  assert zq.ndim == 3
  assert aq.ndim == 3

  # nAeff = min([LY.nA, LY.nB, L.nD])
  # Note: because LY.nA and LY.nB are not static, we deviate from the Matlab
  # implementation here and set nDeff = L.nD
  nAeff = L.nD
  is_doublet = L.shot == 82 or L.shot == 88

  num_outputs = 31
  outputs = [jnp.zeros((L.nD, L.nQ)) for _ in range(num_outputs)]

  # Part 1: Fill out output for domains with axes
  # Vectorize fsgiD across domains.
  vmapped_fsgiD = jax.vmap(fsgiD, in_axes=(None, None, 0, 0, 0, 0), out_axes=0)
  # _, rq1, rq2 = rq.shape
  results_tuple = vmapped_fsgiD(
      L,
      LY,
      rq[:nAeff],
      zq[:nAeff],
      aq[:nAeff],
      # jax.lax.dynamic_slice(rq, (0, 0, 0), (nAeff, rq1, rq2)),
      # jax.lax.dynamic_slice(zq, (0, 0, 0), (nAeff, rq1, rq2)),
      # jax.lax.dynamic_slice(aq, (0, 0, 0), (nAeff, rq1, rq2)),
      jnp.arange(1, nAeff + 1),
  )

  # Update the outputs dictionary. Since out_axes=1, each result has shape
  # (L.nQ, nAeff), which fits directly into the output arrays.
  new_outputs = []
  for idx, output in enumerate(outputs):
    new_output = output.at[:nAeff].set(results_tuple[idx])
    new_outputs.append(new_output)
  outputs = new_outputs

  # Part 2: Handle the special case for a doublet plasma
  if is_doublet:
    # Get separatrix quantities from computation of inner two lobes
    ItQ = outputs[6]
    LpQ = outputs[7]
    SlQ = outputs[10]
    IpVQ = outputs[13]
    FtPVQ = outputs[14]

    ItS = jnp.sum(ItQ[:2, -1])
    LpS = jnp.sum(LpQ[:2, -1])
    SlS = jnp.sum(SlQ[:2, -1])
    IpVS = jnp.sum(IpVQ[:2, -1])
    FtPVS = jnp.sum(FtPVQ[:2, -1])

    # Call the special function for the mantle domain
    mantle_results_tuple = fsgi_mantle(
        L=L,
        LY=LY,
        rq=rq[2],
        zq=zq[2],
        rS=rO[2],
        zS=zO[2],
        ItS=ItS,
        LpS=LpS,
        SlS=SlS,
        IpVS=IpVS,
        FtPVS=FtPVS,
    )

    # Update the 3rd row of each output array.
    new_outputs = []
    for idx, output in enumerate(outputs):
      new_output = output.at[2].set(mantle_results_tuple[idx])
      new_outputs.append(new_output)
    outputs = new_outputs

  return tuple(outputs)


def fsgiD(
    L: types.StaticData,
    LY: types.OutputData,
    rqD: jt.Float[jt.Array, 'npq noq'],
    zqD: jt.Float[jt.Array, 'npq noq'],
    aqD: jt.Float[jt.Array, 'npq noq'],
    iD: jt.Float[jt.Array, ''],
):
  """Returns flux surface averages and geometric quantities on single axis domain.

  Computes a multitude of flux surface averaged quantities as well as
  geometrical quantities for a single plasma domain with magnetic axis.

  Arguments:
    L: standard L structure
    LY: standard LY structure containing converged GS-equilibrium
    rqD: r-positions of discreitzed flux surfaces of iD domain
    zqD: z-positions of discreitzed flux surfaces of iD domain
    aqD: radial positions for the discreitzed flux surfaces of iD domain
    iD: index of axis domain to run processing on
  """
  qaq = jnp.square(aqD)
  # da^2/qdq
  M1q = L.M1q @ qaq
  # ((da^2/2do)^2/a^2+a^2)/(da^2/qdq) = ((da/do)^2+a^2)/(da^2/qdq)
  M2q = (jnp.square(qaq @ L.M2q) / qaq + qaq) / M1q

  # some info from LY
  idx = iD - 1

  rA = LY.rA[idx]
  zA = LY.zA[idx]
  FA = LY.FA[idx]
  FB = LY.FB[idx]
  dr2FA = LY.dr2FA[idx]
  dz2FA = LY.dz2FA[idx]
  detFA = jnp.sign(LY.IpD[idx]) * jnp.sqrt(
      dr2FA * dz2FA - jnp.square(LY.drzFA[idx])
  )
  lX = LY.lX[idx]
  rB = LY.rB[idx]
  zB = LY.zB[idx]
  iTQ = LY.iTQ[idx]

  # Contour integrals
  res_fsgimex = fsgi_lib.fsgimex(
      M1q=M1q,
      M2q=M2q,
      rq=rqD,
      irq=1.0 / rqD,
      rA=rA,
      FA=FA,
      FB=FB,
      BA=detFA,
      lX=lX,
      rX=rB,
      iTQ=iTQ,
      idoq=1 / L.doq,
  )
  Q2Q = res_fsgimex[2]

  # FS volume/area
  res_fsg2mex = fsg2mex(aqD, rA, L.crq, L.doq)
  VQ, _ = res_fsg2mex

  # Plasma current inside flux suface computation by volume integral
  IpVQ = cumtrapz(VQ, LY.PpQ[idx] + LY.TTpQ[idx] * Q2Q / MU0)

  # Toroidal flux computation by volume integral
  FtPVQ = cumtrapz(VQ, (LY.TQ[idx] * Q2Q) / (2 * math.pi))

  # Shape profiles
  res_shapmex = shapmex_lib.shap(
      rq=rqD, zq=zqD, rB=rB, zB=zB, rA=rA, zA=zA, dr2fa=dr2FA, dz2fa=dz2FA
  )
  return res_fsgimex + res_fsg2mex + (IpVQ, FtPVQ) + res_shapmex


def fsgi_mantle(
    L: types.StaticData,
    LY: types.OutputData,
    rq: jt.Float[jt.Array, 'npq noq'],
    zq: jt.Float[jt.Array, 'npq noq'],
    rS: jt.Float[jt.Array, 'noq'],
    zS: jt.Float[jt.Array, 'noq'],
    ItS: jt.Float[jt.Array, ''],
    LpS: jt.Float[jt.Array, ''],
    SlS: jt.Float[jt.Array, ''],
    IpVS: jt.Float[jt.Array, ''],
    FtPVS: jt.Float[jt.Array, ''],
):
  """Returns flux surfaces and geometric quantities for a mantle domain.

  As the mantle domain does not have a single axis from which
  constant-theta rays originate, some flux surface integral computations
  have to be adapted.

  Arguments:
    L: standard L structure
    LY: standard LY structure containing converged GS-equilibrium
    rq: r-positions of discreitzed flux surfaces of mantle
    zq: z-positions of discreitzed flux surfaces of mantle
    rS: r-positions of separatrix points
    zS: z-positions of separatrix points
    ItS: Ip contained inside separatrix from fsgi computation
    LpS: Length of separatrix
    SlS: R-barycenter of separatrix
    IpVS: Ip contained inside separatrix from volume integral computation
    FtPVS: Toroidal flux inside separatrix from volume integral computation
  """
  # Wrapper of shapmex adding separatrix as innermost surface
  res_shapmantle = shapmantle(rq, zq, rS, zS, LY.rB[2], LY.zB[2])

  # Volume and surface integrals using separatrix as innermost surface
  rS_rq = jnp.concatenate([rS[None], rq], axis=0)
  zS_zq = jnp.concatenate([zS[None], zq], axis=0)
  res_fsg2mantle = fsg2mantle(rS_rq, zS_zq)
  VQ, _ = res_fsg2mantle

  # Flux surface averages
  res_fsavgmantle = fsavgmantle(
      rq=rq,
      zq=zq,
      rS=rS,
      zS=zS,
      M3q=L.M3q,
      M2q=L.M2q,
      doq=L.doq,
      FAB=LY.F1[2] - LY.F0[2],
      rX=LY.rX,
      lXm=LY.lX[2],
      rBm=LY.rB[2],
      iTQ=LY.iTQ[2, 1:],
      ItS=ItS,
      LpS=LpS,
      SlS=SlS,
  )
  Q2Q = res_fsavgmantle[2]

  # Plasma current inside flux suface computation by volume integral
  IpVQ = IpVS + cumtrapz(VQ, LY.PpQ[2] + LY.TTpQ[2] * Q2Q / MU0)

  # Toroidal flux computation by volume integral
  FtPVQ = FtPVS + cumtrapz(VQ, (LY.TQ[2] * Q2Q) / (2. * math.pi))

  return res_fsavgmantle + res_fsg2mantle + (IpVQ, FtPVQ) + res_shapmantle


def cumtrapz(
    X: jt.Float[jt.Array, 'n'],
    Y: jt.Float[jt.Array, 'n'],
):
  """JAX cumulative trapezoidal rule, as in MATLAB's cumtrapz.

  Args:
      X: array-like, optional. The coordinates corresponding to Y values.
      Y: The array-like data to integrate.

  Returns:
      An array of the cumulative integral of Y, with the same shape as Y.
      The first element along the integration axis is always 0.
  """
  # Average of adjacent values
  Y_avg = (Y[1:] + Y[:-1]) / 2.0

  # Deltas
  dX = X[1:] - X[:-1]

  # Multiply by d (which is either a scalar or a broadcastable array)
  trapezoids = Y_avg * dX

  # # Perform cumulative sum along the integration axis
  cumulative_integral = jnp.cumsum(trapezoids)

  # Concatenate the zeros and the cumulative integral
  Q = jnp.concatenate([jnp.array([0.0]), cumulative_integral])
  return Q


def shapmantle(
    rq: jt.Float[jt.Array, 'npq noq'],
    zq: jt.Float[jt.Array, 'npq noq'],
    rS: jt.Float[jt.Array, 'noq'],
    zS: jt.Float[jt.Array, 'noq'],
    rB: jt.Float[jt.Array, ''],
    zB: jt.Float[jt.Array, ''],
):
  """Computes geometric quantities for mantle domain."""
  rS_rq = jnp.concatenate([rS[None], rq], axis=0)
  zS_zq = jnp.concatenate([zS[None], zq], axis=0)
  res = shapmex_lib.shap(
      rS_rq, zS_zq, rB, zB, jnp.nan, jnp.nan, jnp.nan, jnp.nan
  )
  cropped_res = jax.tree.map(lambda x: x[1:], res)
  return cropped_res


def fsavgmantle(
    rq: jt.Float[jt.Array, 'npq noq'],
    zq: jt.Float[jt.Array, 'npq noq'],
    rS: jt.Float[jt.Array, 'noq'],
    zS: jt.Float[jt.Array, 'noq'],
    M3q: jt.Float[jt.Array, 'npq npq'],
    M2q: jt.Float[jt.Array, 'noq noq'],
    doq: jt.Float[jt.Array, ''],
    FAB: jt.Float[jt.Array, ''],
    rX: jt.Float[jt.Array, 'nX'],
    lXm: jt.Bool[jt.Array, ''],
    rBm: jt.Float[jt.Array, ''],
    iTQ: jt.Float[jt.Array, 'nQ'],
    ItS: jt.Float[jt.Array, ''],
    LpS: jt.Float[jt.Array, ''],
    SlS: jt.Float[jt.Array, ''],
):
  """Returns flux surface averages for the mantle domain.

  Flux surface averages are computed without assumptions on magnetic
  geometry. The surface integrals are discretized as integrals over the
  edge elements and quantities are summed up using the midpoint rule

  Arguments:
    rq: r-positions of discreitzed flux surfaces
    zq: z-positions of discreitzed flux surfaces
    rS: r-positions of separatrix
    zS: z-positions of separatrix
    M3q: radial spline derivative matrix
    M2q: azimuthal spline derivative matrix
    doq: 2 * pi / noq
    FAB: F1(mantle) - F0(mantle)
    rX: r-positions of X points (the first should be the central one)
    lXm: flag indicating a diverted mantle
    rBm: radial coordinate of point defining mantle LCFS (limiter- or X-point)
    iTQ: 1 / TQ (see meqpost)
    ItS: Ip contained inside separatrix
    LpS: Length of separatrix
    SlS: R-barycenter of separatrix
  """
  dzdpsi = 0.5 * (M3q.dot(zq - zS)) / FAB
  drdpsi = 0.5 * (M3q.dot(rq - rS)) / FAB

  drdtheta = 2 * doq * (rq.dot(M2q))
  dzdtheta = 2 * doq * (zq.dot(M2q))

  # grad_psi perp grad_theta
  # => |grad_psi| * dl = det([grad_psi, grad_theta]) = det(J^-1)
  dl = jnp.sqrt(drdtheta**2 + dzdtheta**2)
  detJ = drdpsi * dzdtheta - dzdpsi * drdtheta
  dpsi = -dl / detJ
  idpsidl = -detJ

  # useful things
  irq = 1.0 / rq
  irX = 1.0 / rX[0]

  # more useful things
  dpsidV = 1.0 / (2 * math.pi * jnp.sum(rq * idpsidl, axis=-1))
  int_iR_idpsi_dl = jnp.sum(irq * idpsidl, axis=-1)
  int_iR_dpsi_dl = jnp.sum(irq * dpsi * dl, axis=-1)
  int_R_dl = jnp.sum(rq * dl, axis=-1)

  # flux surface averages
  Q1Q = jnp.concatenate((jnp.array([0.0]), -dpsidV))
  Q0Q = jnp.concatenate(
      (jnp.array([irX]), 2 * math.pi * jnp.sum(idpsidl, axis=-1) * dpsidV)
  )
  Q2Q = jnp.concatenate(
      (jnp.array([irX * irX]), 2 * math.pi * int_iR_idpsi_dl * dpsidV)
  )
  Q3Q = jnp.concatenate(
      (jnp.array([0.0]), 2 * math.pi * int_iR_dpsi_dl * dpsidV)
  )
  Q4Q = jnp.concatenate((
      jnp.array([0.0]),
      2 * math.pi * jnp.sum(rq * dpsi * dl, axis=-1) * dpsidV,
  ))
  Q5Q = jnp.concatenate((jnp.array([0.0]), int_R_dl * jnp.abs(dpsidV)))

  iqQ = jnp.concatenate((jnp.array([0.0]), -iTQ.ravel() / int_iR_idpsi_dl))
  ItQ = jnp.concatenate(
      (jnp.array([ItS]), -int_iR_dpsi_dl / (2 * math.pi * MU0))
  )
  LpQ = jnp.concatenate((jnp.array([LpS]), jnp.sum(dl, axis=-1)))
  SlQ = jnp.concatenate((jnp.array([SlS]), 2 * math.pi * int_R_dl))

  # Element-wise division is fine as both are column vectors
  rbQ = SlQ / (2 * math.pi * LpQ)

  state = (Q0Q, Q1Q, Q2Q, Q3Q, Q4Q, iqQ, ItQ, LpQ, rbQ, Q5Q, SlQ)

  def diverted_case(state):
    Q0Q, Q1Q, Q2Q, Q3Q, Q4Q, iqQ, ItQ, LpQ, rbQ, Q5Q, SlQ = state
    irX2 = 1.0 / rBm
    Q0Q = Q0Q.at[-1].set(irX2)
    Q1Q = Q1Q.at[-1].set(0.0)
    Q2Q = Q2Q.at[-1].set(irX2 * irX2)
    Q3Q = Q3Q.at[-1].set(0.0)
    Q4Q = Q4Q.at[-1].set(0.0)
    iqQ = iqQ.at[-1].set(0.0)
    Q5Q = Q5Q.at[-1].set(0.0)
    return (Q0Q, Q1Q, Q2Q, Q3Q, Q4Q, iqQ, ItQ, LpQ, rbQ, Q5Q, SlQ)

  state = jax.lax.cond(
      lXm, lambda _: diverted_case(state), lambda _: state, None
  )
  return state


def fsg2mantle(
    rq: jt.Float[jt.Array, 'npq noq'],
    zq: jt.Float[jt.Array, 'npq noq'],
) -> tuple[jt.Float[jt.Array, 'npq'], jt.Float[jt.Array, 'npq']]:
  """Computes volume and area enclosed by flux surfaces.

  To avoid assumptions on flux surface geometry, these computations are
  carried out fully by using greens formula

  Arguments:
    rq: r-positions of discreitzed flux surfaces
    zq: z-positions of discreitzed flux surfaces

  Returns:
    VQ: Volume inside FS = int(2*pi*R dA)
    AQ: Surface inside contour = int(dA)
  """
  # Roll the array along the second axis to get the (i+1)th element
  rq_rolled = jnp.roll(rq, -1, axis=1)
  zq_rolled = jnp.roll(zq, -1, axis=1)

  # Calculate dr, rc, and zc
  dr = rq_rolled - rq
  rc = 0.5 * (rq_rolled + rq)
  zc = 0.5 * (zq_rolled + zq)

  AQ = jnp.sum(zc * dr, axis=-1)

  # int(2piR dA) = intC(2pi R Z dr)
  VQ = jnp.sum(2 * math.pi * zc * rc * dr, axis=-1)
  return VQ, AQ


def fsg2mex(
    aq: jt.Float[jt.Array, 'npq noq'],
    rA: jt.Float[jt.Array, ''],
    crq: jt.Float[jt.Array, 'noq'],
    doq: jt.Float[jt.Array, ''],
):
  """Returns the flux-surface volume VQ and cross-section area AQ."""
  C1 = 0.5 * doq
  C2 = 2 * math.pi * doq

  qaq = jnp.square(aq)
  s1q = jnp.sum(qaq * (rA / 2 + aq * crq / 3), axis=-1)
  s2q = jnp.sum(qaq, axis=-1)

  VQ = jnp.concatenate([jnp.array([0.0]), C2 * s1q])
  AQ = jnp.concatenate([jnp.array([0.0]), C1 * s2q])
  return VQ, AQ


# pylint: enable=invalid-name
