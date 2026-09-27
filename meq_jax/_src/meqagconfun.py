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

"""meqagconfun.py: jax implementation of the meqagconfun function."""

from typing import Callable

import jax
import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import bfct_router
from meq_jax._src import types
from meq_jax._src import vizr

# pylint: disable=invalid-name

ConstraintFn = Callable[..., tuple[jt.Float[jt.Array, ''], ...]]


def _make_constraint_fn(
    core_fn: Callable[..., jt.Float[jt.Array, '']],
) -> ConstraintFn:
  """Wraps a core residual function to create a full JAX constraint function.

  This wrapper uses `jax.value_and_grad` to automatically compute the gradients
  of the residual with respect to its inputs, replacing the manual Jacobian
  calculations in the original MATLAB code.

  Args:
    core_fn: A function that computes the constraint residual. It must have the
      signature `core_fn(L, LX, F0, F1, rA, dr2FA, dz2FA, drzFA, ag, Fx, Opy,
      TpDg, ITpDg, iD, Co)`.

  Returns:
    A full constraint function with the signature required by the optimization
    routines, returning the residual and all its gradients.
  """

  # This makes sure that the order of returned gradients matches the order of
  # the arguments in the original MATLAB code.
  argnums_to_diff = (
      2,  # F0
      3,  # F1
      8,  # ag
      9,  # Fx
      11,  # TpDg
      12,  # ITpDg
      4,  # rA
      5,  # dr2FA
      6,  # dz2FA
      7,  # drzFA
      14,  # Co
  )
  grad_fn = jax.value_and_grad(core_fn, argnums=argnums_to_diff)

  def wrapped_fn(
      L: types.StaticData,
      LX: types.InputData,
      F0: jt.Float[jt.Array, ''],
      F1: jt.Float[jt.Array, ''],
      rA: jt.Float[jt.Array, ''],
      dr2FA: jt.Float[jt.Array, ''],
      dz2FA: jt.Float[jt.Array, ''],
      drzFA: jt.Float[jt.Array, ''],
      ag: jt.Float[jt.Array, 'nP+nT'],
      Fx: jt.Float[jt.Array, 'nr2+2 nz2+2'],
      Opy: jt.Float[jt.Array, 'nr2 nz2'],
      TpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
      ITpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
      iD: int,
      Co: jt.Float[jt.Array, ''],
  ) -> tuple[jt.Float[jt.Array, ''], ...]:
    """Computes the residual and its gradients."""
    # Call the value_and_grad function with arguments in the expected order
    res, grads = grad_fn(
        L, LX, F0, F1, rA, dr2FA, dz2FA, drzFA, ag, Fx, Opy, TpDg, ITpDg, iD, Co
    )

    return (res, *grads)

  return wrapped_fn


# --- Core Residual Functions ---
def _ag_core(
    L: types.StaticData,
    LX: types.InputData,
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    rA: jt.Float[jt.Array, ''],
    dr2FA: jt.Float[jt.Array, ''],
    dz2FA: jt.Float[jt.Array, ''],
    drzFA: jt.Float[jt.Array, ''],
    ag: jt.Float[jt.Array, 'nP+nT'],
    Fx: jt.Float[jt.Array, 'nr2+2 nz2+2'],
    Opy: jt.Float[jt.Array, 'nr2 nz2'],
    TpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    ITpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    iD: int,
    Co: jt.Float[jt.Array, ''],
) -> jt.Float[jt.Array, '']:
  """Directly specify ag = LX.ag."""
  del F0, F1, Fx, ITpDg, rA, dr2FA, dz2FA, drzFA, LX, Opy, TpDg
  ind = L.ind
  assert ind is not None
  return (ag[iD - 1] - Co) / L.xscal[
      ind.ixg[iD - 1] - 1
  ]  # adjust for 0-based indexing


def _bp_core(
    L: types.StaticData,
    LX: types.InputData,
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    rA: jt.Float[jt.Array, ''],
    dr2FA: jt.Float[jt.Array, ''],
    dz2FA: jt.Float[jt.Array, ''],
    drzFA: jt.Float[jt.Array, ''],
    ag: jt.Float[jt.Array, 'nP+nT'],
    Fx: jt.Float[jt.Array, 'nr2+2 nz2+2'],
    Opy: jt.Float[jt.Array, 'nr2 nz2'],
    TpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    ITpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    iD: int,
    Co: jt.Float[jt.Array, ''],
) -> jt.Float[jt.Array, '']:
  """Impose bp = LX.bp (beta poloidal per domain) or total bp using LIUQE."""
  del F0, F1, Fx, rA, dr2FA, dz2FA, drzFA, LX, Opy
  vecT = TpDg if TpDg.ndim == 1 else TpDg[:, iD - 1]
  vecI = ITpDg if ITpDg.ndim == 1 else ITpDg[:, iD - 1]
  P = L.P
  assert P is not None

  LIp02 = L.Ip0**2
  WN0 = 1e-7 * jnp.pi * P.r0 * LIp02
  return ((vecI * L.fPg) @ ag) / WN0 - Co * (vecT @ ag)**2 / LIp02


def _ip_core(
    L: types.StaticData,
    LX: types.InputData,
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    rA: jt.Float[jt.Array, ''],
    dr2FA: jt.Float[jt.Array, ''],
    dz2FA: jt.Float[jt.Array, ''],
    drzFA: jt.Float[jt.Array, ''],
    ag: jt.Float[jt.Array, 'nP+nT'],
    Fx: jt.Float[jt.Array, 'nr2+2 nz2+2'],
    Opy: jt.Float[jt.Array, 'nr2 nz2'],
    TpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    ITpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    iD: int,
    Co: jt.Float[jt.Array, ''],
) -> jt.Float[jt.Array, '']:
  """Core residual calculation for the Ip constraint. Co = target Ip."""
  del F0, F1, Fx, ITpDg, rA, dr2FA, dz2FA, drzFA, LX, Opy
  vec = TpDg if TpDg.ndim == 1 else TpDg[:, iD - 1]
  return vec @ ag / L.Ip0 - Co / L.Ip0


def _li_core(
    L: types.StaticData,
    LX: types.InputData,
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    rA: jt.Float[jt.Array, ''],
    dr2FA: jt.Float[jt.Array, ''],
    dz2FA: jt.Float[jt.Array, ''],
    drzFA: jt.Float[jt.Array, ''],
    ag: jt.Float[jt.Array, 'nP+nT'],
    Fx: jt.Float[jt.Array, 'nr2+2 nz2+2'],
    Opy: jt.Float[jt.Array, 'nr2 nz2'],
    TpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    ITpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    iD: int,
    Co: jt.Float[jt.Array, ''],
) -> jt.Float[jt.Array, '']:
  """Constrain plasma li to LX.li (LIUQE li definition with fixed r0)."""
  del F0, F1, ITpDg, rA, dr2FA, dz2FA, drzFA
  P = L.P
  assert P is not None
  vec = TpDg if TpDg.ndim == 1 else TpDg[:, iD - 1]
  Wp, _, _ = vizr.vizrmex(
      Fx, Opy == iD, L.ry, L.iry, LX.rBt, L.drx, L.dzx)

  LIp02 = L.Ip0 ** 2
  WN0 = 1e-7 *jnp.pi * P.r0 * LIp02

  # Co = li = Wp/WN;
  return Wp / WN0 - Co * (vec @ ag) ** 2 / LIp02


def _bt_core(
    L: types.StaticData,
    LX: types.InputData,
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    rA: jt.Float[jt.Array, ''],
    dr2FA: jt.Float[jt.Array, ''],
    dz2FA: jt.Float[jt.Array, ''],
    drzFA: jt.Float[jt.Array, ''],
    ag: jt.Float[jt.Array, 'nP+nT'],
    Fx: jt.Float[jt.Array, 'nr2+2 nz2+2'],
    Opy: jt.Float[jt.Array, 'nr2 nz2'],
    TpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    ITpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    iD: int,
    Co: jt.Float[jt.Array, ''],
) -> jt.Float[jt.Array, '']:
  """Core residual calculation for the bt constraint. Co = target bt."""
  del F0, F1, TpDg, rA, dr2FA, dz2FA, drzFA
  P = L.P
  assert P is not None
  vec = ITpDg if ITpDg.ndim == 1 else ITpDg[:, iD - 1]

  _, Ft0, _ = vizr.vizrmex(
      Fx, Opy == iD, L.ry, L.iry, LX.rBt, L.drx, L.dzx)
  Wt0 = 2.5e6 * LX.rBt * Ft0

  # 100*mu0/pi/(b0^2 * r0^2 * Sx), with Sx the area of the flux grid.
  scal = 1e2 / (2.5e6 * P.r0 * P.b0 * (P.b0 * P.r0 * (L.nx * L.dsx)))
  return scal * ((vec * L.fPg) @ ag - Co * Wt0)


def _qA_core(
    L: types.StaticData,
    LX: types.InputData,
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    rA: jt.Float[jt.Array, ''],
    dr2FA: jt.Float[jt.Array, ''],
    dz2FA: jt.Float[jt.Array, ''],
    drzFA: jt.Float[jt.Array, ''],
    ag: jt.Float[jt.Array, 'nP+nT'],
    Fx: jt.Float[jt.Array, 'nr2+2 nz2+2'],
    Opy: jt.Float[jt.Array, 'nr2 nz2'],
    TpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    ITpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    iD: int,
    Co: jt.Float[jt.Array, ''],
) -> jt.Float[jt.Array, '']:
  """Core residual calculation for the qA constraint. Co = target qA."""
  del Fx, TpDg, ITpDg, Opy
  P = L.P
  assert P is not None
  gAg, IgAg = bfct_router.bfct5(F0=F0, F1=F1, Bfp=L.bfp)

  if gAg.ndim == 1:
    gAg = gAg[:, None]
    IgAg = IgAg[:, None]

  rATyAg = (
      gAg[:, iD - 1] * rA[iD - 1] * (rA[iD - 1] * rA[iD - 1] * L.fPg + L.fTg)
  )

  gA = jnp.sqrt(dr2FA[iD - 1] * dz2FA[iD - 1] - drzFA[iD - 1] ** 2) / jnp.abs(
      dr2FA[iD - 1] + dz2FA[iD - 1]
  )

  A = (
      2e-7
      * (IgAg[:, iD - 1] * L.fTg - 2 * Co * jnp.pi * LX.rBt * gA * rATyAg)
      / L.dsx
  )
  return (A @ ag + LX.rBt**2) / (P.r0 * P.b0) ** 2


def _wk_core(
    L: types.StaticData,
    LX: types.InputData,
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    rA: jt.Float[jt.Array, ''],
    dr2FA: jt.Float[jt.Array, ''],
    dz2FA: jt.Float[jt.Array, ''],
    drzFA: jt.Float[jt.Array, ''],
    ag: jt.Float[jt.Array, 'nP+nT'],
    Fx: jt.Float[jt.Array, 'nr2+2 nz2+2'],
    Opy: jt.Float[jt.Array, 'nr2 nz2'],
    TpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    ITpDg: jt.Float[jt.Array, 'nP+nT'] | jt.Float[jt.Array, 'batch nP+nT'],
    iD: int,
    Co: jt.Float[jt.Array, ''],
) -> jt.Float[jt.Array, '']:
  """Core residual calculation for the Wk constraint. Co = target Wk."""
  del F0, F1, Fx, TpDg, rA, dr2FA, dz2FA, drzFA, Opy, LX
  vec = ITpDg if ITpDg.ndim == 1 else ITpDg[:, iD - 1]
  P = L.P
  assert P is not None

  WN0 = 1e-7 * jnp.pi * P.r0 * L.Ip0**2
  A = 1.5 * vec * L.fPg
  return (A @ ag - Co) / WN0


def meqagconfun() -> dict[str, ConstraintFn]:
  """Returns a dictionary of available `ag` constraint functions.

  This is the JAX equivalent of the `meqagconfun.m` script.

  Each returned function is JIT-compiled and computes a constraint residual
  and its gradients with respect to all relevant plasma state variables.

  Function Signature:
  ```
  res, dresdF0, ..., dresdCo = constraint_fn(
      L, LX, F0, F1, rA, dr2FA, dz2FA, drzFA,
      ag, Fx, Opy, TpDg, ITpDg, iD, Co
  )
  ```
  where:
    - `iD`: A 0-indexed integer specifying the domain.
    - `Co`: The target value for the constraint (e.g., target Ip, Wk, or qA).
  """
  return {
      'ag': _make_constraint_fn(_ag_core),
      'bp': _make_constraint_fn(_bp_core),
      'bt': _make_constraint_fn(_bt_core),
      'Ip': _make_constraint_fn(_ip_core),
      'li': _make_constraint_fn(_li_core),
      'Wk': _make_constraint_fn(_wk_core),
      'qA': _make_constraint_fn(_qA_core),
  }


def meqagconfun_core(fun_name: str) -> Callable[..., jt.Float[jt.Array, '']]:
  """Returns a constraint function for a given core function (without grads)."""
  match fun_name:
    case 'ag':
      return _ag_core
    case 'bp':
      return _bp_core
    case 'bpD':
      return _bp_core
    case 'bt':
      return _bt_core
    case 'btD':
      return _bt_core
    case 'Ip':
      return _ip_core
    case 'IpD':
      return _ip_core
    case 'li':
      return _li_core
    case 'Wk':
      return _wk_core
    case 'WkD':
      return _wk_core
    case 'qA':
      return _qA_core
    case _:
      raise ValueError(f'Unknown constraint function: {fun_name}')

# pylint: enable=invalid name
