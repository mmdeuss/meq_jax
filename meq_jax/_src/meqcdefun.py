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

"""meqcdefun.py: jax implementation of the meqcdefun function."""

from typing import Callable

import jax
import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import types

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
  grad_fn = jax.value_and_grad(core_fn, argnums=tuple(range(3, 17)))

  def wrapped_fn(
      L: types.StaticData,
      LX: types.InputData,
      LY: types.OutputData,
      Fx: jt.Float[jt.Array, 'nr2+2 nz2+2'],
      ag: jt.Float[jt.Array, ''],
      Iy: jt.Float[jt.Array, 'nr2 nz2'],
      F0: jt.Float[jt.Array, ''],
      F1: jt.Float[jt.Array, ''],
      Ie: jt.Float[jt.Array, 'ne'],
      Ini: jt.Float[jt.Array, ''],
      Rp: jt.Float[jt.Array, ''],
      Opy: jt.Float[jt.Array, 'nr2 nz2'],
      idt: float,
      iD: int,
  ) -> tuple[jt.Float[jt.Array, ''], ...]:
    """Computes the residual and its gradients."""
    # Call the value_and_grad function with arguments in the expected order
    (
        res,
        (
            dFx,
            dag,
            dIy,
            dF0,
            dF1,
            dIe,
            dIni,
            dRp,
            dFxdot,
            dagdot,
            dIydot,
            dF0dot,
            dF1dot,
            dIedot,
        ),
    ) = grad_fn(
        L,
        LX,
        LY,
        Fx,
        ag,
        Iy,
        F0,
        F1,
        Ie,
        Ini,
        Rp,
        Fx * idt,
        ag * idt,
        Iy * idt,
        F0 * idt,
        F1 * idt,
        Ie * idt,
        Opy,
        idt,
        iD,
    )

    # NOTE: we need to add the contributions of time derivatives to the
    # original gradients (i.e. they are total derivatives).
    return (
        res,
        dFx + dFxdot * idt,
        dag + dagdot * idt,
        dIy + dIydot * idt,
        dF0 + dF0dot * idt,
        dF1 + dF1dot * idt,
        dIe + dIedot * idt,
        dIni,
        dRp,
        dFxdot,
        dagdot,
        dIydot,
        dF0dot,
        dF1dot,
        dIedot,
    )

  return wrapped_fn


def ohm_tor_rigid_0d(
    L: types.StaticData,
    LX: types.InputData,
    LY: types.OutputData,
    Fx: jt.Float[jt.Array, 'nr2+2 nz2+2'],
    ag: jt.Float[jt.Array, ''],
    Iy: jt.Float[jt.Array, 'nr2 nz2'],
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    Ie: jt.Float[jt.Array, 'ne'],
    Ini: jt.Float[jt.Array, ''],
    Rp: jt.Float[jt.Array, ''],
    Fxdot: jt.Float[jt.Array, 'nr2+2 nz2+2'],
    agdot: jt.Float[jt.Array, ''],
    Iydot: jt.Float[jt.Array, 'nr2 nz2'],
    F0dot: jt.Float[jt.Array, ''],
    F1dot: jt.Float[jt.Array, ''],
    Iedot: jt.Float[jt.Array, 'ne'],
    Opy: jt.Float[jt.Array, 'nr2 nz2'],
    idt: float,
    iD: int,
) -> jt.Float[jt.Array, '']:
  """CDE of plasma current under rigid plasma assumption."""
  del ag, F0, F1, agdot, F0dot, F1dot, Fxdot
  # iD is 0-based domain index
  P = L.P
  assert P is not None
  assert LY.Iy is not None
  assert LY.Opy is not None

  IyD = Iy * (Opy == iD)
  zIyD = LY.Iy * (LY.Opy == iD)
  IpD = jnp.sum(IyD)
  IpDdot = jnp.sum(Iydot * (Opy == iD))
  IpDz = jnp.sum(zIyD)

  if L.nD != 1:
    # For multi domain plasmas, changes in other plasma domains have to be
    # considered.
    raise NotImplementedError(
        'L.nD > 1 requires meqFx which is not implemented in Python.'
    )
  assert L.Mye is not None
  assert LY.Ia is not None
  assert LY.Iu is not None
  assert P.iLpext is not None
  # For single domain plasmas, only external coils change the flux
  Fye = Ie @ L.Mye
  Fydot = (Iedot - jnp.concatenate([LY.Ia, LY.Iu]) * idt) @ L.Mye

  if P.iLpext:
    assert LX.Lp is not None
    Lp = LX.Lp[iD] if LX.Lp.ndim == 1 else LX.Lp
  else:
    # equivalent to IyD' Myy IyD / IpD^2
    assert L.ny is not None
    FyD = Fx[1:-1, 1:-1].flatten() - Fye
    # Plasma flux contribution is needed for Lp computation
    Lp = IyD.flatten() @ FyD / IpD**2

  return (
      Lp * (IpDdot - IpDz * idt)
      + Rp * (IpD - Ini)
      + jnp.dot(Fydot, IyD.flatten()) / IpD
  )


def ohm_tor_0d(
    L: types.StaticData,
    LX: types.InputData,
    LY: types.OutputData,
    Fx: jt.Float[jt.Array, 'nr2+2 nz2+2'],
    ag: jt.Float[jt.Array, ''],
    Iy: jt.Float[jt.Array, 'nr2 nz2'],
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    Ie: jt.Float[jt.Array, 'ne'],
    Ini: jt.Float[jt.Array, ''],
    Rp: jt.Float[jt.Array, ''],
    Fxdot: jt.Float[jt.Array, 'nr2+2 nz2+2'],
    agdot: jt.Float[jt.Array, ''],
    Iydot: jt.Float[jt.Array, 'nr2 nz2'],
    F0dot: jt.Float[jt.Array, ''],
    F1dot: jt.Float[jt.Array, ''],
    Iedot: jt.Float[jt.Array, 'ne'],
    Opy: jt.Float[jt.Array, 'nr2 nz2'],
    idt: float,
    iD: int,
) -> jt.Float[jt.Array, '']:
  """CDE of plasma current by toroidal projection of Ohm's law."""
  del ag, F0, F1, Ie, L, Fx, agdot, F0dot, F1dot, Iedot, Iydot  # Unused.
  assert LY.Fx is not None
  assert LX.Rp is not None

  iy_d = Iy.flatten() * (Opy.flatten() == iD)

  dfydot = (Fxdot - LY.Fx * idt)[1:-1, 1:-1].flatten()
  ip_d = jnp.sum(iy_d)
  active_domains = Opy.flatten() == iD

  iy_n = jax.lax.cond(
      ip_d == 0,
      lambda: active_domains / jnp.sum(active_domains),
      lambda: iy_d / ip_d,
  )

  return jnp.dot(dfydot, iy_n) + Rp * (ip_d - Ini)


def meqcdefun() -> dict[str, ConstraintFn]:
  """Returns a dictionary of available `cde` constraint functions.

  This is the JAX equivalent of the `meqcdefun.m` script.

  Each returned function is JIT-compiled and computes a constraint residual
  and its gradients with respect to all relevant plasma state variables.
  """
  return {
      'OhmTor_rigid_0D': _make_constraint_fn(ohm_tor_rigid_0d),
      'OhmTor_0D': _make_constraint_fn(ohm_tor_0d),
  }


def meqcdefun_core(fun_name: str) -> Callable[..., jt.Float[jt.Array, '']]:
  """Returns a constraint function for a given core function (without grads)."""
  match fun_name:
    case 'OhmTor_rigid_0D':
      return ohm_tor_rigid_0d
    case 'OhmTor_0D':
      return ohm_tor_0d
    case _:
      raise ValueError(f'Unknown constraint function: {fun_name}')


# pylint: enable=invalid name
