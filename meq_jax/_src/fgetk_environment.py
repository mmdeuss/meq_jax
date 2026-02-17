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

"""Top level time stepper for FGE environment."""

from collections.abc import Mapping
import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
from meq_jax._src import fgetk_implicit
from meq_jax._src import meq_utils
from meq_jax._src import types


# Stop codes for fgetk_environment
STOP_NONE = 0
STOP_CONVERGENCE = 1
STOP_ALARM = 2
STOP_RESIDUAL = 3


def fgetk_environment(
    state: types.StateData,
    voltages: chex.Array,
    static: types.StaticData,
    lx: types.InputData,
    num_steps: int,
    agconc: list[types.ConcData] | None = None,
    cdeconc: list[types.ConcData] | None = None,
    fge_variation: Mapping[str, chex.Array] | None = None,
) -> tuple[types.OutputData, int, types.StateData, dict[str, Any]]:
  """Combines Matlab fgetk_environment with callmethod='step' and fget."""

  assert state.LYt is not None
  ly_return = state.LYt
  stop = STOP_NONE
  solverinfo = {'isconverged': False}

  fge_variation = fge_variation or {}

  voltages = jnp.asarray(voltages, dtype=jnp.float64)

  for _ in range(num_steps):
    # ly_return is previous LYt
    ly_return = state.LYt

    state.it += 1
    lxt = dataclasses.replace(lx, t=state.LYt.t + state.dt, Va=voltages)
    lxt = dataclasses.replace(lxt, **fge_variation)

    psstate, vact = meq_utils.meqps(static.G, state.PSstate, state.dt, voltages)
    state = dataclasses.replace(state, PSstate=psstate)
    lxt = dataclasses.replace(lxt, Va=vact)

    xnl_prev = state.xnl
    xnlt, lyt, solverinfo = fgetk_implicit.fgetk_implicit(
        xnlp=state.xnl,
        xnlpdot=state.xnldot,
        l=static,
        lxt=lxt,
        lyp=state.LYt,
        prec=state.Prec,
        agconc=agconc,
        cdeconc=cdeconc,
    )

    xnldot = (xnlt - xnl_prev) / state.dt
    Um = (lyt.Bm - state.LYt.Bm) / state.dt  # pylint: disable=invalid-name
    Uf = (lyt.Ff - state.LYt.Ff) / state.dt  # pylint: disable=invalid-name
    lyt = dataclasses.replace(lyt, Um=Um, Uf=Uf)

    tstate, parel, tarel, iarel, alarm = meq_utils.meqlim(
        static, lyt, state.dt, state.Tstate
    )
    lyt = dataclasses.replace(lyt, Parel=parel, Tarel=tarel, Iarel=iarel)

    nnoc = jnp.where(solverinfo['isconverged'], 0, state.nnoc + 1)

    state = dataclasses.replace(
        state,
        it=state.it + 1,
        xnl=xnlt,
        xnldot=xnldot,
        LYt=lyt,
        Tstate=tstate,
        nnoc=nnoc,
    )

    assert state.nnoc is not None
    assert static.P is not None
    assert static.P.nnoc is not None

    # TODO(pfau): Return if stop is triggered, even after jit.
    stop = jax.lax.cond(
        state.nnoc > static.P.nnoc,
        lambda: STOP_CONVERGENCE,
        lambda: STOP_NONE)
    stop = jax.lax.cond(
        alarm,
        lambda: STOP_ALARM,
        lambda: stop)
    stop = jax.lax.cond(
        state.LYt.res > 1e2,
        lambda: STOP_RESIDUAL,
        lambda: stop)

  # Needed for type checking.
  assert isinstance(ly_return, types.OutputData)
  return ly_return, stop, state, solverinfo

