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


def _fgetk_step(
    state: types.StateData,
    voltages: chex.Array,
    static: types.StaticData,
    lx: types.InputData,
    agconc: list[types.ConcData] | None,
    cdeconc: list[types.ConcData] | None,
    fge_variation: Mapping[str, chex.Array],
) -> tuple[types.StateData, jax.Array, dict[str, Any]]:
  """A single simulator step of the FGE environment."""
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

  # The state produced by a step contains weak-dtype arrays wherever Python
  # scalars entered the computation, while `utils.normalize_state` produces
  # strong dtypes. Canonicalize to strong dtypes (a no-op at the XLA level)
  # so step outputs have the same abstract values as normalized inputs: this
  # keeps the lax.scan carry consistent and lets consecutive calls to a
  # jitted fgetk_environment hit the same compilation cache entry.
  def _strongify(x):
    if isinstance(x, (bool, int, float)):
      x = jnp.asarray(x)
    if isinstance(x, jnp.ndarray) and hasattr(x, 'weak_type') and x.weak_type:
      return x.astype(x.dtype)
    return x

  state = jax.tree_util.tree_map(_strongify, state)

  return state, stop, solverinfo


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
  """Combines Matlab fgetk_environment with callmethod='step' and fget.

  The Python loop over simulator steps is expressed with `jax.lax.scan` over
  all steps whose state pytree structure is stable, so the compiled program
  size (and hence compile time) does not grow with `num_steps`. A state
  freshly initialized from Octave typically has a different pytree structure
  than the state produced by a step (see `utils.normalize_state`); in that
  case the first step is unrolled and the remaining steps are scanned.
  """

  assert state.LYt is not None
  ly_return = state.LYt
  stop = STOP_NONE
  solverinfo = {'isconverged': False}

  fge_variation = fge_variation or {}

  voltages = jnp.asarray(voltages, dtype=jnp.float64)

  def step_fn(state):
    return _fgetk_step(
        state, voltages, static, lx, agconc, cdeconc, fge_variation)

  remaining = num_steps
  # Unroll leading steps until the state structure becomes stationary
  # (normally at most one iteration).
  while remaining > 0:
    out_struct = jax.eval_shape(lambda s: step_fn(s)[0], state)
    if (jax.tree_util.tree_structure(out_struct) ==
        jax.tree_util.tree_structure(state)):
      stationary = all(
          a.shape == b.shape and a.dtype == b.dtype
          for a, b in zip(jax.tree_util.tree_leaves(out_struct),
                          jax.tree_util.tree_leaves(state))
          if hasattr(a, 'shape') and hasattr(b, 'shape')
      )
    else:
      stationary = False
    if stationary:
      break
    ly_return = state.LYt
    state, stop, solverinfo = step_fn(state)
    remaining -= 1

  if remaining > 1:
    def scan_body(state, _):
      new_state, stop, solverinfo = step_fn(state)
      return new_state, None

    state, _ = jax.lax.scan(scan_body, state, None, length=remaining - 1)
    remaining = 1

  if remaining == 1:
    # The last (or only) step is unrolled so its LYt/stop/solverinfo are
    # returned directly without stacked scan outputs.
    ly_return = state.LYt
    state, stop, solverinfo = step_fn(state)

  # Needed for type checking.
  assert isinstance(ly_return, types.OutputData)
  return ly_return, stop, state, solverinfo

