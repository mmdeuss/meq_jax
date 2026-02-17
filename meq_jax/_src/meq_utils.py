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

"""Python/JAX implementations of MEQ utility functions."""

import chex
import jax.numpy as jnp
from meq_jax._src import types


safe_item = lambda x: x.item() if isinstance(x, jnp.ndarray) else x


def meqps(
    g: types.GeometryData, psstate: chex.Array, dt: float, vcmd: chex.Array
) -> tuple[chex.Array, chex.Array]:
  """General power supply model with voltage saturations and delays."""
  if g.Vadelay is None:
    vadelay = jnp.zeros(g.na)
  else:
    vadelay = g.Vadelay

  idelays = jnp.round(vadelay / (dt + 1e-30) + 1e-10).astype(int)

  # Saturate input voltage
  vsat = jnp.maximum(jnp.minimum(vcmd, g.Vamax), g.Vamin)

  if psstate.shape[1] == 0:
    vact = vsat
    new_psstate = psstate
  else:
    indices = jnp.arange(vsat.shape[0])
    vact = jnp.where(
        idelays == 0,
        vsat,
        psstate[indices, jnp.clip(idelays - 1, 0, psstate.shape[1] - 1)],
    )
    new_psstate = jnp.concatenate(
        [vsat[:, jnp.newaxis], psstate[:, :-1]], axis=1
    )
  return new_psstate, vact


def meqlim(
    l: types.StaticData,
    ly: types.OutputData,
    dt: float,
    tstate: chex.Array,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
  """Checks coil system operational limits."""
  assert l.G is not None
  tstate = _meqtherm(ly.Ia, dt, tstate)

  # Coil current limit
  iarel, ialarm = _meqialim(l.G, ly.Ia)

  # Thermal limit
  tarel = tstate / l.G.Talim
  talarm = tarel > 1

  # Coil combination equation limit
  if hasattr(l.P, 'eqfct') and l.P.eqfct is not None:
    if l.P.eqfct == 'meqetcv':
      parel, palarm = _meqetcv(l.G, ly.Ia)
    else:
      raise NotImplementedError(f'eqfct {l.P.eqfct} not implemented')
  else:
    palarm = jnp.zeros(l.G.na)
    parel = jnp.zeros(l.G.na)

  alarm = jnp.any(talarm) | jnp.any(ialarm) | jnp.any(palarm)

  return tstate, parel, tarel, iarel, alarm


def _meqialim(g: types.GeometryData,
              ia: chex.Array) -> tuple[chex.Array, chex.Array]:
  """Checks current limits."""
  ialarm = (ia > g.Iamax) | (ia < g.Iamin)
  iarel = jnp.maximum(ia / g.Iamax, ia / g.Iamin)
  return iarel, ialarm


def _meqtherm(ia: chex.Array, dt: float, tstate: chex.Array) -> chex.Array:
  """Thermal protection implementation for coil currents."""
  tstate = tstate + dt * ia**2  # time integral of current squared
  return tstate


def _meqetcv(g: types.GeometryData,
             ia: chex.Array) -> tuple[chex.Array, chex.Array]:
  """Implements coil equation constraints for TCV."""
  ieq = ia @ g.Ceq
  # In the original Matlab code, this is 1 if g.Seq is not 1, OR it is Ia if
  # at the index for a coil with "OH_002" in the name. However, only the last
  # (19th) index for TCV is the OH_002 coil, and g.Seq has length 13, so it is
  # not clear how to apply this logic.
  # TODO(pfau): reintroduce the special logic for OH_002.
  ioh2 = g.Seq != 1
  iprot = ieq * ioh2
  ip = iprot >= 0
  parel = jnp.where(ip, iprot / g.Ihig, iprot / g.Ilow)
  palarm = jnp.logical_or(iprot > g.Ihig, iprot < g.Ilow)
  return parel, palarm

