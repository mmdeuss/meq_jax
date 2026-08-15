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

"""JAX implementation of meqcde, for computing CDE residuals."""

import jax
import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import meqcdefun
from meq_jax._src import types


# pylint: disable=invalid-name
def meqcde(
    L: types.StaticData,
    LX: types.InputData,
    LY: types.OutputData,
    cdeconc: list[types.ConcData],
    Fx: jt.Float[jt.Array, 'nr2+2 nz2+2'],
    ag: jt.Float[jt.Array, 'ng'],
    Iy: jt.Float[jt.Array, 'nr2 nz2'],
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
    Ie: jt.Float[jt.Array, 'ne'],
    Opy: jt.Float[jt.Array, 'nr2 nz2'],
) -> jt.Float[jt.Array, 'nC']:
  """Computes residuals for CDE constraints in MEQ.

  TODO(joeljennings): Add support for inactive CDE domains.

  Args:
      L: StaticData object containing static data for the equilibrium.
      LX: InputData object containing input data for the equilibrium.
      LY: OutputData object containing output data from the previous time step.
      cdeconc: List of CdeConcData objects containing the constraint function
        names and domain ids.
      Fx: Flux field on X-grid.
      ag: ag values.
      Iy: Plasma current field on Y-grid.
      F0: Axis flux value per domain.
      F1: Separatrix flux value per domain.
      Ie: Active coil and vessel component currents.
      Opy: Plasma domain segmentation map on Y-grid.

  Returns:
      res: The residual of each of the constraints.
  """
  assert L.nD is not None
  assert L.nD == 1, 'Multi-domain CDEs are not supported in Python.'

  dt = LX.t - LY.t
  idt = jax.lax.cond(dt == 0, lambda: 0.0, lambda: 1.0 / dt)
  if cdeconc:
    assert LX.Rp is not None
    assert LX.Ini is not None
    results = [
        meqcdefun.meqcdefun()[cde_data.fun_name](
            L,
            LX,
            LY,
            Fx,
            ag,
            Iy,
            F0,
            F1,
            Ie,
            LX.Ini,
            LX.Rp,
            Opy,
            idt,
            cde_data.domain_id,
        )
        for cde_data in cdeconc
    ]
    return [jnp.stack(col) for col in zip(*results)]
  else:
    res = jnp.zeros((L.np, 1))
    dresdFx = jnp.zeros((L.np, L.nx))
    dresdag = jnp.zeros((L.np, L.ng))
    dresdIy = jnp.zeros((L.np, L.ny))
    dresdF0 = jnp.zeros((L.np, L.nD))
    dresdF1 = jnp.zeros((L.np, L.nD))
    dresdIe = jnp.zeros((L.np, L.ne))
    dresdIni = jnp.zeros((L.np, L.nD))
    dresdRp = jnp.zeros((L.np, L.nD))
    dresdFxdot = jnp.zeros((L.np, L.nx))
    dresdagdot = jnp.zeros((L.np, L.ng))
    dresdIydot = jnp.zeros((L.np, L.ny))
    dresdF0dot = jnp.zeros((L.np, L.nD))
    dresdF1dot = jnp.zeros((L.np, L.nD))
    dresdIedot = jnp.zeros((L.np, L.ne))
    return (
        res,
        dresdFx,
        dresdag,
        dresdIy,
        dresdF0,
        dresdF1,
        dresdIe,
        dresdIni,
        dresdRp,
        dresdFxdot,
        dresdagdot,
        dresdIydot,
        dresdF0dot,
        dresdF1dot,
        dresdIedot,
    )


# pylint: enable=invalid-name
