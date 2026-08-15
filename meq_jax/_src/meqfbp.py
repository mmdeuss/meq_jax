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

"""JAX implementation of the MEQFBP MATLAB function."""

import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import gszr
from meq_jax._src import nfdb
from meq_jax._src import types


def meqfbp(
    filament_currents: jt.Float[jt.Array, 'nr2 nz2'],
    static_data: types.StaticData,
    ilackner: int,
) -> jt.Float[jt.Array, '2*nz2+2*nr2+4']:
  """JAX implementation of the MEQFBP MATLAB function.

  Args:
    filament_currents: Filament currents.
    static_data: Static data struct.
    ilackner: The ilackner parameter.

  Returns:
    [2 * (nz2 + nr2 + 2)]
  """
  match ilackner:
    case 1:
      nr2, nz2 = filament_currents.shape
      if static_data.gszr_iy_op is not None:
        grid = gszr.apply_gszr_operator(
            jnp.zeros(2 * (nr2 + nz2 + 2)),
            filament_currents,
            static_data.gszr_bc_op,
            static_data.gszr_iy_op,
            0,
        )
      else:
        grid = gszr.gszrjax(
            jnp.zeros(2 * (nr2 + nz2 + 2)),
            filament_currents,
            static_data.cx,
            static_data.cq,
            static_data.cr,
            static_data.cs,
            static_data.ci,
            static_data.co,
            0,
        )
      return nfdb.nfdb(grid) @ static_data.Tbc
    case _:
      return filament_currents.flatten() @ static_data.Mby
