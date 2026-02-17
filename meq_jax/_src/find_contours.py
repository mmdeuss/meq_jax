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

"""find_contours jax implementation.

This replaces the Matlab implementation in
  third_party/meq/find_contours.m
"""

import jax
import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import rtci
from meq_jax._src import rtcics
from meq_jax._src import types


# pylint: disable=invalid-name


def find_contours(
    L: types.StaticData,
    LY: types.OutputData,
    F: jt.Float[jt.Array, 'npq'],
    aq: jt.Float[jt.Array, 'npq noq'],
    rO: jt.Float[jt.Array, 'n'],
    zO: jt.Float[jt.Array, 'n'],
    crq: jt.Float[jt.Array, 'noq'],
    czq: jt.Float[jt.Array, 'noq'],
    Fo: jt.Float[jt.Array, 'n'],
    Opo: jt.Float[jt.Array, 'n'],
):
  """Flux contour finding algorithm based on Newton iterations.

  Finds contours either by rtcics.py or rtcimex.py. Contours are computated as
  points where, origninating in ro,zo and going along cr,cz, the values of
  LY.Fx are equal to the supplied F

  Arguments:
    L: standard L structure
    LY: standard LY structure containing converged GS-equilibrium
    F: requested flux values of flux surfaces
    aq: radial positions init
    rO: r-positions of constant-theta ray origins
    zO: z-positions of constant-theta ray origins
    crq: r-directions of constant-theta rays
    czq: z-directions of constant-theta rays
    Fo: Flux values at origin points
    Opo: domain identifier of origin points

  Returns:
    rq: r-positions of discretized flux surfaces
    zq: z-positions of discretized flux surfaces
    aq: radial positions (distance to rO, zO)
  """
  assert L.P is not None
  assert L.G is not None

  if L.P.icsint:
    aq, _ = rtcics.rtcics(
        a=aq,
        Fx=LY.Fx,
        F=F,
        Opy=LY.Opy,
        ro=rO,
        zo=zO,
        Opo=Opo,
        cr=crq,
        cz=czq,
        L=L,
    )

  else:
    ero = (rO - L.G.rx[0]) * L.idrx
    ezo = (zO - L.G.zx[0]) * L.idzx
    cdr = crq * L.idrx
    cdz = czq * L.idzx

    def cond_fun(state):
      k, _, daq = state
      assert L.P is not None
      is_converged = daq < L.drx * L.P.tolq
      continue_iterating = k < L.P.iterq

      # For the non-itert case, stop if converged.
      # For the itert case (L.liurtemu=True), always run the full loop.
      return jax.lax.cond(
          L.liurtemu,
          lambda: continue_iterating,
          lambda: continue_iterating & ~is_converged,
      )

    def body_fun(state):
      k, aq, _ = state
      # Call the core function to get the updated values
      aq_new, daq_new = rtci.rtcimex(
          a=aq,
          er0=ero,
          ez0=ezo,
          Fx=LY.Fx,
          F=F,
          c=cdr,
          s=cdz,
          Opy=LY.Opy,
          Fo=Fo,
          Opo=Opo,
          dap=L.drx,
      )
      # Return the updated state for the next iteration
      return (k + 1, aq_new, daq_new)

    initial_state = (0, aq, jnp.inf)
    _, aq, _ = jax.lax.while_loop(cond_fun, body_fun, initial_state)

  # compute r, z coordinates at the end
  rq = rO + crq * aq
  zq = zO + czq * aq
  return rq, zq, aq


# pylint: enable=invalid-name
