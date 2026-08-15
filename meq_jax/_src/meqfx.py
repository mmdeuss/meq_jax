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

"""meqFx JAX implementation.

This overrides the Matlab implementation in
  third_party/meq/meqFx.m
"""

import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import gszr
from meq_jax._src import meqfbp
from meq_jax._src import types


# pylint: disable=invalid-name
def meqFx(
    L: types.StaticData,
    IyD: jt.Float[jt.Array, 'nD nrx-2 nzx-2'],
    Ie: jt.Float[jt.Array, 'nb ne'] | None = None,
    Fb_ext: jt.Float[jt.Array, '2'] | None = None,
    dz: jt.Float[jt.Array, 'nD'] | None = None,
    rst: bool | None = None,
    Jh: jt.Float[jt.Array, 'nrx nzx'] | None = None,
) -> jt.Float[jt.Array, 'nD nrx nzx']:
  """Returns flux (and optionally its derivatives) for Iy, and optionally Ie.

  Returns the solution of the poisson equation for given current density
  parametrization.
  The L structure contains MEQ ancillary data, IYD contains the breakdown of
  the plasma current per plasma domain , IE is the current in the external
  conductors, assumed zero if not passed.
  FB_EXT is an optional argument with precomputed contributions from
  external currents.
  The last 3 arguments are optional and are typically used only in LIU. DZ
  is the vector of shifts of IY with respect to FX for each domain, if RST
  is true then JH contains the current in each FE for the LIU initial
  guess.

  The parameter L.P.gsxe determins how the computation is done
   - if gsxe=1, then the boundary flux Fb is determined using meqfbp(Iy) and
   Fbe*Ie then Iy is modified to include filaments from external currents
   within the computational domain and the poisson equation is solved using
   gszr with this modified Iy and the previously determined Fb (so the Iys
   going into meqfbp and gszr are different)
   - if gsxe=2, then the filaments from external currents within the
   computational domain are excluded from the computation of Fb and their
   contribution to the poisson solution is determined solely by analytical
   green functions
   - if gsxe=3, then the full solution is determined by analytical green
   functions
  """
  assert L.P is not None
  assert L.P.gsxe is not None

  assert IyD.ndim == 3
  assert IyD.shape[0] == 1, 'Only tested with 1D IYD'
  nIyD = IyD.shape[0]

  if Ie is None:
    Ie = jnp.zeros((1, L.ne))

  assert Ie.ndim == 2
  assert Ie.shape[0] == 1, 'MeqFx has only been tested with 1D Ie'
  nb = Ie.shape[0]

  if nb > 1 and nIyD != nb:
    raise ValueError(f'nIyD={nIyD} and nb={nb} are incompatible.')

  # if not (hasIe or hasIy):
  #  return jnp.zeros((nD, L.nrx, L.nzx))

  # Pre-initialize for cases where they aren't calculated
  Fbe = None
  Iyie = None

  # External current contributions
  if Ie is not None:
    if L.P.gsxe == 1 or L.P.gsxe == 2:
      if Fb_ext is None:
        Fbe = Ie @ L.Mbe
        Iyie = Ie @ L.Tye
        Iyie = Iyie.reshape((nb, L.nry, L.nzy))
      else:
        Fbe = Fb_ext[0]
        Iyie = Fb_ext[1]

  if dz is None or rst or not jnp.any(dz):
    if dz is None:
      rst = False
    if rst and dz is not None and jnp.any(dz):
      raise ValueError('If rst is true, all dzs should be 0')
    dz = jnp.array([False])
    dzMbe = jnp.zeros((L.ne,))

    if nb == 1 and nIyD > 1:
      IyD = IyD.at[0].set(IyD.sum(axis=0))
  else:
    dzMbe = L.dzMbe
    # Effective number of domains for jitability
    assert dz.shape[0] == nIyD
    if nb > 1 and jnp.any(dz):
      raise ValueError('If dz is not None, then nb should be 1')

  # Boundary flux
  if nb == 1:
    Iy = IyD[0]
    Ie = Ie[0]
    if L.P.gsxe != 3:
      Iyie = Iyie[0]
  else:
    Iy = IyD

  Fb = jnp.zeros((L.nx - L.ny,))
  if L.P.gsxe in [1, 2]:  # Fb not needed for gsxe=3
    # From FE
    if rst:
      Fb = Jh @ L.Mbh
    else:
      # Boundary current contribution from plasma - Lackner's trick!
      # TODO(adedieu): meqfbp seems slow
      Fb = meqfbp.meqfbp(Iy, L, L.P.ilackner)

  # Full solution calculation depends on the mode
  if L.P.gsxe in [1, 2]:
    if Ie is not None:
      # Add effect of external currents and dz
      assert Fbe is not None
      Fb = Fb + Fbe.reshape(Fb.shape)

      # The mapping from e to y does not take dz into account
      Fb = Fb - dz[0] * Ie @ dzMbe

      # Add external currents on computational grid
      Iy = Iy + Iyie

    if L.gszr_iy_op is not None:
      Fx = gszr.apply_gszr_operator(
          boundary_conditions=Fb,
          filament_currents=Iy,
          bc_op=L.gszr_bc_op,
          iy_op=L.gszr_iy_op,
          dz=L.idzx * dz[0],
      )
    else:
      Fx = gszr.gszrjax(
          boundary_conditions=Fb,
          filament_currents=Iy,
          cx=L.cx,
          cq=L.cq,
          cr=L.cr,
          cs=L.cs,
          ci=L.ci,
          co=L.co,
          dz=L.idzx * dz[0],
      )
  elif L.P.gsxe == 3:
    # Direct calculation from Green's functions
    assert L.Mxy is not None
    Fx = L.Mxy.T @ IyD.ravel()
    Fx = Fx.reshape(L.nrx, L.nzx)

    # Finite difference for dF/dz
    Fx_up = jnp.roll(Fx, shift=-1, axis=-1)
    Fx_down = jnp.roll(Fx, shift=1, axis=-1)
    weights = jnp.full((L.nzx,), 0.5)
    weights = weights.at[0].set(1.0)
    weights = weights.at[-1].set(1.0)
    dFxdzD = (Fx_up - Fx_down) * weights[None] * L.idzx

    # Optional addition of dF/dz
    Fx = Fx + dz[0] * dFxdzD

    Fx = Fx + (Ie @ L.Mxe).reshape(L.nrx, L.nzx)
  else:
    raise ValueError(f'gsxe={L.P.gsxe} is invalid')

  # Note: the code below has not been tested in unit tests
  for iD in range(1, nIyD):
    Iy = IyD[iD]
    if L.P.gsxe < 3:
      Fb = meqfbp.meqfbp(Iy, L, L.P.ilackner)
      if L.gszr_iy_op is not None:
        Fx = gszr.apply_gszr_operator(
            boundary_conditions=Fb,
            filament_currents=Iy,
            bc_op=L.gszr_bc_op,
            iy_op=L.gszr_iy_op,
            dz=L.idzx * dz[iD],
        )
      else:
        Fx = gszr.gszrjax(
            boundary_conditions=Fb,
            filament_currents=Iy,
            cx=L.cx,
            cq=L.cq,
            cr=L.cr,
            cs=L.cs,
            ci=L.ci,
            co=L.co,
            dz=L.idzx * dz[iD],
        )
    else:
      # New flux from this IyD
      FxD = (Iy @ L.Mxy).reshape(L.nrx, L.nzx)

      # Finite difference for dF/dz
      FxD_up = jnp.roll(FxD, shift=-1, axis=-1)
      FxD_down = jnp.roll(FxD, shift=1, axis=-1)
      weights = jnp.full((L.nzx,), 0.5)
      weights = weights.at[0].set(1.0)
      weights = weights.at[-1].set(1.0)
      dFxdzD = (FxD_up - FxD_down) * weights[None] * L.idzx
      # Optional addition of dF/dz
      FxD = FxD + dFxdzD * dz[iD]

      Fx = Fx + FxD
  return Fx
