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

"""asxycs JAX implementation.

IMPORTANT: This implementation differs from the Octave/Matlab implementation
in that it expects inputs with fixed shape (i.e. with infinities).
"""

import functools

import jax
import jax.numpy as jnp
import jaxopt
import jaxtyping as jt

from meq_jax._src import cs2devalmex


# pylint: disable=invalid-name


# This residual function is the core of the jaxopt implementation.
# It takes the parameters to be optimized (coords) and returns the
# vector that should be zeroed out (the gradient).
@jax.jit
def _residual_fun(coords, rk, zk, c) -> jnp.ndarray:
  """Computes the gradient of the flux, which serves as the residual."""
  rs, zs = coords
  # We only need the first derivative (gradient) for the residual
  _, gr, gz, *_ = cs2devalmex.cs2devalmex(
      # Note: dmax is ignored.
      Tx=rk, Ty=zk, C=c, xp=rs, yp=zs, dmax=None
  )
  # The residual is the flattened gradient vector
  return jnp.concatenate([gr, gz])


@functools.partial(jax.jit, static_argnames=('itercs', 'tolcs'))
def asxycs(
    Fx: jt.Float[jt.Array, 'ny nx'],
    zA: jt.Float[jt.Array, 'ny nx'],  # with infinities
    rA: jt.Float[jt.Array, 'ny nx'],  # with infinities
    zX: jt.Float[jt.Array, 'ny nx'],  # with infinities
    rX: jt.Float[jt.Array, 'ny nx'],  # with infinities
    Mr: jt.Float[jt.Array, 'nx+2 nx'],
    Mz: jt.Float[jt.Array, 'ny+2 ny'],
    rk: jt.Float[jt.Array, 'ny+6'],
    zk: jt.Float[jt.Array, 'nx+6'],
    itercs: int,
    tolcs: float,
) -> tuple[
    jt.Float[jt.Array, 'ny nx'],  # zA
    jt.Float[jt.Array, 'ny nx'],  # rA
    jt.Float[jt.Array, 'ny nx'],  # FA
    jt.Float[jt.Array, 'ny nx'],  # dz2FA
    jt.Float[jt.Array, 'ny nx'],  # dr2FA
    jt.Float[jt.Array, 'ny nx'],  # drzFA
    jt.Float[jt.Array, 'ny nx'],  # zX
    jt.Float[jt.Array, 'ny nx'],  # rX
    jt.Float[jt.Array, 'ny nx'],  # FX
    jt.Float[jt.Array, 'ny nx'],  # dz2FX
    jt.Float[jt.Array, 'ny nx'],  # dr2FX
    jt.Float[jt.Array, 'ny nx'],  # drzFX
]:
  """Refines points using jaxopt.GaussNewton."""
  c = jnp.einsum('xr,yz,xy->zr', Mr, Mz, Fx)

  # Combine initial coordinates into a PyTree (a tuple of arrays)
  # which jaxopt will optimize.
  init_coords = (jnp.concatenate([rA, rX]), jnp.concatenate([zA, zX]))

  gn = jaxopt.GaussNewton(residual_fun=_residual_fun, maxiter=itercs, tol=tolcs)
  # Static arguments (knots, coefficients) are passed directly to run()
  final_coords, _ = gn.run(init_coords, rk, zk, c)

  rs_final, zs_final = final_coords

  # Final Evaluation
  Fs, _, _, dr2Fs, drzFs, dz2Fs, *_ = cs2devalmex.cs2devalmex(
      # Note: dmax is ignored.
      Tx=rk, Ty=zk, C=c, xp=rs_final, yp=zs_final, dmax=None
  )

  # Split the results back into O-point and X-point variables
  rA_final, rX_final = jnp.split(rs_final, 2)
  zA_final, zX_final = jnp.split(zs_final, 2)
  FA, FX = jnp.split(Fs, 2)
  dr2FA, dr2FX = jnp.split(dr2Fs, 2)
  drzFA, drzFX = jnp.split(drzFs, 2)
  dz2FA, dz2FX = jnp.split(dz2Fs, 2)

  return (
      zA_final,
      rA_final,
      FA.reshape(Fx.shape),
      dz2FA.reshape(Fx.shape),
      dr2FA.reshape(Fx.shape),
      drzFA.reshape(Fx.shape),
      zX_final,
      rX_final,
      FX.reshape(Fx.shape),
      dz2FX.reshape(Fx.shape),
      dr2FX.reshape(Fx.shape),
      drzFX.reshape(Fx.shape),
  )


def jac(Fx: jt.Float[jt.Array, 'ny nx'],
        zs: jt.Float[jt.Array, 'ns'],
        rs: jt.Float[jt.Array, 'ns'],
        dz2Fs: jt.Float[jt.Array, 'ns'],
        dr2Fs: jt.Float[jt.Array, 'ns'],
        drzFs: jt.Float[jt.Array, 'ns'],
        Mr: jt.Float[jt.Array, 'nx+2 nx'],
        Mz: jt.Float[jt.Array, 'ny+2 ny'],
        rk: jt.Float[jt.Array, 'ny+6'],
        zk: jt.Float[jt.Array, 'nx+6'],
        idrx: float,
        idzx: float,
        dmax: int) -> tuple[
            jt.Float[jt.Array, 'ns ny*nx'],  # dFs
            jt.Float[jt.Array, 'ns ny*nx'],  # dzs
            jt.Float[jt.Array, 'ns ny*nx'],  # drs
            jt.Float[jt.Array, 'ns ny*nx'],  # ddz2Fs
            jt.Float[jt.Array, 'ns ny*nx'],  # ddr2Fs
            jt.Float[jt.Array, 'ns ny*nx'],  # ddrzFs
        ]:
  """JAX implementation of asxycsJac.m."""
  del dmax
  ns = zs.size
  if ns == 0:
    return 6 * (jnp.zeros((0, Fx.size)),)
  if zs.ndim == 0:
    zs = jnp.array([zs])
    rs = jnp.array([rs])

  nrx, nzx = Fx.shape

  irs = jnp.minimum(
      jnp.maximum(jnp.floor((rs-rk[0])*idrx)+1, 4), nrx+3).astype(jnp.int64)
  izs = jnp.minimum(
      jnp.maximum(jnp.floor((zs-zk[0])*idzx)+1, 4), nzx+3).astype(jnp.int64)

  drxs = (rs-rk[irs-1]) * idrx
  dzxs = (zs-zk[izs-1]) * idzx

  def _fun(ir, iz, drx, dzx):
    Mr_slice = jax.lax.dynamic_slice(Mr, (0, ir-4), (nrx, 4))
    Mz_slice = jax.lax.dynamic_slice(Mz, (0, iz-4), (nzx, 4))

    A = cs2devalmex.csint(drx, 0, 1)
    Ar = Mr_slice @ A
    B = cs2devalmex.csint(dzx, 0, 1)
    Bz = Mz_slice @ B

    dF = jnp.outer(Ar, Bz)

    dA = cs2devalmex.csint(drx, 1, idrx)
    dAr = Mr_slice @ dA
    dB = cs2devalmex.csint(dzx, 1, idzx)
    dBz = Mz_slice @ dB

    dzdF = jnp.outer(Ar, dBz)
    drdF = jnp.outer(dAr, Bz)

    d2A = cs2devalmex.csint(drx, 2, idrx**2)
    d2Ar = Mr_slice @ d2A
    d2B = cs2devalmex.csint(dzx, 2, idzx**2)
    d2Bz = Mz_slice @ d2B

    dr2dF = jnp.outer(d2Ar, Bz)
    drzdF = jnp.outer(dAr, dBz)
    dz2dF = jnp.outer(Ar, d2Bz)

    return dF, dzdF, drdF, dz2dF, dr2dF, drzdF

  dFs, dzdFs, drdFs, dz2dFs, dr2dFs, drzdFs = jax.vmap(_fun)(
      irs, izs, drxs, dzxs)

  detHs = dr2Fs * dz2Fs - drzFs**2
  imdetHs = jnp.where(detHs == 0, 0, -1 / detHs)
  drs = imdetHs[:, None, None] * (
      dz2Fs[:, None, None] * drdFs - drzFs[:, None, None] * dzdFs)
  dzs = imdetHs[:, None, None] * (
      -drzFs[:, None, None] * drdFs + dr2Fs[:, None, None] * dzdFs)

  c = Mr.T @ (Fx @ Mz)
  _, _, _, _, _, _, d3rrrFs, d3rrzFs, d3rzzFs, d3zzzFs, *_ = (
      cs2devalmex.cs2devalmex(rk, zk, c.T, rs, zs, None))
  ddr2Fs = dr2dFs + d3rrrFs[:, None, None] * drs + d3rrzFs[:, None, None] * dzs
  ddrzFs = drzdFs + d3rrzFs[:, None, None] * drs + d3rzzFs[:, None, None] * dzs
  ddz2Fs = dz2dFs + d3rzzFs[:, None, None] * drs + d3zzzFs[:, None, None] * dzs

  dFs = jnp.reshape(dFs, (ns, -1))
  dzs = jnp.reshape(dzs, (ns, -1))
  drs = jnp.reshape(drs, (ns, -1))
  ddrzFs = jnp.reshape(ddrzFs, (ns, -1))
  ddz2Fs = jnp.reshape(ddz2Fs, (ns, -1))
  ddr2Fs = jnp.reshape(ddr2Fs, (ns, -1))

  return dFs, dzs, drs, ddz2Fs, ddr2Fs, ddrzFs
