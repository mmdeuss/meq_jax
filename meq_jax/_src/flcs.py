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

"""Functions for processing plasma domain."""

import jax
import jax.numpy as jnp
import jaxtyping as jt
from meq_jax._src import types
from meq_jax._src.bspsum import bspsum  # pylint: disable=g-importing-member
from meq_jax._src.cs2devalmex import cs2devalmex  # pylint: disable=g-importing-member
from meq_jax._src.cs2devalmex import csint  # pylint: disable=g-importing-member


bspsum = jax.jit(bspsum)
cs2deval = jax.jit(cs2devalmex)


eps = jnp.finfo(float).eps


# pylint: disable=invalid-name
def jac(rl: jt.Float[jt.Array, 'ns'],
        zl: jt.Float[jt.Array, 'ns'],
        L: types.StaticData) -> jt.Float[jt.Array, 'ns nrx*nzx']:
  """Jacobian of flcs w.r.t. rl and zl."""

  assert L.taur is not None and L.tauz is not None
  rk, zk = jnp.asarray(L.taur.ravel()), jnp.asarray(L.tauz.ravel())

  irs = jnp.minimum(
      jnp.maximum(jnp.floor((rl-rk[0])*L.idrx)+1, 4), L.nrx+3).astype(jnp.int64)
  izs = jnp.minimum(
      jnp.maximum(jnp.floor((zl-zk[0])*L.idzx)+1, 4), L.nzx+3).astype(jnp.int64)

  drs = (rl-rk[irs-1]) * L.idrx
  dzs = (zl-zk[izs-1]) * L.idzx

  def _fun(ir, iz, dr, dz):
    # R-interpolation
    A = csint(dr, 0, 1)
    Ar = jax.lax.dynamic_slice(L.Mr, (0, ir-4), (L.nrx, 4)) @ A

    # Z-interpolation
    B = csint(dz, 0, 1)
    Bz = jax.lax.dynamic_slice(L.Mz, (0, iz - 4), (L.nzx, 4)) @ B
    return jnp.outer(Ar, Bz).ravel()

  return jax.vmap(_fun)(irs, izs, drs, dzs)


def flcs(Fx: jt.Float[jt.Array, 'ny nx'],
         FN: jt.Float[jt.Array, ''],
         L: types.StaticData) -> tuple[
             jt.Float[jt.Array, 'ns'],  # Fl
             jt.Float[jt.Array, 'ns'],  # rl
             jt.Float[jt.Array, 'ns'],  # zl
             jt.Float[jt.Array, 'ns'],  # drFl
             jt.Float[jt.Array, 'ns'],  # dzFl
             ]:
  """Cubic spline interpolation of flux on limiter and id of local extremas.

  This function performs cubic spline interpolation of flux on a limiter and
  identifies local extrema. The limiter is described using piecewise cubic
  polynomials.

  Args:
    Fx: Flux field.
    FN: Limiter threshold.
    L: Static data struct.

  Returns:
    Flux on the limiter, limiter positions, a derivatives of flux on the
    limiter.
  """
  # Assume knots with multiplicity 3 for a cubic spline
  assert L.taur is not None and L.tauz is not None and L.taul is not None
  taul = jnp.array(L.taul.ravel())
  breaks = jnp.concatenate([taul[3:4], taul[4:-4:3], taul[-4:-3]])
  nb = len(breaks)
  assert L.G is not None and L.G.rl is not None and L.G.zl is not None
  rl, zl = jnp.array(L.G.rl.ravel()), jnp.array(L.G.zl.ravel())
  nl = len(rl)

  assert L.Mr is not None and L.Mz is not None and L.Ml is not None
  def bspsum_to_rz(s, d):
    x = bspsum(taul, L.Ml, s, d, 0, 0)
    return x[0], x[1]

  # Spline coefficients for the flux Fx
  c = L.Mr.T @ (Fx @ L.Mz)

  # Create a 2D spline interpolator for the flux F(r, z)
  rk, zk = jnp.array(L.taur.ravel()), jnp.array(L.tauz.ravel())

  # Evaluate flux on the initial limiter points
  # TODO(pfau): double check row-vs-col here
  Fl, *_ = cs2deval(rk, zk, c.T, rl, zl, 0.)
  drFl = jnp.zeros(nl)
  dzFl = jnp.zeros(nl)

  sl = jnp.arange(nl) / nl

  # Eliminate non-extrema points
  dFl = jnp.diff(jnp.concatenate([jnp.array([Fl[-1]]), Fl]))
  dFl_shifted = jnp.roll(dFl, -1)

  mask_non_extrema = (dFl * dFl_shifted > 0) | ((dFl - dFl_shifted) * FN > 0)
  Fl = jnp.where(mask_non_extrema, FN, Fl)

  assert L.P is not None and L.P.itercs is not None and L.P.tolcs is not None
  niter = L.P.itercs
  tol = L.P.tolcs
  dsmax = 1.0 / nl

  # initial values for iter, dl, mask, spike, rs, zs, ss
  # spike = jnp.zeros(len(il), dtype=bool)
  # init_val = 0, jnp.inf, jnp.ones(len(il), dtype=bool), spike, rs, zs, ss
  spike = jnp.zeros_like(Fl, dtype=bool)
  init_val = 0, jnp.inf, ~mask_non_extrema, spike, rl, zl, sl

  # condition for continuing the loop
  # TODO(pfau): Make sure this is identical to original convergence criterion.
  # The test tolerance had to be loosened after implementing this, because the
  # optimization is now run on all points and not just extrema.
  def _cond(val):
    i, dl, mask, *_ = val
    return jnp.logical_and(jnp.logical_or(dl > tol, jnp.any(mask)), i < niter)

  # body function for the loop
  def _body(val):
    i, _, _, spike, rs, zs, ss = val
    ib = jnp.searchsorted(breaks, ss, side='right')
    smin, smax = breaks[ib - 1], breaks[ib % nb]

    # Compute flux gradient and Hessian
    _, drFs, dzFs, dr2Fs, drzFs, dz2Fs, *_ = cs2deval(rk, zk, c.T, rs, zs, 1.)

    # Compute limiter tangent
    dsrs, dszs = bspsum_to_rz(ss, 1)
    ds2rs, ds2zs = bspsum_to_rz(ss, 2)

    # Compute flux derivatives along limiter
    # TODO(pfau): refactor into subfunc
    dsFs = drFs * dsrs + dzFs * dszs
    ds2Fs = (dr2Fs * dsrs**2 + 2 * drzFs * dsrs * dszs + dz2Fs * dszs**2 +
             drFs * ds2rs + dzFs * ds2zs)
    dss = -dsFs / ds2Fs
    dss = jnp.where(spike, 0, dss)
    mask = ds2Fs * FN < 0
    dss = jnp.where(mask, -jnp.sign(dss) * dsmax, dss)
    dss = jnp.clip(dss, -dsmax, dsmax)

    # Check for extremas when limiter contour has a slope discontinuity
    # (e.g. baffle tip).
    snew = jnp.clip(ss + dss, smin, smax)
    # Restrict to extrema points: Matlab only iterates over extrema (il), so
    # non-extrema points must not enter the discontinuity handling nor keep
    # the loop running via `any(mask)` in the convergence check.
    mask = ~spike & ((snew == smin) | (snew == smax)) & ~mask_non_extrema

    def _dss_fun(dss_, snew_, smin_, ss_, ib_):
      eps_ = jax.lax.cond(snew_ == smin_,
                          lambda: eps * jnp.array([1, -1]),
                          lambda: eps * jnp.array([-1, 1]))
      sx = (snew_ + eps_) % 1.0

      # TODO(pfau): refactor into subfunc
      rx, zx = bspsum_to_rz(sx, 0)
      _, drFx, dzFx, dr2Fx, drzFx, dz2Fx, *_ = cs2deval(
          rk, zk, c.T, rx, zx, 1.0)
      dsrx, dszx = bspsum_to_rz(sx, 1)
      ds2rx, ds2zx = bspsum_to_rz(sx, 2)

      dsFx = drFx * dsrx + dzFx * dszx
      ds2Fx = (dr2Fx * dsrx**2 + 2 * drzFx * dsrx * dszx + dz2Fx * dszx**2 +
               drFx * ds2rx + dzFx * ds2zx)
      dsx = -dsFx / ds2Fx
      maskx = ds2Fx * FN < 0
      dsx = jnp.where(maskx, -jnp.sign(dsx) * dsmax, dsx)

      def _case_0(op):
        _, snew_, ss_, _, _ = op
        return 0.5 * (snew_ - ss_), False

      def _case_1(op):
        _, snew_, ss_, _, _ = op
        return snew_ - ss_, True

      def _case_2(op):
        _, _, ss_, sx, ib_ = op
        dss_ = sx[1] - ss_
        dss_ = jax.lax.cond(
            jnp.logical_and(ib_ == 1, dss_ < 0),
            lambda d: d - 1,
            lambda d: d,
            dss_)
        dss_ = jax.lax.cond(
            jnp.logical_and(ib_ == nb - 1, dss_ > 0),
            lambda d: d + 1,
            lambda d: d,
            dss_)
        return dss_, False

      index = jnp.where(dsx[0] * dss_ < 0, 0,
                        jnp.where(dsx[1] * dss_ < 0, 1, 2))
      return jax.lax.switch(index, [_case_0, _case_1, _case_2],
                            (dss_, snew_, ss_, sx, ib_))

    dss_, spike_ = jax.vmap(_dss_fun)(dss, snew, smin, ss, ib)
    dss = jnp.where(mask, dss_, dss)
    spike = jnp.where(mask, spike_, spike)

    # NOTE: must match Matlab's `ss = mod(ss + dss,1)` exactly: the
    # discontinuity handler above may return an eps-sized step (case 2,
    # `dss = sx[1] - ss`) to move the point across a segment joint. Adding
    # any offset here would cancel that step and pin the point at the break.
    ss = (ss + dss) % 1.0
    rs_, zs_ = bspsum_to_rz(ss, 0)

    # Maximum change in real space coordinates (R,Z)
    # Ignore non-extrema when determining convergence
    dl = jnp.max(jnp.abs(jnp.concatenate([
        jnp.where(mask_non_extrema, 0, rs_ - rs),
        jnp.where(mask_non_extrema, 0, zs_ - zs)
    ])))
    return i+1, dl, mask, spike, rs_, zs_, ss

  _, _, _, _, rs, zs, _ = jax.lax.while_loop(_cond, _body, init_val)

  # Final evaluation at converged points
  Fl_, drFl_, dzFl_, *_ = cs2deval(rk, zk, c.T, rs, zs, 1.)

  # Update output arrays with final extrema positions
  return (jnp.where(mask_non_extrema, Fl, Fl_),
          jnp.where(mask_non_extrema, rl, rs),
          jnp.where(mask_non_extrema, zl, zs),
          jnp.where(mask_non_extrema, drFl, drFl_),
          jnp.where(mask_non_extrema, dzFl, dzFl_))
