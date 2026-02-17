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

"""Python implementation of meqpdom (compute plasma domain)."""

import jax
import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import asxy
from meq_jax._src import asxycs
from meq_jax._src import bavx
from meq_jax._src import fbnd
from meq_jax._src import fl4p
from meq_jax._src import flcs
from meq_jax._src import types


def get_finite(x, size=None):
  """This is a jit-compatible version of x[jnp.isfinite(x)].

  Args:
    x: The array to get the finite elements from.
    size: The size of the output array. If None, the output array will give the
      same result as x[jnp.isfinite(x)], otherwise it will be padded with -inf.
      To be jit-compatible, size must be specified.
  Returns:
    An array containing the finite elements of x, possibly padded with -inf.
  """
  # We can safely set the fill value to the maximum value of int32, because
  # we will never deal with an array with 2**32 elements.
  idx = jnp.nonzero(jnp.isfinite(x),
                    size=size,
                    fill_value=jnp.iinfo(jnp.int32).max)
  val = x[idx]
  # Fill in anything out of bounds with -inf. It should do this anyway in most
  # cases, but we want to be absolutely sure.
  return jnp.where(idx[0] == jnp.iinfo(jnp.int32).max, -jnp.inf, val)


def swap(x, na, nx, dimw):
  """Swap the finite elements of x into the correct location.

  Assume that x is split into blocks of shape (na, dimw - na, nx, dimw - nx),
  where blocks are alternately finite and infinite. This swaps the second and
  third blocks in a way that is jit-compatible.

  Args:
    x: The array to swap the elements of.
    na: The size of the first block.
    nx: The size of the third block.
    dimw: The size of the first two blocks.
  Returns:
    The swapped array.
  """
  idx = jnp.arange(x.shape[0])
  shift_amount = na - dimw
  shifted_x = jnp.roll(x, shift_amount)
  swap_mask = (idx >= na) & (idx < na + nx)
  inf_mask = (idx >= dimw)
  x = jnp.where(swap_mask, shifted_x, x)
  x = jnp.where(inf_mask, -jnp.inf, x)
  return x


# pylint: disable=invalid-name
def meqpdom(
    fx: jt.Float[jt.Array, 'ny nx'],
    Ip: float | jt.Array,
    isaddl: bool,
    L: types.StaticData,
) -> tuple[
    jt.Float[jt.Array, 'na'],  # ra
    jt.Float[jt.Array, 'na'],  # za
    jt.Float[jt.Array, 'na'],  # fa
    jt.Float[jt.Array, 'na'],  # dr2fa
    jt.Float[jt.Array, 'na'],  # dz2fa
    jt.Float[jt.Array, 'na'],  # drzfa
    jt.Float[jt.Array, 'nX'],  # rX
    jt.Float[jt.Array, 'nX'],  # zX
    jt.Float[jt.Array, 'nX'],  # fX
    jt.Float[jt.Array, 'nX'],  # dr2fX
    jt.Float[jt.Array, 'nX'],  # dz2fX
    jt.Float[jt.Array, 'nX'],  # drzfX
    jt.Float[jt.Array, 'nB'],  # rb
    jt.Float[jt.Array, 'nB'],  # zb
    jt.Float[jt.Array, 'nB'],  # fb
    jt.Bool[jt.Array, ''],  # lb
    jt.Bool[jt.Array, 'nB'],  # lx
    jt.Integer[jt.Array, 'ny nx'],  # Opy
    jt.Float[jt.Array, 'nD'],  # f0
    jt.Float[jt.Array, 'nD'],  # f1
    jt.Integer[jt.Array, ''],  # status
    jt.Float[jt.Array, 'nD ny nx'],  # df0dfx
    jt.Float[jt.Array, 'nD ny nx'],  # df1dfx
    jt.Integer[jt.Array, 'np'],  # ixi
]:
  """MEQpdom: Plasma domain parameters.

  meqpdom returns position, flux gradients of magnetic axes, x points,
  boundary points, as well as plasma domain parameters. To make the function
  jittable, the output is padded to length L.dimw with -inf.

  Args:
    fx: is the flux map
    Ip: plasma current for its sign, Ip=0 corresponding to a vacuum case
    isaddl: selects the use of X-points 0: exclude x-points 1: use x-points
    L: static data

  Returns:
    ra, za, fa: position and flux of the magnetic axes
    dr2fa, dz2fa, drzfa: flux 2nd derivatives on axes,
    rX, zX, fX: position and flux of the X-points on the X-points polygon,
    dr2fX, dz2fX, drzfX: flux 2nd derivatives at X-point,
    rb, zb, fb: point and flux defining LCFS or internal separatrices
    lb: is FALSE if no valid point defines the LCFS
    lX: is TRUE if an X-point defines the LCFS or separatrix
    Opy: indices indicate different plasma domain
    f0, f1: limiting flux values for each domain.
    status: is TRUE if the domain identification is successful
    df0dfx, df1dfx: derivatives of F0 and F1 w.r.t the Fx
      (sparse matrices if P.icsint=false)
    iXi: index of grid cells where central points were found

  For details, see: [J-M. Moret et al. Fus.Eng.Des 2015], Section 2.3
  Multi-domain case for doublets extension added later
  """

  sIp = jnp.sign(Ip)
  assert L.nzy is not None and L.nry is not None
  Opy = jnp.zeros((L.nry, L.nzy), dtype=jnp.int8)
  assert L.G is not None and L.G.rx is not None and L.G.zx is not None
  assert L.P is not None and L.P.ilim is not None

  assert L.FN is not None
  fn = L.FN * sIp

  # find all x/o points
  (za, ra, fa, dz2fa, dr2fa, drzfa, ixa,
   zX, rX, fX, dz2fX, dr2fX, drzfX, ixX, stat) = jax.lax.cond(
       jnp.all(fx == fx[0, 0]),
       # all the same flux, happens for vacuum case with Ia=0
       lambda: 14 * (-jnp.inf * jnp.ones_like(fx),) + (True,),
       lambda: asxy.asxy(
           fx, L.G.zx, L.G.rx, L.P.dasm,
           L.dzx, L.drx, L.idzx, L.idrx, L.Oasx, L.dimw))

  # unlike MATLAB, asxy returns masked arrays
  na = jnp.count_nonzero(jnp.isfinite(fa))
  nX = jnp.count_nonzero(jnp.isfinite(fX))
  isvacuum = jnp.logical_or(na == 0, sIp == 0)

  if L.P.ilim == 0:
    fl = jnp.zeros(0)
    drfl = fl
    dzfl = fl
    rl = fl
    zl = fl
  elif L.P.ilim == 1:  # default limiter treatment - only on rl,zl points
    # interpolate and keep only compatible sign of extrema
    fl, drfl, dzfl = fl4p.fl4pmex(fx, L.kxl - 1, L.clx, fn)
    rl = L.G.rl
    zl = L.G.zl
  elif L.P.ilim == 2:
    raise NotImplementedError('fl4pinterp not implemented')
  elif L.P.ilim == 3:
    # Use cubic spline for Fx interpolation and pp interpolation for the limiter
    fl, rl, zl, drfl, dzfl = flcs.flcs(fx, fn, L)
  else:
    raise ValueError(f'unknown ilim value {L.P.ilim}')

  # Move this out from the above if statement for jit-compatibility
  fl, drfl, dzfl, rl, zl = jax.lax.cond(
      isvacuum,
      lambda: 5 * (jnp.ones_like(fl) * fn,),
      lambda: (fl, drfl, dzfl, rl, zl)
  )

  if L.P.icsint:
    output = asxycs.asxycs(
        fx, za, ra, zX, rX, L.Mr, L.Mz, L.taur, L.tauz, L.P.itercs, L.P.tolcs)
    za, ra, fa, dz2fa, dr2fa, drzfa, zX, rX, fX, dz2fX, dr2fX, drzfX = output

  iAok = -jnp.sign(dr2fa + dz2fa) == sIp
  fa, ra, za, dr2fa, dz2fa, drzfa, ixa = jax.lax.cond(
      L.P.ihole or isvacuum,
      lambda: (fa, ra, za, dr2fa, dz2fa, drzfa, ixa),
      lambda: (jnp.where(iAok, fa, -jnp.inf),
               jnp.where(iAok, ra, -jnp.inf),
               jnp.where(iAok, za, -jnp.inf),
               jnp.where(iAok, dr2fa, -jnp.inf),
               jnp.where(iAok, dz2fa, -jnp.inf),
               jnp.where(iAok, drzfa, -jnp.inf),
               jnp.where(iAok, ixa, -jnp.inf))
  )
  na = jnp.count_nonzero(jnp.isfinite(fa))

  za = get_finite(za, size=L.dimw)
  ra = get_finite(ra, size=L.dimw)
  fa = get_finite(fa, size=L.dimw)
  dz2fa = get_finite(dz2fa, size=L.dimw)
  dr2fa = get_finite(dr2fa, size=L.dimw)
  drzfa = get_finite(drzfa, size=L.dimw)
  ixa = get_finite(ixa, size=L.dimw)

  zX = get_finite(zX, size=L.dimw)
  rX = get_finite(rX, size=L.dimw)
  fX = get_finite(fX, size=L.dimw)
  dz2fX = get_finite(dz2fX, size=L.dimw)
  dr2fX = get_finite(dr2fX, size=L.dimw)
  drzfX = get_finite(drzfX, size=L.dimw)
  ixX = get_finite(ixX, size=L.dimw)

  # order fa and fX by vertical position
  k = jnp.argsort(za, descending=True)
  fa = fa[k]
  za = za[k]
  ra = ra[k]
  dr2fa = dr2fa[k]
  dz2fa = dz2fa[k]
  drzfa = drzfa[k]
  ixa = ixa[k]

  # Use X point hessians to compute directions of X point
  # (towards positive gradient direction)
  H1 = sIp * (dr2fX - dz2fX) * 0.5
  H0 = jnp.sqrt(H1 ** 2 + drzfX ** 2)
  vrx0 = jnp.sqrt(H0 + H1)
  vzx0 = jnp.sqrt(H0 - H1) * ((drzfX >= 0)*2 - 1) * sIp

  # Set up loop over potential domains
  np = 2 * L.dimw  # Number of potential domains
  active = ~isvacuum  # Flag to continue the scan
  lb = False  # Flag for successful domain identification
  # Number of domains connected to X-point
  kx = jnp.zeros(L.dimw, dtype=jnp.int32)
  ix = jnp.ones(L.dimw, dtype=jnp.bool_)  # Flag for X-points yet to be treated
  init_carry = active, lb, kx, ix, Opy

  # What follows is a truly horrific set of nested cond statements that
  # reproduces the logic of lines 155-205 of the Matlab code, due to the
  # presence of multiple "break" statements in the original loop.
  def update_kx_Opy(kx, kb, fi, fb, lx, OX, Opy, kp):
    # Increase number of domains associated to X-point
    kx_ = jax.lax.cond(
        lx,
        lambda: kx.at[kb-1].add(1),
        lambda: kx
    )
    Oxyk = jnp.logical_and(OX, (fx - fi) * (fx - fb) <= 0)

    # Shift indices as if the padding was not there.
    kp_ = jax.lax.cond(kp < na,
                       lambda k: k,
                       lambda k: k - (L.dimw - na),
                       kp)
    # This depends on lxy always being a mask with the boundary set to zero
    Opy_ = jnp.where(Oxyk[1:-1, 1:-1], kp_ + 1, Opy)  # indicized Opy
    return kx_, jnp.astype(Opy_, jnp.int8)

  def inner_fun(fX_, fl_, ri, zi, fi, vrx, vzx, kx, Opy, kp):
    """This branch contains the part shared for axis and saddle points."""
    # use bavx to 'cut' regions outside of X-point polygon
    vsx = jnp.sign(vrx * (ri - rX) + vzx * (zi - zX))
    vrx = vrx * vsx
    vzx = vzx * vsx
    # Eliminate X-points outside of X-point polygon
    selX = fX_ != fn
    k = bavx.bavx(rX, zX, vrx, vzx, 0, rX, zX, selX)
    fX_ = jnp.where(~k | (not isaddl), fn, fX_)  # isaddl=0 excludes all X-pts
    # cut regions to be excluded on (r,z)
    OX = bavx.bavx2(rX, zX, vrx, vzx, 0, L.G.rx, L.G.zx, selX, None)

    # Exclude limiter points with wrong sign of derivatives
    dafl = (drfl * (ri - rl) + dzfl * (zi - zl))
    fl_ = jnp.where(dafl * fn > 0, fn, fl_)

    # Exclude candidate limiter points outside x-point polygon
    fl_ = jnp.where(
        bavx.bavx(rX, zX, vrx, vzx, L.P.xdoma, rl, zl, selX), fl_, fn)

    # Boundary flux from limiter points or X-points
    fb, rb, zb, lb, lx, kb = fbnd.fbnd(
        fl_, rl, zl, fX_, rX, zX, jnp.astype(fn, jnp.float64))

    kx, Opy = jax.lax.cond(
        lb,
        update_kx_Opy,
        lambda *args: (kx, Opy),
        kx, kb, fi, fb, lx, OX, Opy, kp
    )

    return fb, rb, zb, lb, lx, kb, kx, Opy

  def pad_fun(carry, kp: int):
    """This branch is a no-op for padded values."""
    del kp  # Unused.
    return carry, (-jnp.inf,  # ri
                   -jnp.inf,  # zi
                   -jnp.inf,  # fi
                   -1.,  # ixi
                   -jnp.inf,  # fb
                   -jnp.inf,  # rb
                   -jnp.inf,  # zb
                   False,  # lx
                   jnp.astype(-1, jnp.int32))   # kb

  def break_fun(carry, kp: int):
    """This branch is taken where a break statement appears in Matlab."""
    del kp  # Unused.
    _, lb, kx, ix, Opy = carry
    return ((False, lb, kx, ix, Opy),
            (0., 0., 0., 0., 0., 0., 0., False, jnp.astype(0, jnp.int32)))

  def axis_fun(carry, kp: int):
    """This branch is taken for domains that contain a max or min (axis)."""
    _, _, kx, ix, Opy = carry
    ri = ra[kp]
    zi = za[kp]
    fi = fa[kp]
    ixi = ixa[kp]
    vrx = ri - rX
    vzx = zi - zX

    fX_ = jnp.where(fX == -jnp.inf, fn, fX)
    fb, rb, zb, lb, lx, kb, kx, Opy = inner_fun(
        fX_, fl, ri, zi, fi, vrx, vzx, kx, Opy, kp)

    return (lb, lb, kx, ix, Opy), (ri, zi, fi, ixi, fb, rb, zb, lx, kb)

  def saddle_inner_fun(carry, kp: int, mask):
    """Take this branch if mask.any() is True."""
    _, _, kx, ix, Opy = carry
    k = jnp.argwhere(mask, size=1)[0, 0]
    # Eliminate X-points already connected to at least 2 domains
    fX_ = jnp.where(jnp.logical_or(kx > 1, fX == -jnp.inf), fn, fX)
    ix = ix.at[k].set(False)  # X-point has been treated
    ri = rX[k]
    zi = zX[k]
    fi = fX[k]
    ixi = ixX[k]
    vrx = vrx0
    vzx = vzx0

    fb, rb, zb, lb, lx, kb, kx, Opy = inner_fun(
        fX_, fl, ri, zi, fi, vrx, vzx, kx, Opy, kp)

    return (lb, lb, kx, ix, Opy), (ri, zi, fi, ixi, fb, rb, zb, lx, kb)

  def saddle_fun(carry, kp: int):
    """This branch is taken for domains that contain a saddle (x-point)."""
    _, _, kx, ix, _ = carry
    mask = jnp.logical_and(ix, (kx > 1))
    new_carry, new_scan = jax.lax.cond(
        mask.any(),
        saddle_inner_fun,
        lambda carry_, kp_, mask_: break_fun(carry_, kp_),
        carry, kp, mask
    )
    return new_carry, new_scan

  def body_fun(carry, kp: int):
    active, *_ = carry
    new_carry, new_scan = jax.lax.cond(
        kp < L.dimw,
        lambda carry_, kp_: jax.lax.cond(
            kp < na,
            lambda carry__, kp__: jax.lax.cond(
                active, axis_fun, break_fun, carry__, kp__),
            pad_fun,
            carry_,
            kp_
        ),
        lambda carry_, kp_: jax.lax.cond(
            kp < L.dimw + nX,
            lambda carry__, kp__: jax.lax.cond(
                active, saddle_fun, break_fun, carry__, kp__),
            pad_fun,
            carry_,
            kp_
        ),
        carry, kp
    )
    return new_carry, new_scan

  carry, (ri, zi, fi, ixi, fb, rb, zb, lx, kb) = jax.lax.scan(body_fun,
                                                              init_carry,
                                                              jnp.arange(np))
  _, lb, _, _, Opy = carry

  ri = swap(ri, na, nX, L.dimw)
  zi = swap(zi, na, nX, L.dimw)
  fi = swap(fi, na, nX, L.dimw)
  ixi = swap(ixi, na, nX, L.dimw)
  fb = swap(fb, na, nX, L.dimw)
  rb = swap(rb, na, nX, L.dimw)
  zb = swap(zb, na, nX, L.dimw)
  lx = swap(lx, na, nX, L.dimw)
  kb = swap(kb, na, nX, L.dimw)

  # Handle edge case for vacuum
  ixi = jax.lax.cond(isvacuum,
                     lambda: jnp.ones_like(ixi) * -jnp.inf,
                     lambda: ixi)

  # truncate to effective number of domains
  nb = jax.lax.cond(jnp.any(rb > 0),
                    lambda: jnp.argwhere(rb <= 0, size=1)[0, 0],
                    lambda: 0)
  mask = jnp.arange(np) < nb
  fb = jnp.where(mask, fb, -jnp.inf)
  rb = jnp.where(mask, rb, -jnp.inf)
  zb = jnp.where(mask, zb, -jnp.inf)
  lx = jnp.where(mask, lx, False)
  kb = jnp.where(mask, kb, -1)
  # mask outside limiter
  assert L.Oly is not None
  Opy = jnp.where(L.Oly, Opy, 0)

  k = jax.lax.cond(
      L.nD == 1 and nb == 1,
      # Order X-points by proximity to FB
      lambda: jnp.argsort(jnp.abs(fX - fb[0])),
      # Order X-points by flux value
      lambda: jnp.argsort(fX, descending=True)
  )

  ixX = ixX[k]
  fX = fX[k]
  zX = zX[k]
  rX = rX[k]
  dr2fX = dr2fX[k]
  dz2fX = dz2fX[k]
  drzfX = drzfX[k]

  # Inverse of k permutation
  ik = jnp.zeros(L.dimw).at[k].set(jnp.arange(L.dimw).astype(jnp.int32))
  kb = jnp.where(lx, ik[kb.astype(jnp.int32)-1], kb).astype(jnp.int32)

  # Limiting values for iD=1..L.nD
  f0 = jnp.zeros(L.nD)
  f1 = f0
  nbeff = jnp.minimum(L.nD, nb)
  mask = jnp.arange(L.nD) < nbeff
  f0 = jnp.where(mask, fi[:L.nD], f0)
  f1 = jnp.where(mask, fb[:L.nD], f1)

  # Compute derivatives of f0 and f1 w.r.t. fx
  if L.P.icsint:
    df0dfx = flcs.jac(ri[: L.nD], zi[: L.nD], L)
    df1dfx = flcs.jac(rb[: L.nD], zb[: L.nD], L)
    df0dfx = jnp.where(mask[:, None], df0dfx, 0.0)
    df1dfx = jnp.where(mask[:, None], df1dfx, 0.0)
  else:
    # L.nD is an upper bound on nbeff, so just evaluate asxy.jac at all possible
    # indices and then select the valid ones.
    ixs = jnp.concatenate(
        [ixi[:L.nD], ixX[kb[:L.nD]]], axis=0).astype(jnp.int32)
    dfsdfx, *_ = asxy.jac(
        fx, L.G.zx, L.G.rx, L.dzx, L.drx, L.idzx, L.idrx, ixs)
    dfx_idx = jnp.tile(jnp.arange(L.nD)[:, None], [1, L.nx])
    df0dfx = jnp.where(dfx_idx < nbeff, dfsdfx[:L.nD], 0.)
    df1dfx = jnp.where(
        jnp.logical_and(dfx_idx < nbeff,
                        jnp.tile(lx[:L.nD, None], [1, L.nx])),
        dfsdfx[L.nD:], 0.)

    # # Limiter points
    if L.P.ilim == 1:
      ib = jnp.tile(jnp.arange(L.nD)[:, None], [1, 4])
      ix = L.kxl[kb[:L.nD] - 1, None] + jnp.array([0, L.nzx, L.nzx + 1, 1])

      arr = jnp.where(
          jnp.tile(
              jnp.logical_and(
                  (jnp.arange(L.nD) < nbeff)[:, None],
                  jnp.logical_not(lx[: L.nD, None]),
              ),
              [1, 4],
          ),
          L.clx[kb[: L.nD] - 1],
          0.0,
      )

      df1dfx = df1dfx.at[ib, ix - 1].add(arr)

  return (ra, za, fa, dr2fa, dz2fa, drzfa, rX, zX, fX, dr2fX, dz2fX, drzfX,
          rb, zb, fb, lb, lx, Opy, f0, f1, stat, df0dfx, df1dfx, ixi)
