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

"""asxy majic implementation."""

import jax
import jax.numpy as jnp
import jaxtyping as jt


roll = lambda x, *shift: jnp.roll(x, shift, (0, 1))


# pylint: disable=invalid-name
def six_pt_interpolant(
    Fxy: jt.Float[jt.Array, 'ny nx'],
    x: jt.Float[jt.Array, 'nx'],
    y: jt.Float[jt.Array, 'ny'],
    Dx: jt.Float[jt.Array, ''],
    Dy: jt.Float[jt.Array, ''],
    IDx: jt.Float[jt.Array, ''],
    IDy: jt.Float[jt.Array, ''],
) -> tuple[
    jt.Float[jt.Array, 'ny+2 nx+2'],  # Fxy_
    jt.Float[jt.Array, 'ny nx'],  # a
    jt.Float[jt.Array, 'ny nx'],  # b
    jt.Float[jt.Array, 'ny nx'],  # c
    jt.Float[jt.Array, 'ny nx'],  # d
    jt.Float[jt.Array, 'ny nx'],  # e
    jt.Float[jt.Array, 'ny nx'],  # fe
    jt.Float[jt.Array, 'ny nx'],  # xe
    jt.Float[jt.Array, 'ny nx'],  # ye
    jt.Float[jt.Array, 'ny nx'],  # xs
    jt.Float[jt.Array, 'ny nx'],  # ys
    jt.Float[jt.Array, 'ny nx'],  # dx2fs
    jt.Float[jt.Array, 'ny nx'],  # dy2fs
    jt.Float[jt.Array, 'ny nx'],  # dxyfs
]:
  """Find location of extrema and saddle from six-point interpolant method."""
  Fxy_ = jnp.pad(Fxy, 1, 'edge')

  # Find location of extrema and saddle from six-point interpolant method.
  d1 = Fxy_[1:-1, 0:-2] - Fxy
  d2 = Fxy_[0:-2, 1:-1] - Fxy
  d3 = Fxy_[0:-2, 2:] - Fxy
  d4 = Fxy_[1:-1, 2:] - Fxy
  d6 = Fxy_[2:, 1:-1] - Fxy
  a = 0.5 * (d4 - d1)
  b = 0.5 * (d6 - d2)
  c = d1 + d4
  d = d2 + d6
  e = d4 + d2 - d3

  xe = jnp.clip((e * b - a * d) / (c * d - e**2), -1, 1)
  ye = jnp.clip((e * a - b * c) / (c * d - e**2), -1, 1)
  fe = (0.5 * c * xe + a) * xe + (0.5 * d * ye + e * xe + b) * ye + Fxy

  xs = xe * Dx + x[None]
  ys = ye * Dy + y[:, None]
  dx2fs = c * IDx * IDx
  dy2fs = d * IDy * IDy
  dxyfs = e * IDx * IDy
  return Fxy_, a, b, c, d, e, fe, xe, ye, xs, ys, dx2fs, dy2fs, dxyfs


def jac(
    Fxy: jt.Float[jt.Array, 'ny nx'],
    x: jt.Float[jt.Array, 'nx'],
    y: jt.Float[jt.Array, 'ny'],
    Dx: jt.Float[jt.Array, ''],
    Dy: jt.Float[jt.Array, ''],
    IDx: jt.Float[jt.Array, ''],
    IDy: jt.Float[jt.Array, ''],
    ixs: jt.Integer[jt.Array, 'ni'],
) -> tuple[
    jt.Float[jt.Array, 'ni ny nx'],  # dfedFx
    jt.Float[jt.Array, 'ni ny nx'],  # dxsdFx
    jt.Float[jt.Array, 'ni ny nx'],  # dysdFx
    jt.Float[jt.Array, 'ni ny nx'],  # ddx2fsdFx
    jt.Float[jt.Array, 'ni ny nx'],  # ddy2fsdFx
    jt.Float[jt.Array, 'ni ny nx'],  # ddxyfsdFx
]:
  """Implementation of asxyJac in JAX."""
  ni = len(ixs)
  nrx, nzx = Fxy.shape
  # Indices of the six stencil points (i5 is unused for some reason)
  i1 = ixs - 1
  i2 = ixs - nzx
  i3 = i2 + 1
  i4 = ixs + 1
  i6 = ixs + nzx
  idxs = [ixs, i1, i2, i3, i4, i6]

  output = six_pt_interpolant(Fxy, x, y, Dx, Dy, IDx, IDy)
  _, a, b, c, d, e, _, xe, ye, _, _, _, _, _ = output

  # Compute gradients at all points
  dabcdf = jnp.array(
      [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, -0.5, 0.0, 0.0, 0.5, 0.0],
       [0.0, 0.0, -0.5, 0.0, 0.0, 0.5],
       [-2.0, 1.0, 0.0, 0.0, 1.0, 0.0],
       [-2.0, 0.0, 1.0, 0.0, 0.0, 1.0],
       [-1.0, 0.0, 1.0, -1.0, 1.0, 0.0]])
  one = jnp.ones_like(xe)
  zero = jnp.zeros_like(xe)
  h = (c * d - e ** 2)[..., None]

  # Jacobian of the flux value
  dFSdFx = jnp.stack(
      [one, xe, ye, 0.5 * xe ** 2, 0.5 * ye ** 2, xe * ye],
      axis=-1) @ dabcdf

  # Jacobian of the r and z position of the extrema
  drSdFx = (jnp.stack(
      [zero, e, -c, -(b + d * ye), -c * ye, (a + 2 * e * ye)],
      axis=-1) * Dy / h) @ dabcdf
  dzSdFx = (jnp.stack(
      [zero, -d, e, -d * xe, -(a + c * xe), (b + 2 * e * xe)],
      axis=-1) * Dx / h) @ dabcdf

  def _make_jac(data):
    idxs_arr = jnp.stack(idxs)

    def body(j, foo):
      bar = data[..., j].ravel()
      return foo.at[jnp.arange(ni), idxs_arr[j] - 1].set(bar[ixs - 1])

    return jax.lax.fori_loop(0, 6, body, jnp.zeros((ni, nrx * nzx)))
  dFSdFx = _make_jac(dFSdFx)
  drSdFx = _make_jac(drSdFx)
  dzSdFx = _make_jac(dzSdFx)

  # Jacobian of the terms in the Hessian matrix
  def _make_hess(i, dx):
    idxs_arr = jnp.stack(idxs)

    def body(j, foo):
      return foo.at[jnp.arange(ni), idxs_arr[j] - 1].set(dabcdf[i, j] * dx)

    return jax.lax.fori_loop(0, 6, body, jnp.zeros((ni, nrx * nzx)))

  ddz2FSdFx = _make_hess(3, IDx * IDx)
  ddr2FSdFx = _make_hess(4, IDy * IDy)
  ddrzFSdFx = _make_hess(5, IDx * IDy)

  return dFSdFx, dzSdFx, drSdFx, ddz2FSdFx, ddr2FSdFx, ddrzFSdFx


def asxy(
    Fxy: jt.Float[jt.Array, 'ny nx'],
    x: jt.Float[jt.Array, 'nx'],
    y: jt.Float[jt.Array, 'ny'],
    Dc: jt.Float[jt.Array, ''],
    Dx: jt.Float[jt.Array, ''],
    Dy: jt.Float[jt.Array, ''],
    IDx: jt.Float[jt.Array, ''],
    IDy: jt.Float[jt.Array, ''],
    Lxy: jt.Bool[jt.Array, 'ny nx'],
    dimw: jt.Float[jt.Array, ''],
) -> tuple[
    jt.Float[jt.Array, 'ny nx'],  # xa
    jt.Float[jt.Array, 'ny nx'],  # ya
    jt.Float[jt.Array, 'ny nx'],  # fa
    jt.Float[jt.Array, 'ny nx'],  # dx2fa
    jt.Float[jt.Array, 'ny nx'],  # dy2fa
    jt.Float[jt.Array, 'ny nx'],  # dxyfa
    jt.Integer[jt.Array, 'ny nx'],  # ixa
    jt.Float[jt.Array, 'ny nx'],  # xs
    jt.Float[jt.Array, 'ny nx'],  # ys
    jt.Float[jt.Array, 'ny nx'],  # fs
    jt.Float[jt.Array, 'ny nx'],  # dx2fs
    jt.Float[jt.Array, 'ny nx'],  # dy2fs
    jt.Float[jt.Array, 'ny nx'],  # dxyfs
    jt.Integer[jt.Array, 'ny nx'],  # ixs
    jt.Bool[jt.Array, ''],  # stat
]:
  """Localise extrema and saddle points of a 2D-map.

  Note that JAX requires output shapes to be derivable from input shapes, so
  we return the outputs as masked arrays of the same size as the input rather
  than as arrays of variable length. The outputs are then shaped correctly in
  asxymex.m if the MAJIC-compiled function is called from Matlab.

  Args:
    Fxy: 2D-map.
    x: 1D-vector of x-coordinates.
    y: 1D-vector of y-coordinates.
    Dc: Threshold for the distance between extrema and saddle points.
    Dx: Distance between points along the x-axis.
    Dy: Distance between points along the y-axis.
    IDx: Inverse of Dx.
    IDy: Inverse of Dy.
    Lxy: Mask for points that should not be considered.
    dimw: Meant to be the maximum length of the output, but not used because we
      return outputs of the same size as the input for JAX compatibility.
  """
  nx, ny = Fxy.shape

  output = six_pt_interpolant(
      Fxy, x, y, Dx, Dy, IDx, IDy
  )
  Fxy_, _, _, _, _, _, fe, _, _, xs, ys, dx2fs, dy2fs, dxyfs = output

  # Loop over neighbors and count the number of times the sign changes
  nbrs = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
  dFx = (
      lambda i: Fxy
      - Fxy_[
          1 + nbrs[i][0] : 1 + nx + nbrs[i][0],
          1 + nbrs[i][1] : 1 + ny + nbrs[i][1],
      ]
  )

  # TODO(pfau): Instead of doubling the computation, only do the computation
  # for the rows that we will keep.

  def six_point_count(idx):
    # Loop around neighboring points, counting sign changes.
    count = jnp.zeros_like(Fxy)
    df = dFx(idx[0])
    dl = dFx(idx[0])
    for i in idx[1:]:
      df = jnp.where(df == 0, dFx(i), df)
      mask = dl * dFx(i) < 0
      count = count + mask
      dl = jnp.where((dl == 0) | mask, dFx(i), dl)
    mask = dl * df < 0
    count = count + mask
    return count

  # Do the count for odd rows, then even rows
  count = six_point_count([3, 5, 6, 7, 0, 1])
  count_even = six_point_count([7, 5, 4, 3, 2, 1])
  count = count.at[::2].set(count_even[::2])
  count = count // 2

  # Eliminate extrema on the boundary unless count == 2
  count = count.at[0].set(jnp.where(count[0] == 2, 2, 1))
  count = count.at[-1].set(jnp.where(count[-1] == 2, 2, 1))
  count = count.at[:, 0].set(jnp.where(count[:, 0] == 2, 2, 1))
  count = count.at[:, -1].set(jnp.where(count[:, -1] == 2, 2, 1))
  count = jnp.where(Lxy, count, 1)

  # Annihilate close extremum-saddle pairs. Note that, since we already removed
  # non-saddle extrema from the boundaries, we do not need to worry about points
  # wrapping around.
  for i in range(-1, 2):
    for j in range(-1, 2):

      def _true_fun(c):
        # pylint: disable=cell-var-from-loop
        mask = jnp.logical_and(c == 0, roll(c >= 2, i, j))
        return jnp.where(jnp.logical_or(mask, roll(mask, -i, -j)), 1, c)
        # pylint: enable=cell-var-from-loop

      count = jax.lax.cond(
          jnp.logical_and(jnp.logical_or(i == 0, j == 0), i**2 + j**2 <= Dc),
          _true_fun,
          lambda c: c,
          count,
      )

  # Remove axis duplicates due to equal neighbouring values.
  for dxy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
    mask = jnp.logical_and(count == 0, roll(count == 0, dxy[0], dxy[1]))
    count = jnp.where(
        jnp.logical_or(mask, roll(mask, -dxy[0], -dxy[1])), 1, count
    )

  # Apparently in the Matlab code, the indices are not allowed to be on the
  # boundary, so any index on the boundary is shifted in by one.
  ixs = jnp.reshape(jnp.arange(nx * ny), (nx, ny)) + 1
  ixs = ixs.at[0].set(ixs[0] + ny)
  ixs = ixs.at[-1].set(ixs[-1] - ny)
  ixs = ixs.at[:, 0].set(ixs[:, 0] + 1)
  ixs = ixs.at[:, -1].set(ixs[:, -1] - 1)

  # Mask out all points that are not extrema or saddles.
  xa = jnp.where(count == 0, xs, -jnp.inf)
  ya = jnp.where(count == 0, ys, -jnp.inf)
  fa = jnp.where(count == 0, fe, -jnp.inf)
  dx2fa = jnp.where(count == 0, dx2fs, -jnp.inf)
  dy2fa = jnp.where(count == 0, dy2fs, -jnp.inf)
  dxyfa = jnp.where(count == 0, dxyfs, -jnp.inf)
  ixa = jnp.where(count == 0, ixs, -jnp.inf)

  xs = jnp.where(count >= 2, xs, -jnp.inf)
  ys = jnp.where(count >= 2, ys, -jnp.inf)
  fs = jnp.where(count >= 2, fe, -jnp.inf)
  dx2fs = jnp.where(count >= 2, dx2fs, -jnp.inf)
  dy2fs = jnp.where(count >= 2, dy2fs, -jnp.inf)
  dxyfs = jnp.where(count >= 2, dxyfs, -jnp.inf)
  ixs = jnp.where(count >= 2, ixs, -jnp.inf)

  # Not actually needed, since we return masked arrays of the same size as the
  # input, but here for consistency with the original implementation.
  stat = (jnp.sum(count == 0) <= dimw) & (jnp.sum(count >= 2) <= dimw)

  return (
      xa,
      ya,
      fa,
      dx2fa,
      dy2fa,
      dxyfa,
      ixa,
      xs,
      ys,
      fs,
      dx2fs,
      dy2fs,
      dxyfs,
      ixs,
      stat,
  )
