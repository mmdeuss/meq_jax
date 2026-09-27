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

"""Limiter flux with sub-segment interpolation (meqpdom ilim=2).

Port of meq/fl4pinterp.m. Like fl4pmex it interpolates the flux on the
limiter contour and keeps only extrema of the compatible sign, but instead
of returning the value at the limiter vertex it fits a parabola through the
vertex, its neighbour and the midpoint between them, and returns that
parabola's extremum. The limiter point itself moves to where the extremum
falls, so rl and zl are outputs rather than only inputs.

fl4pinterp.m loops over the candidate vertices and needs `find` to get them,
which has a data-dependent size and cannot be traced. Here every segment's
extremum is computed and the candidates are selected with a mask, which is
equivalent because the per-candidate computation only reads its immediate
neighbours.
"""

import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import bint

# pylint: disable=invalid-name


def _max3p(
    yl: jt.Float[jt.Array, 'n'],
    yc: jt.Float[jt.Array, 'n'],
    yu: jt.Float[jt.Array, 'n'],
) -> tuple[jt.Float[jt.Array, 'n'], jt.Float[jt.Array, 'n']]:
  """Maximum of the parabola through (-1, yl), (0, yc) and (1, yu).

  Returns its position in [-1, 1] and its value, as max3p does in
  fl4pinterp.m.
  """
  hd2 = 0.5 * (yu + yl) - yc  # the quadratic coefficient
  hd1 = 0.5 * (yu - yc - hd2)  # half the linear coefficient

  # Convex or flat: the maximum is at an end point, or the centre if the two
  # ends are equal.
  xe_convex = jnp.where(yl > yu, -1.0, jnp.where(yl < yu, 1.0, 0.0))
  ye_convex = jnp.where(yl > yu, yl, jnp.where(yl < yu, yu, yc))

  # Concave: the vertex, clamped to the segment. Both branches of the select
  # are evaluated, so the denominator is replaced by a harmless value where
  # this branch is not taken.
  denom = jnp.where(hd2 < 0, hd2, -1.0)
  xv = -hd1 / denom
  xe_concave = jnp.clip(xv, -1.0, 1.0)
  ye_concave = jnp.where(
      xv > 1.0, yu, jnp.where(xv < -1.0, yl, yc + hd1 * xv)
  )

  concave = hd2 < 0
  return (
      jnp.where(concave, xe_concave, xe_convex),
      jnp.where(concave, ye_concave, ye_convex),
  )


def _segment_extremum(r0, z0, r1, z1, F0, F1, Fh, sIp):
  """Flextremum from fl4pinterp.m, over whole arrays of segments.

  Each segment runs from (r0, z0) to (r1, z1) with flux F0 and F1 at its ends
  and Fh at its midpoint. sIp flips the sign so that the plasma-side extremum
  is always a maximum.
  """
  xe, ye = _max3p(sIp * F0, sIp * Fh, sIp * F1)
  frac = 0.5 * (xe + 1.0)
  return r0 + frac * (r1 - r0), z0 + frac * (z1 - z0), sIp * ye


def fl4pinterp(
    Fx: jt.Float[jt.Array, 'nr nz'],
    kl: jt.Integer[jt.Array, 'nl'],
    cl: jt.Float[jt.Array, 'nl 4'],
    klh: jt.Integer[jt.Array, 'nl'],
    clh: jt.Float[jt.Array, 'nl 4'],
    FN: jt.Float[jt.Array, ''],
    rl: jt.Float[jt.Array, 'nl'],
    zl: jt.Float[jt.Array, 'nl'],
) -> tuple[
    jt.Float[jt.Array, 'nl'],  # rl
    jt.Float[jt.Array, 'nl'],  # zl
    jt.Float[jt.Array, 'nl'],  # Fl
    jt.Float[jt.Array, 'nl'],  # drFl
    jt.Float[jt.Array, 'nl'],  # dzFl
]:
  """Interpolates the flux on the limiter, refining the extrema.

  Args:
    Fx: Flux on the grid.
    kl: Zero-based indices of the limiter points into the flattened grid.
    cl: Bilinear weights for those points.
    klh: The same for the midpoint of each segment; klh[i] lies between
      limiter point i and point i+1.
    clh: Bilinear weights for the midpoints.
    FN: Boundary flux. Its sign gives the plasma current direction.
    rl: r of the limiter points.
    zl: z of the limiter points.

  Returns:
    rl, zl moved onto the interpolated extremum where there is one, the flux
    there, and the flux gradients at the original limiter points.
  """
  Fl = bint.bintmex(Fx, kl, cl)

  # A vertex is a candidate extremum unless its two neighbouring differences
  # have the same sign, or the curvature has the wrong sign for this plasma.
  # Same test as fl4pmex, which keeps the points this one discards.
  wrapped = jnp.insert(Fl, 0, Fl[-1])
  dFl = jnp.diff(wrapped)
  rolled = jnp.roll(dFl, -1)
  is_candidate = ~jnp.logical_or(
      dFl * rolled > 0, (dFl - rolled) * FN > 0
  )

  # Gradients at the limiter points. Note the stencil: fl4pinterp.m writes
  # these as Fx(kl+1) .. Fx(kl+nz+2) with kl one-based, which is the cell one
  # grid row in z beyond the one bintmex interpolates on (kl-1 zero-based).
  # fl4pmex uses the interpolation cell itself, so the two disagree by a row.
  # Reproduced as MEQ has it, since the point here is parity, but this looks
  # like an upstream off-by-one and is worth raising.
  flat = Fx.flatten()
  nz = Fx.shape[1]
  base = kl + 1
  drFl = (flat[base + nz] - flat[base]) * (cl[:, 1] + cl[:, 0]) + (
      flat[base + nz + 1] - flat[base + 1]
  ) * (cl[:, 3] + cl[:, 2])
  dzFl = (flat[base + 1] - flat[base]) * (cl[:, 3] + cl[:, 0]) + (
      flat[base + nz + 1] - flat[base + nz]
  ) * (cl[:, 2] + cl[:, 1])

  # The extremum of every segment, whether or not it is wanted. Segment i
  # runs from point i to point i+1, wrapping at the end.
  Flh = bint.bintmex(Fx, klh, clh)
  sIp = -jnp.sign(FN)
  seg_r, seg_z, seg_F = _segment_extremum(
      rl, zl, jnp.roll(rl, -1), jnp.roll(zl, -1),
      Fl, jnp.roll(Fl, -1), Flh, sIp,
  )

  # A candidate vertex takes whichever of its two adjoining segments has the
  # more extreme flux. The segment before vertex i is segment i-1.
  left_r, left_z, left_F = (
      jnp.roll(seg_r, 1), jnp.roll(seg_z, 1), jnp.roll(seg_F, 1)
  )
  take_left = (left_F - seg_F) * sIp > 0
  pick_r = jnp.where(take_left, left_r, seg_r)
  pick_z = jnp.where(take_left, left_z, seg_z)
  pick_F = jnp.where(take_left, left_F, seg_F)

  rl = jnp.where(is_candidate, pick_r, rl)
  zl = jnp.where(is_candidate, pick_z, zl)

  # With at least one candidate every other point is set to the boundary
  # flux, but with none fl4pinterp.m returns early and leaves Fl as
  # interpolated. That early return looks unreachable -- around a closed
  # contour the flux minimum is a candidate when FN > 0, the maximum when
  # FN < 0, and both when FN is 0 -- but it is kept so the two implementations
  # cannot diverge if that reasoning is wrong.
  Fl = jnp.where(
      jnp.any(is_candidate), jnp.where(is_candidate, pick_F, FN), Fl
  )
  return rl, zl, Fl, drFl, dzFl


# pylint: enable=invalid-name
