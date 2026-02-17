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

"""rtciplasma jax implementation.

This replaces the Matlab implementation in
  third_party/meq/rtciplasma.m
"""

import math

import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import find_contours
from meq_jax._src import types

eps = jnp.finfo(float).eps


# pylint: disable=invalid-name
def rtciplasma(L: types.StaticData, LY: types.OutputData):
  """Computes flux surfaces from equilibrium in LY.

  Computes flux surface locations inside the plasma. Internally utilizes
  find_contours and implements special a special case for doublets with
  mantle

  Arguments:
    L: standard L structure
    LY: standard LY structure containing converged GS-equilibrium

  Returns:
    rq: r-positions of discretized flux surfaces
    zq: z-positions of discretized flux surfaces
    aq: radial positions (how far away from origin points) for the discretized
    flux surfaces
    rO: r-positions of constant-theta ray origins
    zO: z-positions of constant-theta ray origins
    crq: r-directions of constant-theta rays
    czq: z-directions of constant-theta rays
    nDq: int32, number of domains for which flux surfaces are computed, nDq <=
      nD
  """
  assert L.G is not None

  # nAeff = min([LY.nA, LY.nB, L.nD])
  # Note: because LY.nA and LY.nB are not static, we deviate from the Matlab
  # implementation here and set nDeff = L.nD
  nAeff = L.nD
  nDeff = nAeff
  is_doublet = L.shot == 82 or L.shot == 88

  rq = jnp.zeros((L.nD, L.npq, L.noq))  # positions of points on flux surfaces
  zq = jnp.zeros_like(rq)
  rO = jnp.zeros((L.nD, L.noq))  # origin points
  zO = jnp.zeros_like(rO)
  crq = jnp.zeros_like(rO)  # constant-theta ray directions
  czq = jnp.zeros_like(rO)

  assert LY.rA is not None
  assert LY.FA is not None
  assert LY.zA is not None
  assert LY.rA.ndim == 1
  assert LY.FA.ndim == 1
  assert LY.zA.ndim == 1
  # LY.rA etc. will have size L.dimw, which is an upper bound on L.nD
  rA = LY.rA[:L.nD]
  FA = LY.FA[:L.nD]
  zA = LY.zA[:L.nD]

  aq = LY.aq
  if aq is None:  # aq init
    aq = rq

    for iD in range(nAeff):
      bbox_size = jnp.min(
          jnp.array([
              rA[iD] - L.G.rx[0],
              L.G.rx[-1] - rA[iD],
              zA[iD] - L.G.zx[0],
              L.G.zx[-1] - zA[iD],
          ])
      )
      aq_iD = jnp.broadcast_to(
          bbox_size * L.pinit * L.pq[..., None], (L.npq, L.noq)
      )
      aq = aq.at[iD].set(aq_iD)

  if aq.ndim == 2:
    assert nAeff == 1
    aq = aq[None]

  FQ = LY.F1[..., None] - L.fq[None] * (LY.F1 - LY.F0)[..., None]

  # All axis domains
  for iD in range(nAeff):
    rq_iD, zq_iD, aq_iD = find_contours.find_contours(
        L=L,
        LY=LY,
        F=FQ[iD],
        aq=aq[iD],
        rO=rA[iD][None],
        zO=zA[iD][None],
        crq=L.crq,
        czq=L.czq,
        Fo=FA[iD][None],
        Opo=jnp.array([iD + 1]),
    )
    rq = rq.at[iD].set(rq_iD)
    zq = zq.at[iD].set(zq_iD)
    aq = aq.at[iD].set(aq_iD)
    # For domains with axes, the axis is always the origin point
    rO = rO.at[iD].set(rA[iD])
    zO = zO.at[iD].set(zA[iD])
    # For domains with axes, the ray directions are predefined
    assert L.crq is not None
    assert L.czq is not None
    crq = crq.at[iD].set(L.crq.ravel())
    czq = czq.at[iD].set(L.czq.ravel())

  if is_doublet:
    # A special version of the flux surface finding algorithm has to be
    # called for the mantle domain which does not have an axis
    rq_id, zq_id, aq_id, rO_id, zO_id, cr_id, cz_id = mantle_domain_contours(
        L=L, LY=LY, rq=rq, zq=zq, F=FQ[2]
    )
    rq = rq.at[2].set(rq_id)
    zq = zq.at[2].set(zq_id)
    aq = aq.at[2].set(aq_id)
    rO = rO.at[2].set(rO_id)
    zO = zO.at[2].set(zO_id)
    crq = crq.at[2].set(cr_id)
    czq = czq.at[2].set(cz_id)
    nDeff = L.nD
  return rq, zq, aq, rO, zO, crq, czq, nDeff


def mantle_domain_contours(
    L: types.StaticData,
    LY: types.OutputData,
    rq: jt.Float[jt.Array, 'nD npq noq'],
    zq: jt.Float[jt.Array, 'nD npq noq'],
    F: jt.Float[jt.Array, 'nD'],
) -> tuple[
    jt.Float[jt.Array, 'npq noq'],
    jt.Float[jt.Array, 'npq noq'],
    jt.Float[jt.Array, 'npq noq'],
    jt.Float[jt.Array, 'noq'],
    jt.Float[jt.Array, 'noq'],
    jt.Float[jt.Array, 'noq'],
    jt.Float[jt.Array, 'noq'],
]:
  """Returns the flux surface contours for the mantle domain.

  For the mantle, constant-theta-ray directions have to be adapted to the
  geometry and are always different and thus a return. For the mantle, the
  origin points for the constant-theta rays form the separatrix

  Arguments:
    L: struct, standard L structure
    LY: struct, standard LY structure containing converged GS-equilibrium

  Returns:
    rm: r-positions of discretized flux surfaces
    zm: z-positions of discretized flux surfaces
    am: radial positions (how far away from origin points) for the discretized
    flux surfaces
    rO: r-positions of separatrix points
    zO: z-positions of separatrix points
    cr: r-directions of constant-theta rays
    cz: z-directions of constant-theta rays
  """
  # Separatrix r and z positions are taken from the LCFS of each lobe
  rS = jnp.zeros((2, L.noq))
  zS = jnp.zeros((2, L.noq))

  for iD in range(2):
    # For each lobe, the points have to be ordered according to their angle
    # w.r.t. the X-point such that they can be combined from the two lobes

    rS_ordered, zS_ordered = x_point_ordering(
        rq[iD, -1, :],
        zq[iD, -1, :],
        LY.rX[0],
        LY.zX[0],
        LY.rA[iD],
        LY.zA[iD],
    )
    rS = rS.at[iD, :].set(rS_ordered)
    zS = zS.at[iD, :].set(zS_ordered)

  # Concatenating and subsampling to get back to L.noq points
  rO = rS.ravel()
  rO = rO[0::2]
  zO = zS.ravel()
  zO = zO[0::2]

  # Compute good directions for the rays to have good sampling around X-point
  assert LY.rA is not None
  assert LY.zA is not None
  cr, cz = get_mantle_directions(
      rO=rO,
      zO=zO,
      rX=LY.rX[0],
      zX=LY.zX[0],
      rA=LY.rA,
      zA=LY.zA,
  )

  am = jnp.zeros((L.npq, L.noq))
  FO = jnp.broadcast_to(LY.FB[0], (L.noq,))
  rm, zm, am = find_contours.find_contours(
      L=L,
      LY=LY,
      F=F,
      aq=am,
      rO=rO,
      zO=zO,
      crq=cr,
      czq=cz,
      Fo=FO,
      Opo=jnp.array([3]),
  )
  return rm, zm, am, rO, zO, cr, cz


def get_mantle_directions(
    rO: jt.Float[jt.Array, 'noq'],
    zO: jt.Float[jt.Array, 'noq'],
    rX: jt.Float[jt.Array, ''],
    zX: jt.Float[jt.Array, ''],
    rA: jt.Float[jt.Array, '2'],
    zA: jt.Float[jt.Array, '2'],
) -> tuple[jt.Float[jt.Array, 'noq'], jt.Float[jt.Array, 'noq']]:
  """Computes constant-theta ray directions for a doublet.

  For the top of the top lobe and the bottom of the bottom lobe, ray
  directions are still originating from the axis. For the region around the
  X-point, ray directions go towards a fictitious point to the right (P_RHS)
  and the left of the doublet (P_LHS).

  Arguments:
    rO: r-positions of separatrix points
    zO: z-positions of separatrix points
    rX: r-position of central X point
    zX: z-position of central X point
    rA: r-positions of two axes
    zA: z-positions of two axes

  Returns:
    cr: r-directions of constant-theta rays
    cz: z-directions of constant-theta rays
  """
  # RHS and LHS points are on a plane orthogonal to the A1-A2 vector centered
  # at the X point and as far out as the distance between A1 and A2
  rRHS = rX + (zA[0] - zA[1])
  zRHS = zX - (rA[0] - rA[1])
  rLHS = rX - (zA[0] - zA[1])
  zLHS = zX + (rA[0] - rA[1])

  # Mask if a point on the separatrix should use the upper lobe axis as
  # origin or the RHS or LHS points (the lower lobe axis as origin is used if
  # all of these are not true)
  maskA1 = (rO - rX) * (rA[0] - rA[1]) + (zO - zX) * (
      zA[0] - zA[1]
  ) > 0  # upper lobe
  # right side of doublet
  maskRHS = inpolygon(
      rO,
      zO,
      jnp.array([rA[0], rX, rA[1], rRHS]),
      jnp.array([zA[0], zX, zA[1], zRHS]),
  )
  # left side of doublet
  maskLHS = inpolygon(
      rO,
      zO,
      jnp.array([rA[0], rX, rA[1], rLHS]),
      jnp.array([zA[0], zX, zA[1], zLHS]),
  )

  # Add axis origin rays according to mask
  cr = (rO - rA[0]) * maskA1 + (rO - rA[1]) * ~maskA1
  cz = (zO - zA[0]) * maskA1 + (zO - zA[1]) * ~maskA1

  # Add RHS and LHS origin rays acoording to mask
  cr = jnp.where(maskRHS, rRHS - rO, cr)
  cz = jnp.where(maskRHS, zRHS - zO, cz)
  cr = jnp.where(maskLHS, rLHS - rO, cr)
  cz = jnp.where(maskLHS, zLHS - zO, cz)

  # Norm ray directions
  c_norm = jnp.sqrt(cr * cr + cz * cz)
  cr /= c_norm
  cz /= c_norm
  return cr, cz


def inpolygon(
    xq: jt.Float[jt.Array, 'nq'],
    yq: jt.Float[jt.Array, 'nq'],
    xv: jt.Float[jt.Array, 'nr'],
    yv: jt.Float[jt.Array, 'nr'],
):
  """JAX implementation of MATLAB's inpolygon (inside OR on boundary).

  Implements the ray casting algorithm and a separate boundary check,
  vectorized over all query points and polygon edges.

  Args:
      xq: X-coordinates of query points. Can be any shape.
      yq: Y-coordinates of query points. Must match xq shape.
      xv: X-coordinates of polygon vertices (ordered).
      yv: Y-coordinates of polygon vertices (ordered).

  Returns:
      Array of the same shape as xq/yq.
      True if the point is inside or on the boundary.
  """

  # 1. Store original shape and flatten query points
  original_shape = xq.shape
  qx = xq.reshape(-1)  # Shape (M,)
  qy = yq.reshape(-1)  # Shape (M,)

  # 2. Get polygon edges (start and end points)
  # xv_start, yv_start are the original (xv, yv)
  # xv_end, yv_end are the "next" vertex (wrapping around)
  xv_end = jnp.roll(xv, -1)  # Shape (N,)
  yv_end = jnp.roll(yv, -1)  # Shape (N,)

  # 3. Broadcast all arrays for (M, N) comparison
  # Query points broadcast from (M, 1)
  # Vertex points broadcast from (1, N)
  qx_b = qx[:, None]
  qy_b = qy[:, None]

  xv_start_b = xv[None, :]
  yv_start_b = yv[None, :]
  xv_end_b = xv_end[None, :]
  yv_end_b = yv_end[None, :]

  # Part A: Ray Casting Algorithm (Finds points STRICTLY INSIDE)

  # Check if the ray (qy) crosses the y-span of the edge.
  # We use the standard robust rule: (start_y <= qy < end_y)
  # OR (end_y <= qy < start_y)
  # This correctly handles vertices.
  c1 = (yv_start_b <= qy_b) & (yv_end_b > qy_b)
  c2 = (yv_end_b <= qy_b) & (yv_start_b > qy_b)
  crosses_y_line = c1 | c2

  # Calculate the x-coordinate of the intersection of the horizontal ray with
  # the edge x_intersect = (vx_end - vx_start) * (qy - vy_start) /
  # (vy_end - vy_start) + vx_start
  # We only care about divisions where crosses_y_line is True, so denom != 0.
  # Where denom is 0 (horizontal line), crosses_y_line is False,
  # so NaNs are ignored.
  denom = yv_end_b - yv_start_b
  x_intersect = (xv_end_b - xv_start_b) * (
      qy_b - yv_start_b
  ) / denom + xv_start_b

  # A valid "crossing" happens if:
  # 1. The edge crosses the ray's y-coordinate (crosses_y_line)
  # 2. The intersection point is to the RIGHT of the query point
  # (qx < x_intersect)
  intersections = crosses_y_line & (qx_b < x_intersect)

  # Count intersections (sum along the edge axis)
  num_intersections = jnp.sum(intersections, axis=1)  # Shape (M,)

  # Apply the Even-Odd Rule: Odd number of intersections means inside.
  strictly_inside = (num_intersections % 2) == 1

  # Part B: Boundary Check (Finds points exactly ON THE EDGE)

  # A point is on a segment if it is (1) collinear AND (2) within the segment's
  # bounding box.

  # 1. Collinearity check (cross product == 0)
  # (dx_edge * dy_query) == (dy_edge * dx_query)
  # Using jnp.isclose for float precision.
  dx_edge = xv_end_b - xv_start_b
  dy_edge = yv_end_b - yv_start_b
  dx_query = qx_b - xv_start_b
  dy_query = qy_b - yv_start_b

  cross_product = (dx_edge * dy_query) - (dy_edge * dx_query)
  is_collinear = jnp.isclose(cross_product, 0.0)

  # 2. Bounding box check (point must be between the segment endpoints)
  within_x = (jnp.minimum(xv_start_b, xv_end_b) <= qx_b) & (
      qx_b <= jnp.maximum(xv_start_b, xv_end_b)
  )
  within_y = (jnp.minimum(yv_start_b, yv_end_b) <= qy_b) & (
      qy_b <= jnp.maximum(yv_start_b, yv_end_b)
  )

  on_segment = is_collinear & within_x & within_y

  # Check if the point is on *any* segment
  on_boundary = jnp.any(on_segment, axis=1)  # Shape (M,)

  # Final Result (Inside OR On Boundary)
  result = strictly_inside | on_boundary

  # Reshape result to match original query input shape
  return result.reshape(original_shape)


def x_point_ordering(
    rq: jt.Float[jt.Array, 'n'],
    zq: jt.Float[jt.Array, 'n'],
    rX: jt.Float[jt.Array, ''],
    zX: jt.Float[jt.Array, ''],
    rA: jt.Float[jt.Array, ''],
    zA: jt.Float[jt.Array, ''],
):
  """Order flux surface points by angle relative to the X-point angle.

  Arguments:
    rq: double(k), r-positions of flux surface points, flattened
    zq: double(k), z-positions of flux surface points, flattened
    rX: double, r-position of X point
    zX: double, z-position of X point
    rA: double, r-position of axis
    zA: double, z-position of axis

  Returns:
    LY:   struct, input LY structure equipped with rS and zS
  """
  angle = jnp.atan2(rq - rA, zq - zA)
  angleX = jnp.atan2(rX - rA, zX - zA)
  # have to add epsilon here for the case the X-point itself is part of rq,zq
  order = jnp.argsort(jnp.mod(angle - angleX + jnp.sqrt(eps), 2 * math.pi))
  rq_ordered = rq[order]
  zq_ordered = zq[order]
  return rq_ordered, zq_ordered


# pylint: enable=invalid-name
