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

"""shapmex python implementation.

Note: we still rely on third_party/meq/shapmex.m to make sure the rB and zB
are not empty since majic does not support empty values.

IMPORTANT: Like the original C code, this implementation does not check for
division by zero.
"""

import jax
import jax.numpy as jnp
import jaxtyping as jt


# TODO(ncasagrande): Give better (pythonic) names.
# pylint: disable=invalid-name


def _find_closest(
    r_target: jnp.ndarray,
    z_target: jnp.ndarray,
    rq_contour: jnp.ndarray,
    zq_contour: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Finds the two points in a contour closest to a target point."""
  sq_dist = (rq_contour - r_target) ** 2 + (zq_contour - z_target) ** 2
  # Mask out the target point itself by setting its distance to infinity.
  sq_dist = jnp.where(sq_dist < 1e-12, jnp.inf, sq_dist)

  # Get indices of the two points with the smallest distances.
  indices = jnp.argsort(sq_dist)

  r2, z2 = rq_contour[indices[0]], zq_contour[indices[0]]
  r3, z3 = rq_contour[indices[1]], zq_contour[indices[1]]
  return r2, z2, r3, z3


def _fitcirc(
    x1: jnp.ndarray,
    y1: jnp.ndarray,
    x2: jnp.ndarray,
    y2: jnp.ndarray,
    x3: jnp.ndarray,
    y3: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Fits a circle through three points (x1,y1), (x2,y2), (x3,y3)."""
  # This solves the system of linear equations for the circle center (xc, yc)
  # derived from the standard circle equation.
  d1x = 2 * (x2 - x1)
  d1y = 2 * (y2 - y1)
  e1 = x2**2 - x1**2 + y2**2 - y1**2

  d2x = 2 * (x3 - x1)
  d2y = 2 * (y3 - y1)
  e2 = x3**2 - x1**2 + y3**2 - y1**2

  det = d1x * d2y - d1y * d2x

  def calculate_circle(
      det_val: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculates circle parameters if points are not collinear."""
    idet = 1.0 / det_val
    xc = (e1 * d2y - e2 * d1y) * idet
    yc = (e2 * d1x - e1 * d2x) * idet
    rc = jnp.sqrt((x1 - xc) ** 2 + (y1 - yc) ** 2)
    status = jnp.array(0)  # Success
    return xc, yc, rc, status

  def fallback(
      det_val: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns fallback values if points are collinear."""
    status = jnp.array(1)  # Failure
    return x1, y1, jnp.array(0.0, dtype=det_val.dtype), status

  # Use lax.cond to handle the case where points are collinear (det=0).
  # NOTE: Matlab implementation assumes jnp.abs(det) == 0.0
  is_collinear = jnp.abs(det) < 1e-13
  return jax.lax.cond(is_collinear, fallback, calculate_circle, det)


def _refine(
    r1: jnp.ndarray,
    z1: jnp.ndarray,
    rq_contour: jnp.ndarray,
    zq_contour: jnp.ndarray,
    is_lcfs: jnp.ndarray,
    rB: float,
    zB: float,
    drd: float,
    dzd: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Refines an extremum point estimate by fitting a circle or using a boundary point."""
  # Condition to replace the extremum with the boundary point (rB, zB).
  cond_replace = jnp.logical_and(
      is_lcfs,
      jnp.logical_and(
          rB > 0, jnp.logical_or(drd * (rB - r1) > 0, dzd * (zB - z1) > 0)
      ),
  )

  # Branch 1: Result if condition is true (replace with boundary).
  r1_b, z1_b = rB, zB

  # Branch 2: Result if condition is false (fit a circle).
  r2, z2, r3, z3 = _find_closest(r1, z1, rq_contour, zq_contour)
  rc, zc, dc, status = _fitcirc(r1, z1, r2, z2, r3, z3)

  # If circle fit succeeded (status=0), use the circle's extremum.
  # Otherwise, keep the original point.
  r1_f = jnp.where(status == 0, rc + drd * dc, r1)
  z1_f = jnp.where(status == 0, zc + dzd * dc, z1)

  # Combine branches using the condition.
  final_r1 = jnp.where(cond_replace, r1_b, r1_f)
  final_z1 = jnp.where(cond_replace, z1_b, z1_f)

  return final_r1, final_z1


def _process_contour(
    rq_c: jnp.ndarray,
    zq_c: jnp.ndarray,
    is_lcfs: jnp.ndarray,
    rB: float,
    zB: float,
) -> tuple[jnp.ndarray, ...]:
  """Processes a single contour to find its shape parameters."""
  # Find initial extremum points.
  idx_rmax = jnp.argmax(rq_c)
  idx_rmin = jnp.argmin(rq_c)
  idx_zmax = jnp.argmax(zq_c)
  idx_zmin = jnp.argmin(zq_c)

  common_kwargs = {
      'rq_contour': rq_c,
      'zq_contour': zq_c,
      'is_lcfs': is_lcfs,
      'rB': rB,
      'zB': zB,
  }

  # Refine the four extremum points.
  rrmax, zrmax = _refine(
      r1=rq_c[idx_rmax], z1=zq_c[idx_rmax], drd=1.0, dzd=0.0, **common_kwargs
  )
  rrmin, zrmin = _refine(
      r1=rq_c[idx_rmin], z1=zq_c[idx_rmin], drd=-1.0, dzd=0.0, **common_kwargs
  )
  rzmax, zzmax = _refine(
      r1=rq_c[idx_zmax], z1=zq_c[idx_zmax], drd=0.0, dzd=1.0, **common_kwargs
  )
  rzmin, zzmin = _refine(
      r1=rq_c[idx_zmin], z1=zq_c[idx_zmin], drd=0.0, dzd=-1.0, **common_kwargs
  )

  # Calculate final shape parameters from refined points.
  rg = (rrmax + rrmin) * 0.5
  am = (rrmax - rrmin) * 0.5

  du = (rg - rzmin) / am
  dl = (rg - rzmax) / am

  rgeom = rg
  zgeom = (zzmax + zzmin) * 0.5
  amino = am
  epsil = am / rg
  kappa = (zzmax - zzmin) / (rrmax - rrmin)
  deltl = du  # Lower triangularity
  deltu = dl  # Upper triangularity
  delta = (du + dl) * 0.5

  return (
      rgeom,
      zgeom,
      amino,
      epsil,
      kappa,
      delta,
      deltl,
      deltu,
      rrmax,
      zrmax,
      rrmin,
      zrmin,
      rzmax,
      zzmax,
      rzmin,
      zzmin,
  )


def shap(
    rq: jt.Float[jt.Array, 'npq noq'],
    zq: jt.Float[jt.Array, 'npq noq'],
    rB: jt.Float[jt.Array, ''],
    zB: jt.Float[jt.Array, ''],
    rA: jt.Float[jt.Array, ''],
    zA: jt.Float[jt.Array, ''],
    dr2fa: jt.Float[jt.Array, ''],
    dz2fa: jt.Float[jt.Array, ''],
):
  """Calculates plasma shape parameters from flux surface contours.

  This is a JAX implementation of the `shamex` C-MEX function.

  Note on Data Layout:
  This function expects `rq` and `zq` to be row-major arrays with shape
  `(npq, noq)`, where `npq` is the number of contours and `noq` is the
  number of points per contour. This is the standard in JAX/NumPy.

  Args:
      rq: R-coordinates of contours, shape (npq, noq).
      zq: Z-coordinates of contours, shape (npq, noq).
      rB: R-coordinate of boundary point. Use a value < 0 if not applicable.
      zB: Z-coordinate of boundary point. Use a value < 0 if not applicable.
      rA: R-coordinate of the magnetic axis.
      zA: Z-coordinate of the magnetic axis.
      dr2fa: Second radial derivative of flux at the axis.
      dz2fa: Second vertical derivative of flux at the axis.

  Returns:
      A tuple of 16 arrays, each of shape (npq + 1,), containing the
      shape parameters. The first element of each array corresponds to
      the magnetic axis, and the subsequent elements correspond to each contour.
  """
  npq = rq.shape[0]

  # Part 1: Calculate values at the magnetic axis (output index 0)
  axis_vals = {
      'rgeom': rA,
      'zgeom': zA,
      'amino': 0.0,
      'epsil': 0.0,
      'kappa': jnp.sqrt(dr2fa / dz2fa),
      'delta': 0.0,
      'deltl': 0.0,
      'deltu': 0.0,
      'rrmax': rA,
      'zrmax': zA,
      'rrmin': rA,
      'zrmin': zA,
      'rzmax': rA,
      'zzmax': zA,
      'rzmin': rA,
      'zzmin': zA,
  }

  # Part 2: Vectorize the contour processing over all contours
  # Create a boolean array to identify the last contour (LCFS).
  is_lcfs_array = jnp.arange(npq) == npq - 1

  # `vmap` the processing function. `in_axes` specifies how to map arguments:
  # 0: map over the first axis of rq, zq, is_lcfs_array
  # None: broadcast the scalar values rB, zB
  vmapped_processor = jax.vmap(_process_contour, in_axes=(0, 0, 0, None, None))

  contour_results = vmapped_processor(rq, zq, is_lcfs_array, rB, zB)

  # Part 3: Concatenate axis and contour results for final output
  output_names = [
      'rgeom',
      'zgeom',
      'amino',
      'epsil',
      'kappa',
      'delta',
      'deltl',
      'deltu',
      'rrmax',
      'zrmax',
      'rrmin',
      'zrmin',
      'rzmax',
      'zzmax',
      'rzmin',
      'zzmin',
  ]

  final_outputs = tuple(
      jnp.concatenate([jnp.array([axis_vals[name]]), contour_vals])
      for name, contour_vals in zip(output_names, contour_results)
  )

  return final_outputs


# pylint: enable=invalid-name
