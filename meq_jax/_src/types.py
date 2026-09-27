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

"""Define Python dataclasses for MEQ structs."""

# We only need to add the minimal subset of fields that are used in the
# functions we're writing. Fields can be added incrementally as needed.

from dataclasses import dataclass, field  # pylint: disable=g-importing-member, g-multiple-import
from typing import Any, Tuple

import chex
import jax.tree_util


# pylint: disable=invalid-name


INT_FIELDS = [
    'nP', 'nT', 'dimw', 'domain_id', 'gsxe', 'ilackner', 'ilim',
    'index', 'itercs', 'iterq', 'nA', 'nB', 'nC', 'nD', 'nFW', 'nN', 'nQ',
    'nR', 'nS', 'nW', 'nX', 'na', 'naR', 'ne', 'ng', 'nh', 'nl', 'noq', 'np',
    'nbf', 'npq', 'nrD', 'nrx', 'nry', 'nrz', 'nx', 'ny', 'nzx', 'nzy', 'nzz',
    'shot', 'nnoc', 'it',
]


BOOL_FIELDS = [
    'dodisp', 'dojacF', 'dojacu', 'dojacx', 'dojacxdot', 'doplot', 'dopost',
    'icsint', 'ifield', 'ihole', 'iLpext', 'isEvolutive', 'isaddl',
    'ivacuum', 'izgrid', 'izxoverlap', 'liurtemu', 'nn', 'smalldia',
]


STR_FIELDS = [
    'algoNL', 'bfct', 'cde', 'code', 'fun_name', 'lx_name', 'tokamak',
    'initguess', 'infct', 'eqfct',
]


@jax.tree_util.register_dataclass
@dataclass
class ParameterData:
  """Equivalent to P struct in MEQ."""

  algoNL: str | None = field(default=None, metadata=dict(static=True))
  r0: float | None = field(default=None, metadata=dict(static=True))
  b0: float | None = field(default=None, metadata=dict(static=True))
  itercs: int | None = field(default=None, metadata=dict(static=True))
  tolcs: float | None = field(default=None, metadata=dict(static=True))
  iterq: int | None = field(default=None, metadata=dict(static=True))
  tolq: float | None = field(default=None, metadata=dict(static=True))
  ilackner: int | None = field(default=None, metadata=dict(static=True))
  ilim: int | None = field(default=None, metadata=dict(static=True))
  icsint: bool | None = field(default=False, metadata=dict(static=True))
  tokamak: str | None = field(default=None, metadata=dict(static=True))
  Ipmin: float | None = field(default=None, metadata=dict(static=True))
  nFW: int | None = field(default=None, metadata=dict(static=True))
  dasm: float | None = field(default=None, metadata=dict(static=True))
  ihole: bool | None = field(default=None, metadata=dict(static=True))
  xdoma: float | None = field(default=None, metadata=dict(static=True))
  iLpext: bool | None = field(default=None, metadata=dict(static=True))
  naR: int | None = field(default=None, metadata=dict(static=True))
  iqR: chex.Array | None = None
  raS: chex.Array | None = None
  gsxe: int | None = field(default=None, metadata=dict(static=True))
  ifield: bool | None = field(default=False, metadata=dict(static=True))
  ivacuum: bool | None = field(default=False, metadata=dict(static=True))
  izgrid: bool | None = field(default=False, metadata=dict(static=True))
  izxoverlap: bool | None = field(default=False, metadata=dict(static=True))
  infct: Any | None = None
  rn: chex.Array | None = None
  zn: chex.Array | None = None
  isaddl: bool | None = field(default=None, metadata=dict(static=True))
  cde: str | None = field(default=None, metadata=dict(static=True))
  initguess: tuple[str, ...] | None = field(
      default=None, metadata=dict(static=True))
  tolF: float | None = field(default=None, metadata=dict(static=True))
  tolok: float | None = field(default=None, metadata=dict(static=True))
  keepok: bool | None = field(default=None, metadata=dict(static=True))
  eqfct: str | None = field(default=None, metadata=dict(static=True))
  nnoc: int | None = field(default=None, metadata=dict(static=True))


@jax.tree_util.register_dataclass
@dataclass
class GeometryData:
  """Equivalent to G struct in MEQ."""

  na: int | None = field(metadata=dict(static=True), default=None)
  rl: chex.Array | None = None
  zl: chex.Array | None = None
  nl: int | None = field(metadata=dict(static=True), default=None)
  rx: chex.Array | None = None
  zx: chex.Array | None = None
  rW: chex.Array | None = None
  zW: chex.Array | None = None
  nW: int | None = field(metadata=dict(static=True), default=None)
  aW: chex.Array | None = None
  Bma: chex.Array | None = None  # 2D
  Bmx: chex.Array | None = None  # 2D
  Bmu: chex.Array | None = None  # 2D
  Mfa: chex.Array | None = None  # 2D
  Mfx: chex.Array | None = None  # 2D
  Mfu: chex.Array | None = None  # 2D
  Tvu: chex.Array | None = None  # 2D
  Tius: chex.Array | None = None
  Vadelay: chex.Array | None = None
  Vamax: chex.Array | None = None
  Vamin: chex.Array | None = None
  Iamax: chex.Array | None = None
  Iamin: chex.Array | None = None
  Talim: chex.Array | None = None
  Mza: chex.Array | None = None  # 2D
  Mzu: chex.Array | None = None  # 2D
  Mzy: chex.Array | None = None  # 2D
  lzx: chex.Array | None = None  # 2D
  Ceq: chex.Array | None = None
  Seq: chex.Array | None = None
  Ihig: chex.Array | None = None
  Ilow: chex.Array | None = None


@jax.tree_util.register_dataclass
@dataclass
class IndexData:
  """Equivalent to L.ind struct in MEQ."""

  ixGS: chex.Array | None = None
  ixg: chex.Array | None = None
  ixa: chex.Array | None = None
  ixu: chex.Array | None = None
  irGS: chex.Array | None = None
  irC: chex.Array | None = None
  irp: chex.Array | None = None
  ira: chex.Array | None = None
  iru: chex.Array | None = None
  iua: chex.Array | None = None
  iuu: chex.Array | None = None
  iuC: chex.Array | None = None
  iuni: chex.Array | None = None
  iurp: chex.Array | None = None


@jax.tree_util.register_dataclass
@dataclass
class BfpData:
  """Equivalent to L.bfp struct in MEQ."""

  shot: int | None = field(default=None, metadata=dict(static=True))
  # For doublet
  ngi_list: Tuple[int, ...] | None = field(
      default=None, metadata=dict(static=True))
  bfp_list: Tuple[Tuple[int, int], ...] | None = field(
      default=None, metadata=dict(static=True))
  iDbf_list: Tuple[Tuple[int, ...], ...] | None = field(
      default=None, metadata=dict(static=True))
  ng: int | None = field(default=None, metadata=dict(static=True))
  nbf: int | None = field(default=None, metadata=dict(static=True))
  nD: int | None = field(default=None, metadata=dict(static=True))
  igm: chex.Array | None = None
  fPg3: chex.Array | None = None
  fTg3: chex.Array | None = None
  # For single-domain
  nP: int | None = field(default=None, metadata=dict(static=True))
  nT: int | None = field(default=None, metadata=dict(static=True))


@chex.dataclass
class Opts:
  """Options for fgeF."""

  dojacx: bool = field(default=False, metadata=dict(static=True))
  dojacu: bool = field(default=False, metadata=dict(static=True))
  dojacxdot: bool = field(default=False, metadata=dict(static=True))
  dojacF: bool = field(default=False, metadata=dict(static=True))
  dopost: bool = field(default=False, metadata=dict(static=True))
  doplot: bool = field(default=False, metadata=dict(static=True))
  dodisp: bool = field(default=False, metadata=dict(static=True))


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclass
class StaticData:
  """Equivalent to L struct in MEQ."""

  G: GeometryData | None = None
  P: ParameterData | None = None
  bfp: BfpData | None = None
  FN: chex.Array | None = None
  taul: chex.Array | None = None
  taur: chex.Array | None = None
  tauz: chex.Array | None = None
  Mr: chex.Array | None = None
  Mz: chex.Array | None = None
  Ml: chex.Array | None = None
  Tbc: chex.Array | None = None  # [2 * (nz2 + nr2)]
  Mby: chex.Array | None = None  # [2 * (nz2 + nr2 + 2), (nz2 * nr2) = ny]
  cx: chex.Array | None = None  # [nr2]
  cq: chex.Array | None = None  # [nz2, nr2]
  cr: chex.Array | None = None  # [nz2, nr2]
  cs: chex.Array | None = None  # [nz2, nr2]
  ci: float | None = field(default=None, metadata=dict(static=True))
  co: float | None = field(default=None, metadata=dict(static=True))
  # Optional dense operators materializing the gszr Poisson solve
  # (see gszr.gszr_operator). When set, meqfx/meqfbp apply the solve as a
  # matmul, which is much faster on accelerators than the sequential
  # cyclic-reduction algorithm in gszrjax.
  gszr_bc_op: chex.Array | None = None  # [2(nz2+nr2+2), nrx*nzx]
  gszr_iy_op: chex.Array | None = None  # [nry*nzy, nrx*nzx]
  nx: int | None = field(default=None, metadata=dict(static=True))
  na: int | None = field(default=None, metadata=dict(static=True))
  nrx: int | None = field(default=None, metadata=dict(static=True))
  nzx: int | None = field(default=None, metadata=dict(static=True))
  nry: int | None = field(default=None, metadata=dict(static=True))
  nzy: int | None = field(default=None, metadata=dict(static=True))
  nzz: int | None = field(default=None, metadata=dict(static=True))
  nrz: int | None = field(default=None, metadata=dict(static=True))
  idrx: float | None = field(default=None, metadata=dict(static=True))
  idzx: float | None = field(default=None, metadata=dict(static=True))
  idsx: float | None = field(default=None, metadata=dict(static=True))
  dsx: float | None = field(default=None, metadata=dict(static=True))
  drx: float | None = field(default=None, metadata=dict(static=True))
  dzx: float | None = field(default=None, metadata=dict(static=True))
  liurtemu: bool | None = field(default=False, metadata=dict(static=True))
  crq: chex.Array | None = None
  czq: chex.Array | None = None
  crW: chex.Array | None = None
  czW: chex.Array | None = None
  kxW: chex.Array | None = None
  cWx: chex.Array | None = None  # 2D
  nW: int | None = field(default=None, metadata=dict(static=True))
  ny: int | None = field(default=None, metadata=dict(static=True))
  ne: int | None = field(default=None, metadata=dict(static=True))
  fPg: chex.Array | None = None
  fTg: chex.Array | None = None
  TDg: chex.Array | None = None  # 2D
  Ip0: float | None = field(default=None, metadata=dict(static=True))
  ng: int | None = field(default=None, metadata=dict(static=True))
  nD: int | None = field(default=None, metadata=dict(static=True))
  nQ: int | None = field(default=None, metadata=dict(static=True))
  noq: int | None = field(default=None, metadata=dict(static=True))
  npq: int | None = field(default=None, metadata=dict(static=True))
  nR: int | None = field(default=None, metadata=dict(static=True))
  nS: int | None = field(default=None, metadata=dict(static=True))
  raN: float | None = field(default=None, metadata=dict(static=True))
  ry: chex.Array | None = None
  iry: chex.Array | None = None
  rry: chex.Array | None = None  # 2D
  zzy: chex.Array | None = None
  xscal: chex.Array | None = None
  ind: IndexData | None = None
  pinit: float | None = field(default=None, metadata=dict(static=True))
  pq: chex.Array | None = None
  pQ: chex.Array | None = None
  fq: chex.Array | None = None
  Oasx: chex.Array | None = None
  dimw: int | None = field(default=None, metadata=dict(static=True))
  kxl: chex.Array | None = None
  clx: chex.Array | None = None
  kxlh: chex.Array | None = None
  clhx: chex.Array | None = None
  lxy: chex.Array | None = None  # 2D
  Oly: chex.Array | None = None
  M1q: chex.Array | None = None  # 2D
  M2q: chex.Array | None = None  # 2D
  M3q: chex.Array | None = None  # 2D
  doq: float | None = field(default=None, metadata=dict(static=True))
  Mye: chex.Array | None = None
  Myy: chex.Array | None = None
  doq: float | None = field(default=None, metadata=dict(static=True))
  i95: chex.Array | None = None
  c95: chex.Array | None = None
  Mbe: chex.Array | None = None  # 2D
  Tye: chex.Array | None = None  # 2D
  Mxy: chex.Array | None = None  # 2D
  Mxe: chex.Array | None = None  # 2D
  Mbh: chex.Array | None = None
  dzMbe: chex.Array | None = None
  nN: int | None = field(default=None, metadata=dict(static=True))
  isEvolutive: bool | None = field(default=None, metadata=dict(static=True))
  resscal: chex.Array | None = None
  nh: int | None = field(default=None, metadata=dict(static=True))
  dlst: chex.Array | None = None
  rhsf: chex.Array | None = None
  Txy: chex.Array | None = None
  Txy_sum: chex.Array | None = None
  Txb: chex.Array | None = None
  Re: chex.Array | None = None
  Mee: chex.Array | None = None
  Mey: chex.Array | None = None
  nrD: int | None = field(default=None, metadata=dict(static=True))
  nC: int | None = field(default=None, metadata=dict(static=True))
  np: int | None = field(default=None, metadata=dict(static=True))
  Jx: chex.Array | None = None
  Ju: chex.Array | None = None
  Jxdot: chex.Array | None = None
  fPg: chex.Array | None = None
  fTg: chex.Array | None = None
  i4pirxdzx: chex.Array | None = None
  i4pirxdrx: chex.Array | None = None
  i4pirzdzz: chex.Array | None = None
  i4pirzdrz: chex.Array | None = None
  smalldia: bool | None = field(default=None, metadata=dict(static=True))
  nn: bool | None = field(default=None, metadata=dict(static=True))
  rx: chex.Array | None = None
  zx: chex.Array | None = None
  inM: float | None = field(default=None, metadata=dict(static=True))
  bfct: str | None = field(default=None, metadata=dict(static=True))
  FN: float | None = field(default=None, metadata=dict(static=True))
  code: str | None = field(default=None, metadata=dict(static=True))
  # Static argument from LY
  shot: int | None = field(default=None, metadata=dict(static=True))


@jax.tree_util.register_dataclass
@dataclass
class InputData:
  """Equivalent to LX (fge) struct in MEQ."""

  ag: chex.Array | None = None
  bp: chex.Array | None = None
  bpD: chex.Array | None = None
  Ip: chex.Array | None = None
  IpD: chex.Array | None = None
  li: float | None = None
  qA: chex.Array | None = None
  rBt: float | None = None
  Wk: chex.Array | None = None
  WkD: chex.Array | None = None
  Lp: chex.Array | None = None
  Rp: chex.Array | None = None
  Ini: chex.Array | None = None
  IniD: chex.Array | None = None
  Va: chex.Array | None = None
  Ia: chex.Array | None = None
  Iu: chex.Array | None = None
  Iy: chex.Array | None = None
  Fx: chex.Array | None = None
  t: float | None = None
  shot: int | None = None
  aq: chex.Array | None = None
  aW: chex.Array | None = None
  signeo: chex.Array | None = None


@jax.tree_util.register_dataclass
@dataclass
class OutputData:
  """Equivalent to LY struct in MEQ."""

  FA: chex.Array | None = None
  FB: chex.Array | None = None
  FX: chex.Array | None = None
  Fx: chex.Array | None = None
  F0: chex.Array | None = None
  F1: chex.Array | None = None
  lB: chex.Array | None = None
  lX: chex.Array | None = None
  # not sure why this ends up as an array in oct2py
  nX: chex.Array | int | None = None
  Opy: chex.Array | None = None  # 2D
  t: chex.Array | None = None
  shot: int | None = None
  aq: chex.Array | None = None
  ag: chex.Array | None = None
  aW: chex.Array | None = None
  nA: int | None = False
  nB: int | None = False
  Opy: chex.Array | None = None
  rA: chex.Array | None = None
  zA: chex.Array | None = None
  rB: chex.Array | None = None
  zB: chex.Array | None = None
  rX: chex.Array | None = None
  zX: chex.Array | None = None
  dr2FA: chex.Array | None = None
  dz2FA: chex.Array | None = None
  drzFA: chex.Array | None = None
  dr2FX: chex.Array | None = None
  dz2FX: chex.Array | None = None
  drzFX: chex.Array | None = None
  iTQ: chex.Array | None = None
  PpQ: chex.Array | None = None
  TTpQ: chex.Array | None = None
  TQ: chex.Array | None = None
  IpD: chex.Array | None = None
  Iy: chex.Array | None = None
  Ia: chex.Array | None = None
  Iu: chex.Array | None = None
  Ip: float | None = None
  rBt: float | None = None
  iqQ: chex.Array | None = None
  jtorQ: chex.Array | None = None
  q95: chex.Array | None = None
  lp: chex.Array | None = None
  rbary: chex.Array | None = None
  IpVQ: chex.Array | None = None
  FtPVQ: chex.Array | None = None
  rhotornorm: chex.Array | None = None
  rgeom: chex.Array | None = None
  zgeom: chex.Array | None = None
  aminor: chex.Array | None = None
  epsilon: chex.Array | None = None
  kappa: chex.Array | None = None
  delta: chex.Array | None = None
  deltal: chex.Array | None = None
  deltau: chex.Array | None = None
  rrmax: chex.Array | None = None
  zrmax: chex.Array | None = None
  rrmin: chex.Array | None = None
  zrmin: chex.Array | None = None
  rzmax: chex.Array | None = None
  zzmax: chex.Array | None = None
  rzmin: chex.Array | None = None
  zzmin: chex.Array | None = None
  Q0Q: chex.Array | None = None
  Q1Q: chex.Array | None = None
  Q2Q: chex.Array | None = None
  Q3Q: chex.Array | None = None
  Q4Q: chex.Array | None = None
  ItQ: chex.Array | None = None
  LpQ: chex.Array | None = None
  rbQ: chex.Array | None = None
  Q5Q: chex.Array | None = None
  VQ: chex.Array | None = None
  AQ: chex.Array | None = None
  SlQ: chex.Array | None = None
  aq: chex.Array | None = None
  rq: chex.Array | None = None
  zq: chex.Array | None = None
  aW: chex.Array | None = None
  FW: chex.Array | None = None
  raqmin: chex.Array | None = None
  qmin: chex.Array | None = None
  raQ: chex.Array | None = None
  raR: chex.Array | None = None
  rS: chex.Array | None = None
  zS: chex.Array | None = None
  Ip: float | None = None
  Wk: float | None = None
  Wp: float | None = None
  Vp: float | None = None
  Ft: float | None = None
  Ft0: float | None = None
  Wt: float | None = None
  Wt0: float | None = None
  WN: float | None = None
  IpD: chex.Array | None = None
  WkD: chex.Array | None = None
  WpD: chex.Array | None = None
  VpD: float | None = None
  liD: float | None = None
  FtD: float | None = None
  FtPQ: float | None = None
  Ft0D: float | None = None
  zY: float | None = None
  bt: float | None = None
  VpQ: chex.Array | None = None
  li: float | None = None
  bp: float | None = None
  FR: float | None = None
  PpQg: chex.Array | None = None  # 2D
  zIpD: float | None = None
  WtD: float | None = None
  IpQ: chex.Array | None = None
  rYD: float | None = None
  WND: float | None = None
  mu: float | None = None
  rIpD: float | None = None
  zYD: float | None = None
  btD: float | None = None
  bpD: float | None = None
  Wt0D: float | None = None
  zIp: float | None = None
  rY: float | None = None
  qA: float | None = None
  rIp: float | None = None
  PQ: chex.Array | None = None
  bpli2: float | None = None
  Iv: chex.Array | None = None
  Is: chex.Array | None = None
  Bm: chex.Array | None = None
  Ff: chex.Array | None = None
  Fn: chex.Array | None = None
  Brn: chex.Array | None = None
  Bzn: chex.Array | None = None
  Brrn: chex.Array | None = None
  Brzn: chex.Array | None = None
  Bzrn: chex.Array | None = None
  Bzzn: chex.Array | None = None
  Bzx: chex.Array | None = None
  Btx: chex.Array | None = None
  Brx: chex.Array | None = None
  Brz: chex.Array | None = None
  Bzz: chex.Array | None = None
  F0x: chex.Array | None = None
  Fz: chex.Array | None = None
  F0z: chex.Array | None = None
  Br0x: chex.Array | None = None
  Bz0x: chex.Array | None = None
  Br0z: chex.Array | None = None
  Bz0z: chex.Array | None = None
  Va: chex.Array | None = None
  resy: float | None = None
  resC: float | None = None
  rese: float | None = None
  resFx: float | None = None
  IniD: float | None = None
  Ini: chex.Array | None = None
  Rp: chex.Array | None = None
  Lp: chex.Array | None = None
  res: float | None = None
  mkryl: int | None = None
  nfeval: int | None = None
  niter: int | None = None
  isconverged: bool | None = None
  Parel: chex.Array | None = None
  Iarel: chex.Array | None = None
  Tarel: chex.Array | None = None
  Um: chex.Array | None = None
  Uf: chex.Array | None = None
  Un: chex.Array | None = None


@jax.tree_util.register_dataclass
@dataclass
class StateData:
  """Equivalent to State struct in MEQ."""
  LYt: OutputData | None = None
  it: int | None = None
  dt: float | None = None
  Prec: chex.Array | None = None
  xnl: chex.Array | None = None
  xnldot: chex.Array | None = None
  Tstate: chex.Array | None = None
  PSstate: chex.Array | None = None
  dstate: chex.Array | None = None
  nnoc: int | None = None


@jax.tree_util.register_dataclass
@dataclass
class ConcData:
  """An interpretation of the Constraint structs in MEQ."""

  fun_name: str = field(metadata=dict(static=True))
  domain_id: int = field(metadata=dict(static=True))
  lx_name: str | None = field(metadata=dict(static=True), default=None)
  index: int | None = field(metadata=dict(static=True), default=None)

# pylint: enable=invalid name
