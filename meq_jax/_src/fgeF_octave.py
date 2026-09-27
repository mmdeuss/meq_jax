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

"""Builds fgeF's arguments from an Octave MEQ session.

Converting MEQ's L and LX structs to the MEQ-JAX dataclasses is long but
mechanical. It used to live inline in fgeF_test.py, so anything else calling
fgeF against an Octave equilibrium had to copy it. Moved here unchanged.

The Octave session must already define L, LX, x0, u0, aux and opts, as MEQ's
own callers do:

    [L,LX] = fgs('ana',shot,0,'nr',nr,'nz',nz,'icsint',false,'ilim',1);
    x0 = L.LX2x(LX);
    u0 = [LX.Ia(:);LX.Iu(:);LX2Co(L,LX)];
    aux = cell(1,6); aux{6} = true;
    opts = optsF('dojacx',true);
"""

import dataclasses
from typing import Any

from absl import logging

import jax.numpy as jnp
from meq_jax._src import fgeF
from meq_jax._src import types
import numpy as np

# pylint: disable=invalid-name

# Octave setup producing an L/LX pair to call fgeF on. Shared by the tests
# and the benchmark so both measure the same thing. Format with shot, icsint,
# algosNL and grid; grid is '' or ",'nr',<nr>,'nz',<nz>".

# isEvolutive is always false
FBT_SETUP = """
usecs = {icsint};
if usecs
  PPcs = {{'icsint',true ,'ilim',3,'tolcs',1e-12}};
else
  PPcs = {{'icsint',false,'ilim',1}};
end
% Generate baseline eq
[Lfbt,~,LYfbt] = fbt('ana',{shot},[],PPcs{{:}}{grid});
algosNL_ = '{algosNL}';

PP = {{
  'algoNL',algosNL_,...
  'debug',1,...
  'selu','e',...
  'cde','OhmTor_0D',...  % We could try OhmTor_0D
  'nu',30,...
  'wreg',2e-6,...
  'ipm',true,...
  'idml',true,...
  'stabz',false,...
  'iqR', linspace(0,1,20),...
  'raS', linspace(0,1,20),...
}};

L = fbt('ana',{shot},[],PP{{:}},PPcs{{:}}{grid});
LX = meqxconvert(Lfbt,LYfbt,L,true); % Generate minimal LX
LX.ag = LYfbt.ag;
LX.IpD = LX.Ip/L.nD;
x0 = L.LX2x(LYfbt);
u0 = [LX.Ia(:);LX.Iu(:);LX2Co(L,LYfbt)];
aux = cell(1,6);
aux{{6}} = true; % Do not compute approximate jacobian

% Turn on
opts = optsF('dojacx',true,'dojacu',true,'dojacxdot',true,'dopost',true);
"""


# isEvolutive is always true
FGE_SETUP = """
usecs = {icsint};
if usecs
  PPcs = {{'icsint',true ,'ilim',3,'tolcs',1e-12}};
else
  PPcs = {{'icsint',false,'ilim',1}};
end
% Generate baseline eq
algosNL_ = '{algosNL}';

PP = {{
  'algoNL',algosNL_,...
  'debug',1,...
  'selu','e',...
  'nu',30,...
  'wreg',2e-6,...
  'ipm',true,...
  'idml',true,...
  'stabz',false,...
  'iqR', linspace(0,1,20),...
  'raS', linspace(0,1,20),...
}};

[L,LX] = fge('ana',{shot},0,PP{{:}},PPcs{{:}});

x0 = L.LX2x(LX);

u0 = [LX.Ia(:);LX.Iu(:);LX2Co(L,LX)];

aux = cell(1,6);
aux{{6}} = true; % Do not compute approximate jacobian

% Turn on
opts = optsF('dojacx',true,'dojacu',true,'dojacxdot',true,'dopost',true);
"""


@dataclasses.dataclass(frozen=True)
class FgeFInputs:
  """The arguments fgeF takes, converted from an Octave session."""

  x0: jnp.ndarray
  u0: jnp.ndarray
  L: types.StaticData
  LX: types.InputData
  opts: types.Opts
  aux: list[Any]
  agconc: list[types.ConcData]
  cdeconc: list[types.ConcData]


def fgeF_inputs_from_octave(
    octave, shot: int, test_type: str = 'fbt'
) -> FgeFInputs:
  """Converts the L/LX structs in an Octave session into fgeF's arguments.

  Args:
    octave: An oct2py instance with L, LX, x0, u0, aux and opts defined.
    shot: The 'ana' shot the session used. Selects the doublet basis-function
      branch (shots 82 and 84), and is recorded on the converted structs.
    test_type: 'fbt' or 'fge'. The two fill different index and coupling
      fields on L, so the conversion differs.

  Returns:
    The converted arguments, with LX already updated by co2lx.

  Raises:
    ValueError: If test_type is not 'fbt' or 'fge'.
  """
  if test_type not in ('fbt', 'fge'):
    raise ValueError(f"test_type must be 'fbt' or 'fge', got {test_type!r}")
  def _process_int(var):
    return int(octave.eval(var).item())

  def _process_float(var):
    return float(octave.eval(var).item())

  def _process_bool(var):
    return bool(octave.eval(var).item())

  def _process_str(var):
    return str(octave.eval(var))

  def _process_1D(var):
    attr = octave.eval(var, nout=1).squeeze().T
    return jnp.atleast_1d(attr)

  def _process_2D(var):
    attr = octave.eval(var, nout=1).squeeze().T
    if attr.ndim == 1:
      return jnp.array(attr[None])
    else:
      return jnp.array(attr)

  ilackner = _process_int('L.P.ilackner;')
  if ilackner not in (1, 2):
    raise ValueError(f'ilackner={ilackner} is invalid')

  def _process_list(var, tuple_element=False):
    llist = octave.eval(var, nout=1)

    res = []
    for l in llist:
      val = l[0].squeeze().astype(int)
      if val.shape == ():
        el = val.item()
        if tuple_element:
          el = (el,)
        res.append(el)
      else:
        res.append(tuple(val))
    return tuple(res)

  # Bfct data
  if shot == 88:  # doublet_with_mantle
    ngi_list = _process_list('L.bfp.PbfgenD.ngi;')
    bfp_list = _process_list('L.bfp.PbfgenD.bfp;')
    iDbf_list = _process_list('L.bfp.PbfgenD.iDbf;', tuple_element=True)

    ng = _process_int('L.bfp.PbfgenD.ng')
    nbf = _process_int('L.bfp.PbfgenD.nbf')
    nD = _process_int('L.bfp.PbfgenD.nD;')

    # Index set
    igm = _process_1D('L.bfp.igm;').astype(int)
    fPg3 = _process_1D('L.bfp.fPg;')
    fPg3 = fPg3[igm - 1]
    fTg3 = _process_1D('L.bfp.fTg;')
    fTg3 = fTg3[igm - 1]

    Bfp = types.BfpData(
        ngi_list=ngi_list,
        bfp_list=bfp_list,
        iDbf_list=iDbf_list,
        ng=ng,
        nbf=nbf,
        nD=nD,
        igm=igm,
        fPg3=fPg3,
        fTg3=fTg3,
        shot=shot,
    )

  elif shot in [82, 84]:  # doublet
    ngi_list = _process_list('L.bfp.ngi;')
    bfp_list = _process_list('L.bfp.bfp;')
    # Must be a list of lists
    iDbf_list = _process_list('L.bfp.iDbf;', tuple_element=True)

    ng = _process_int('L.bfp.ng')
    nbf = _process_int('L.bfp.nbf')
    nD = _process_int('L.bfp.nD;')

    Bfp = types.BfpData(
        ngi_list=ngi_list,
        bfp_list=bfp_list,
        iDbf_list=iDbf_list,
        ng=ng,
        nbf=nbf,
        nD=nD,
        shot=shot,
    )

  else:
    bfp = _process_1D('L.bfp;')
    Bfp = types.BfpData(nP=int(bfp[0]), nT=int(bfp[1]))
  logging.debug('Created BfpData')

  P = types.ParameterData(
      ### Arguments for meqpost.py
      gsxe=_process_int('L.P.gsxe;'),
      ilackner=ilackner,
      iterq=_process_int('L.P.iterq;'),
      tolq=_process_float('L.P.tolq;'),
      icsint=_process_bool('L.P.icsint;'),
      nFW=_process_int('L.P.nFW;'),
      naR=_process_int('L.P.naR;'),
      r0=_process_float('L.P.r0;'),
      raS=None
      if len(octave.eval('L.P.raS;')) == 0
      else _process_1D('L.P.raS'),
      iqR=None
      if len(octave.eval('L.P.iqR;')) == 0
      else _process_1D('L.P.iqR;'),
      ifield=_process_bool('L.P.ifield;'),
      ivacuum=_process_bool('L.P.ivacuum;'),
      izgrid=_process_bool('L.P.izgrid;'),
      izxoverlap=_process_bool('L.P.izxoverlap;'),
      ### Arguments for meqpdom.py
      ilim=_process_int('L.P.ilim;'),
      dasm=_process_int('L.P.dasm;'),
      itercs=_process_int('L.P.itercs;'),
      xdoma=_process_int('L.P.xdoma;'),
      ihole=_process_bool('L.P.ihole;'),
      tolcs=_process_float('L.P.tolcs;'),
      isaddl=_process_bool('L.P.isaddl;'),
      ### Arguments for meqagconfun
      b0=_process_float('L.P.b0;'),
      ### New arguments
      algoNL=_process_str('L.P.algoNL;'),
      Ipmin=_process_float('L.P.Ipmin;'),
      cde=_process_str('L.P.cde;')
      if len(octave.eval('L.P.cde;')) > 0
      else None,
  )
  logging.debug('Created ParameterData')

  G = types.GeometryData(
      ### Arguments for meqpost.py
      rW=_process_1D('L.G.rW;'),
      zW=_process_1D('L.G.zW;'),
      nW=_process_int('L.G.nW;'),
      aW=_process_1D('L.G.aW;'),
      rx=_process_1D('L.G.rx;'),
      zx=_process_1D('L.G.zx;'),
      Bma=_process_2D('L.G.Bma;'),
      Bmx=_process_2D('L.G.Bmx;'),
      Bmu=_process_2D('L.G.Bmu;')
      if octave.eval('L.G.Bmu;').shape[1] > 0
      else None,
      Mfa=_process_2D('L.G.Mfa;'),
      Mfx=_process_2D('L.G.Mfx;'),
      Mfu=_process_2D('L.G.Mfu;')
      if octave.eval('L.G.Mfu;').shape[1] > 0
      else None,
      Tvu=_process_2D('L.G.Tvu;')
      if octave.eval('L.G.Tvu;').shape[1] > 0
      else None,
      Tius=_process_1D('L.G.Tius;')
      if len(octave.eval('L.G.Tius;')) > 0
      else None,
      ### Arguments for flcs.py
      rl=_process_1D('L.G.rl;'),
      zl=_process_1D('L.G.zl;'),
      ### New arguments
      na=_process_int('L.G.na;'),
  )
  logging.debug('Created GeometryData')

  Ind = types.IndexData(
      ixGS=_process_1D('L.ind.ixGS;').astype(int),
      ixg=_process_1D('L.ind.ixg;').astype(int),
      ixa=_process_1D('L.ind.ixa;').astype(int),
      ixu=_process_1D('L.ind.ixu;').astype(int),
      irGS=_process_1D('L.ind.irGS;').astype(int),
      irC=_process_1D('L.ind.irC;').astype(int),
      irp=_process_1D('L.ind.irp;').astype(int),
      ira=_process_1D('L.ind.ira;').astype(int),
      iru=_process_1D('L.ind.iru;').astype(int),
      iua=_process_1D('L.ind.iua;').astype(int),
      iuC=_process_1D('L.ind.iuC;').astype(int),
  )
  if test_type == 'fbt':
    Ind = dataclasses.replace(
        Ind,
        iuu=_process_1D('L.ind.iuu;').astype(int),
    )
  if test_type == 'fge':
    Ind = dataclasses.replace(
        Ind,
        iuni=_process_1D('L.ind.iuni;').astype(int),
        iurp=_process_1D('L.ind.iurp;').astype(int),
    )
  logging.debug('Created IndexData')

  L = types.StaticData(
      ### Arguments for meqpost.py
      P=P,
      G=G,
      bfp=Bfp,
      doq=_process_float('L.doq;'),
      nD=_process_int('L.nD;'),
      nQ=_process_int('L.nQ;'),
      nR=_process_int('L.nR;'),
      nS=_process_int('L.nS;'),
      noq=_process_int('L.noq;'),
      npq=_process_int('L.npq;'),
      crq=_process_1D('L.crq;'),
      czq=_process_1D('L.czq;'),
      crW=_process_1D('L.crW;'),
      czW=_process_1D('L.czW;'),
      cWx=_process_2D('L.cWx;'),  # 2D
      pq=_process_1D('L.pq;'),
      fq=_process_1D('L.fq;'),
      pinit=_process_float('L.pinit;'),
      c95=_process_1D('L.c95;'),
      i95=_process_1D('L.i95;').astype(int),
      kxW=_process_1D('L.kxW;'),
      nW=_process_int('L.nW;'),
      raN=_process_float('L.raN;'),
      idrx=_process_float('L.idrx;'),
      idzx=_process_float('L.idzx;'),
      drx=_process_float('L.drx;'),
      liurtemu=_process_bool('L.liurtemu;'),
      shot=int(shot),  # static argument
      Mbe=_process_2D('L.Mbe;'),
      Tbc=_process_2D('L.Tbc;'),
      Tye=_process_2D('full(L.Tye);'),  # undo sparse matrix
      ne=_process_int('L.ne;'),
      nx=_process_int('L.nx;'),
      ny=_process_int('L.ny;'),
      rx=_process_1D('L.rx;'),
      nrx=_process_int('L.nrx;'),
      nzx=_process_int('L.nzx;'),
      nry=_process_int('L.nry;'),
      nzy=_process_int('L.nzy;'),
      Mxe=_process_2D('L.Mxe;'),
      Mxy=_process_1D('L.Mxy;'),
      Mby=_process_1D('L.Mby;'),
      cx=_process_1D('L.cx;'),
      cq=_process_2D('L.cq;'),
      cr=_process_2D('L.cr;'),
      cs=_process_2D('L.cs;'),
      ci=_process_float('L.ci;'),
      co=_process_float('L.co;'),
      fPg=_process_1D('L.fPg;'),
      fTg=_process_1D('L.fTg;'),
      TDg=_process_2D('L.TDg'),
      idsx=_process_float('L.idsx'),
      dsx=_process_float('L.dsx'),
      rry=_process_2D('L.rry;'),
      pQ=_process_1D('L.pQ;'),
      dzx=_process_float('L.dzx;'),
      zzy=_process_1D('L.zzy;'),
      smalldia=_process_bool('L.smalldia;'),
      lxy=_process_2D('L.lxy;').astype(bool),
      nn=_process_bool('L.nn;'),
      bfct=octave.eval('func2str(L.bfct);'),
      ### Arguments for meqpdom
      FN=_process_float('L.FN;'),
      dimw=_process_int('L.dimw;'),
      Oasx=_process_2D('L.Oasx;'),
      clx=_process_2D('L.clx;'),
      Oly=_process_2D('L.Oly;'),
      kxl=_process_1D('L.kxl;').astype(int),
      ### Arguments for bfct
      ry=_process_1D('L.ry;'),
      iry=_process_1D('L.iry;'),
      ### Arguments for meqagcon
      Ip0=_process_float('L.Ip0;'),
      ### New arguments
      ind=Ind,
      isEvolutive=_process_bool('L.isEvolutive;'),
      nC=_process_int('L.nC;'),
      np=_process_int('L.np;'),
      nN=_process_int('L.nN;'),
      ng=_process_int('L.ng;'),
      Jx=_process_2D('L.Jx;'),
      Ju=_process_2D('L.Ju;'),
      Jxdot=_process_2D('L.Jxdot;')
      if len(octave.eval('L.Jxdot;')) > 0
      else None,
      resscal=_process_1D('L.resscal;'),
      xscal=_process_1D('L.xscal;'),
      dlst=_process_2D('full(L.dlst);'),  # undo sparse matrix
      rhsf=_process_1D('L.rhsf;'),
      code=_process_str('L.code;'),
      Txy=_process_2D('full(L.Txy);'),
  )
  L = dataclasses.replace(L, Txy_sum=jnp.sum(L.Txy, axis=0).astype(bool))
  logging.debug('Created StaticData')

  assert L.P is not None
  if L.P.icsint:
    L = dataclasses.replace(
        L,
        taur=_process_1D('L.taur;'),
        tauz=_process_1D('L.tauz;'),
        taul=_process_1D('L.taul;'),
        Mr=_process_2D('L.Mr;'),  # 2D
        Mz=_process_2D('L.Mz;'),  # 2D
        nrx=_process_int('L.nrx;'),
        nzx=_process_int('L.nzx;'),
        ### Arguments for flcs
        Ml=_process_2D('L.Ml;'),
    )

  if test_type == 'fge':
    L = dataclasses.replace(
        L,
        Mee=_process_2D('L.Mee;'),
        Mey=_process_2D('L.Mey;'),
        Re=_process_1D('L.Re;'),
        nrD=_process_int('L.nrD;'),
    )

  LX = types.InputData(
      ag=_process_1D('LX.ag;'),
      Ip=_process_1D('LX.Ip;').item(),
      qA=_process_1D('LX.qA;'),
      Iy=_process_2D('LX.Iy;'),
      Fx=_process_2D('LX.Fx;'),
      Ia=_process_1D('LX.Ia;'),
      Iu=_process_1D('LX.Iu;'),
      IpD=_process_1D('LX.IpD;'),
      t=_process_float('LX.t;'),
      rBt=_process_float('LX.rBt;'),
      shot=_process_int('LX.shot;'),
  )
  if shot in [82, 84, 88]:
    LX = dataclasses.replace(
        LX, WkD=_process_1D('LX.WkD;'), bpD=_process_1D('LX.bpD;')
    )
  else:
    LX = dataclasses.replace(
        LX, Wk=_process_1D('LX.Wk;'), bp=_process_1D('LX.bp;')
    )
  logging.debug('Created InputData')

  # ConcData. Copied from meqagcon_test.py
  fun_names = octave.eval('L.agconc(:,3);').squeeze().tolist()
  iDs = octave.eval('L.agconc(:,2);').squeeze().tolist()
  iis = (octave.eval('L.agconc(:,4);').squeeze()).tolist()
  agconc = []
  for fn_name, iD, ii in zip(fun_names, iDs, iis):
    agdata = types.ConcData(
        fun_name=fn_name,
        domain_id=int(np.asarray(iD).squeeze().item()),
        lx_name=fn_name,
        index=int(np.asarray(ii).squeeze().item()),
    )
    agconc.append(agdata)

  # Opts
  opts_oct = octave.eval('opts;', nout=1)
  opts = {}
  for k, v in opts_oct.items():
    opts[k] = bool(v)
  opts = types.Opts(**opts)

  # cdeconc
  cdeconc = None
  # TODO(adedieu): test with cdeconc non null
  try:
    assert L.P is not None
    if L.P.cde is not None:
      # Copied from meqcde_test
      fun_names = [L.P.cde]
      iDs = (
          (octave.eval('L.cdec(:,2);') - 1).astype(int).squeeze().tolist()
      )
      iis = (
          octave.eval('L.cdec(:,4);').astype(int).squeeze() - 1
      ).tolist()
      if isinstance(iDs, int):
        iDs = [iDs]
        iis = [iis]
      cdeconc = [
          types.ConcData(fun_name=fn, domain_id=iD, lx_name='', index=ii)
          for fn, iD, ii in zip(fun_names, iDs, iis)
      ]
    else:
      cdeconc = None
  except:
    logging.debug('L.P.cde is not None but L.cdec is undefined')

  aux = octave.eval('aux;', nout=1)
  aux_list = []
  for this_aux in aux[0]:
    if len(this_aux) == 0:
      aux_list.append(None)
    else:
      aux_list.append(bool(this_aux))
  logging.debug('Created aux, opts, cdeconc, agconc')

  # Process inputs
  x0 = _process_1D('x0;')
  u0 = _process_1D('u0;')

  # One time update of LX based on static u0
  LX = fgeF.co2lx(L=L, LX=LX, agconc=agconc, u=u0)
  return FgeFInputs(
      x0=x0,
      u0=u0,
      L=L,
      LX=LX,
      opts=opts,
      aux=aux_list,
      agconc=agconc,
      cdeconc=cdeconc,
  )


# pylint: enable=invalid-name
