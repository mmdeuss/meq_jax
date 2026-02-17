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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import asxycs
from meqpy import octave_utils
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name


_TASKS = (
    'Vacuum',
    'Lim-X',
    'SND',
    'DND',
    'Double-Snowflake-Minus',
    'Double-Snowflake-Plus',
    'Doublet-div',
    'Doublet-div-nomantle',
    'Triplet-madness',
)
_SIPS = (-1, 1)


_FIELDS_CMD = """
L = fgs('ana',0,0,'cappav',2,'nc',16,'ac',0.6,...
        'debug',false,'nz',32,'nr',32,'infct',@qintmex,...
        'ilim',3,'icsint',1);

% Those are all the fields we need to pass to asxycs.
S = meq_test.generate_flux_map('{task}', {sIp});
Fx = S.Fxh(L.rrx,L.zzx);
% Add infinitesimal flux to break symmetry
eps = 1e-10;
Fx = Fx + eps*(1:size(Fx)(2)) + eps*(1:size(Fx)(1))';
[Mr,rk] = csdec(L.G.rx, L.G.rx, 'a');  % L.Mr, L.taur
[Mz,zk] = csdec(L.G.zx, L.G.zx, 'a');  % L.Mz, L.tauz

[itercs] = L.P.itercs;
[tolcs] = L.P.tolcs;
[idrx] = L.idrx;
[idzx] = L.idzx;

[zA,rA,~,~,~,~,ixA,zX,rX,~,~,~,~,ixX] = ...
    asxymex(Fx,L.G.zx,L.G.rx,L.P.dasm,L.dzx,L.drx,L.idzx,L.idrx,L.Oasx,L.dimw);
"""

_ASXYCS_CMD = """
[zA,rA,~,~,~,~,~,zX,rX,~,~,~,~,~,~] = ...
  asxymex(Fx,L.G.zx,L.G.rx,L.P.dasm,L.dzx,L.drx,L.idzx,L.idrx,L.Oasx,L.dimw);
[L.Mr,L.taur] = csdec(L.G.rx, L.G.rx, 'a');  % L.Mr, L.taur
[L.Mz,L.tauz] = csdec(L.G.zx, L.G.zx, 'a');  % L.Mz, L.tauz

[zA,rA,FA,dz2FA,dr2FA,drzFA,zX,rX,FX,dz2FX,dr2FX,drzFX] = asxycs(Fx,zA,rA,zX,rX,L);
"""

_ASXYCS_JAC_CMD = """
[dF{s},dz{s},dr{s},ddz2F{s},ddr2F{s},ddrzF{s}] = asxycsJac(Fx,z{s},r{s},dz2F{s},dr2F{s},drzF{s},L,2);
"""


_INPUT_FIELDS = (
    'Fx',
    'Mr',
    'rk',
    'Mz',
    'zk',
    'itercs',
    'tolcs',
    'zA',
    'rA',
    'zX',
    'rX',
    'ixA',
    'ixX',
    'idrx',
    'idzx',
)


class AsxycsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  @parameterized.parameters(*itertools.product(_TASKS, _SIPS))
  def test_asxycs(self, task, sIp):
    atol = 1e-10
    jac_atol = 1e-4
    self._oct.eval(_FIELDS_CMD.format(task=task, sIp=sIp))
    fields = dict(zip(_INPUT_FIELDS, self._oct.pull(_INPUT_FIELDS)))
    # The python implementation of asxycs.
    # Note: passing transposed as JAX expects row-major.

    # Create masked versions of the outputs for consistency with JAX version
    zA = np.ones_like(fields['Fx']).ravel() * -np.inf
    rA = zA.copy()
    zX = zA.copy()
    rX = zA.copy()

    ixA = fields['ixA'].astype(np.int32)
    ixX = fields['ixX'].astype(np.int32)
    zA[ixA-1] = fields['zA']
    rA[ixA-1] = fields['rA']
    zX[ixX-1] = fields['zX']
    rX[ixX-1] = fields['rX']

    zA = zA.reshape(fields['Fx'].shape[::-1])
    rA = rA.reshape(fields['Fx'].shape[::-1])
    zX = zX.reshape(fields['Fx'].shape[::-1])
    rX = rX.reshape(fields['Fx'].shape[::-1])

    zA, rA, FA, dz2FA, dr2FA, drzFA, zX, rX, FX, dz2FX, dr2FX, drzFX = (
        jax.jit(asxycs.asxycs, static_argnames=['itercs', 'tolcs'])(
            Fx=fields['Fx'].T,
            zA=zA,
            rA=rA,
            zX=zX,
            rX=rX,
            Mr=fields['Mr'].T,
            Mz=fields['Mz'].T,
            rk=jnp.squeeze(fields['rk']),
            zk=jnp.squeeze(fields['zk']),
            itercs=int(fields['itercs'].item()),
            tolcs=fields['tolcs'].item(),
        )
    )

    dFA, dzA, drA, ddz2FA, ddr2FA, ddrzFA = jax.jit(asxycs.jac)(
        Fx=fields['Fx'].T,
        zs=zA[~jnp.isinf(zA)],
        rs=rA[~jnp.isinf(rA)],
        dz2Fs=dz2FA[~jnp.isinf(dz2FA)],
        dr2Fs=dr2FA[~jnp.isinf(dr2FA)],
        drzFs=drzFA[~jnp.isinf(drzFA)],
        Mr=fields['Mr'].T,
        Mz=fields['Mz'].T,
        rk=jnp.squeeze(fields['rk']),
        zk=jnp.squeeze(fields['zk']),
        idrx=fields['idrx'].item(),
        idzx=fields['idzx'].item(),
        dmax=2,
    )

    dFX, dzX, drX, ddz2FX, ddr2FX, ddrzFX = jax.jit(asxycs.jac)(
        Fx=fields['Fx'].T,
        zs=zX[~jnp.isinf(zX)],
        rs=rX[~jnp.isinf(rX)],
        dz2Fs=dz2FX[~jnp.isinf(dz2FX)],
        dr2Fs=dr2FX[~jnp.isinf(dr2FX)],
        drzFs=drzFX[~jnp.isinf(drzFX)],
        Mr=fields['Mr'].T,
        Mz=fields['Mz'].T,
        rk=jnp.squeeze(fields['rk']),
        zk=jnp.squeeze(fields['zk']),
        idrx=fields['idrx'].item(),
        idzx=fields['idzx'].item(),
        dmax=2,
    )

    # The Matlab implementation of asxycs.
    (
        zA_,
        rA_,
        FA_,
        dz2FA_,
        dr2FA_,
        drzFA_,
        zX_,
        rX_,
        FX_,
        dz2FX_,
        dr2FX_,
        drzFX_,
    ) = self._oct.eval(_ASXYCS_CMD, verbose=False, nout=12)

    dFA_, dzA_, drA_, ddz2FA_, ddr2FA_, ddrzFA_ = self._oct.eval(
        _ASXYCS_JAC_CMD.format(s='A'), nout=6)

    dFX_, dzX_, drX_, ddz2FX_, ddr2FX_, ddrzFX_ = self._oct.eval(
        _ASXYCS_JAC_CMD.format(s='X'), nout=6)

    # Note: I remove the infinities when comparing with Octave's output.
    np.testing.assert_allclose(zA[~jnp.isinf(zA)], zA_.T[0], atol=atol)
    np.testing.assert_allclose(rA[~jnp.isinf(rA)], rA_.T[0], atol=atol)
    np.testing.assert_allclose(zX[~jnp.isinf(zX)], zX_.T[0], atol=atol)
    np.testing.assert_allclose(rX[~jnp.isinf(rX)], rX_.T[0], atol=atol)
    np.testing.assert_allclose(FA[~jnp.isinf(FA)], FA_.T[0], atol=atol)
    np.testing.assert_allclose(FX[~jnp.isinf(FX)], FX_.T[0], atol=atol)
    np.testing.assert_allclose(
        dz2FA[~jnp.isinf(dz2FA)], dz2FA_.T[0], atol=atol
    )
    np.testing.assert_allclose(
        dr2FA[~jnp.isinf(dr2FA)], dr2FA_.T[0], atol=atol
    )
    np.testing.assert_allclose(
        drzFA[~jnp.isinf(drzFA)], drzFA_.T[0], atol=atol
    )
    np.testing.assert_allclose(
        dz2FX[~jnp.isinf(dz2FX)], dz2FX_.T[0], atol=atol
    )
    np.testing.assert_allclose(
        dr2FX[~jnp.isinf(dr2FX)], dr2FX_.T[0], atol=atol
    )
    np.testing.assert_allclose(
        drzFX[~jnp.isinf(drzFX)], drzFX_.T[0], atol=atol
    )

    # Jacobian
    np.testing.assert_allclose(dFA, dFA_, atol=jac_atol)
    np.testing.assert_allclose(dzA, dzA_, atol=jac_atol)
    np.testing.assert_allclose(drA, drA_, atol=jac_atol)
    np.testing.assert_allclose(ddz2FA, ddz2FA_, atol=jac_atol)
    np.testing.assert_allclose(ddr2FA, ddr2FA_, atol=jac_atol)
    np.testing.assert_allclose(ddrzFA, ddrzFA_, atol=jac_atol)

    np.testing.assert_allclose(dFX, dFX_, atol=jac_atol)
    np.testing.assert_allclose(dzX, dzX_, atol=jac_atol)
    np.testing.assert_allclose(drX, drX_, atol=jac_atol)
    np.testing.assert_allclose(ddz2FX, ddz2FX_, atol=jac_atol)
    np.testing.assert_allclose(ddr2FX, ddr2FX_, atol=jac_atol)
    np.testing.assert_allclose(ddrzFX, ddrzFX_, atol=jac_atol)


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
