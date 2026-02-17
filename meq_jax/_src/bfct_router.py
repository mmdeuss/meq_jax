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

"""Router for bfct functions."""

from meq_jax._src import bfct
from meq_jax._src import bfctnD
from meq_jax._src import bfctnD_doublet_mantle
from meq_jax._src import types


# pylint: disable=invalid-name


def bfct1(Bfp: types.BfpData, **kwargs):
  """Router for bfct1 functions."""
  if Bfp.shot == 88:
    return bfctnD_doublet_mantle.bfct1_doublet_mantle(Bfp=Bfp, **kwargs)
  elif Bfp.shot in [82, 84]:
    return bfctnD.bfct1_nD(Bfp=Bfp, **kwargs)
  else:
    return bfct.bfct1(Bfp=Bfp, **kwargs)


def bfct2(Bfp: types.BfpData, **kwargs):
  """Router for bfct2 functions."""
  if Bfp.shot == 88:
    return bfctnD_doublet_mantle.bfct2_doublet_mantle(Bfp=Bfp, **kwargs)
  elif Bfp.shot in [82, 84]:
    return bfctnD.bfct2_nD(Bfp=Bfp, **kwargs)
  else:
    return bfct.bfct2(Bfp=Bfp, **kwargs)


def bfct3(Bfp: types.BfpData, **kwargs):
  """Router for bfct3 functions."""
  if Bfp.shot == 88:
    return bfctnD_doublet_mantle.bfct3_doublet_mantle(Bfp=Bfp, **kwargs)
  elif Bfp.shot in [82, 84]:
    return bfctnD.bfct3_nD(Bfp=Bfp, **kwargs)
  else:
    return bfct.bfct3(Bfp=Bfp, **kwargs)


def bfct5(Bfp: types.BfpData, **kwargs):
  """Router for bfct5 functions."""
  if Bfp.shot == 88:
    return bfctnD_doublet_mantle.bfct5_doublet_mantle(Bfp=Bfp, **kwargs)
  elif Bfp.shot in [82, 84]:
    return bfctnD.bfct5_nD(Bfp=Bfp, **kwargs)
  else:
    return bfct.bfct5(Bfp=Bfp, **kwargs)


def bfct11(Bfp: types.BfpData, **kwargs):
  """Router for bfct11 functions."""
  if Bfp.shot == 88:
    return bfctnD_doublet_mantle.bfct11_doublet_mantle(Bfp=Bfp, **kwargs)
  elif Bfp.shot in [82, 84]:
    return bfctnD.bfct11_nD(Bfp=Bfp, **kwargs)
  else:
    return bfct.bfct11(Bfp=Bfp, **kwargs)


def bfct91(Bfp: types.BfpData, **kwargs):
  """Router for bfct91 functions."""
  if Bfp.shot == 88:
    return bfctnD_doublet_mantle.bfct91_doublet_mantle(Bfp=Bfp, **kwargs)
  elif Bfp.shot in [82, 84]:
    return bfctnD.bfct91_nD(Bfp=Bfp, **kwargs)
  else:
    return bfct.bfct91(Bfp=Bfp, **kwargs)
