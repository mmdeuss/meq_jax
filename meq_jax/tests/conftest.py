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

"""Shared pytest configuration for meq_jax tests.

Some MEQ test helpers (notably ``meq_test.generate_flux_map``) live in a
MATLAB-only class (``meq/tests/meq_test.m`` inherits from
``matlab.unittest.TestCase``, which GNU Octave cannot load). The
``octave_compat`` directory next to this file provides Octave-compatible
shims. We wrap ``octave_utils.create_meq_oct2py_instance`` so every Octave
instance created by the tests has the shim directory at the front of its
path.
"""

import functools
import os

from meqpy import octave_utils

_OCTAVE_COMPAT_DIR = os.path.join(os.path.dirname(__file__), 'octave_compat')


def _wrap(create_fn):
  if getattr(create_fn, '_meq_jax_octave_compat', False):
    return create_fn

  @functools.wraps(create_fn)
  def wrapper(*args, **kwargs):
    octave = create_fn(*args, **kwargs)
    # addpath prepends, so the shim shadows the MATLAB-only class.
    octave.addpath(_OCTAVE_COMPAT_DIR)
    return octave

  wrapper._meq_jax_octave_compat = True
  return wrapper


octave_utils.create_meq_oct2py_instance = _wrap(
    octave_utils.create_meq_oct2py_instance
)
