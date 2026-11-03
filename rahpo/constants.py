"""Physical constants of the solver.

The numerical kernels use these values. They are not free parameters of a run.
"""
from __future__ import annotations

import numpy as np

C0 = 299_792_458.0
MU0 = 4.0 * np.pi * 1e-7
EPS0 = 1.0 / (C0 * C0 * MU0)
FC = 100e6  # center frequency of the Ricker wavelet, Hz
