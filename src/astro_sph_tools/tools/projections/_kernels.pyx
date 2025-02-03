# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

import numpy as np
cimport numpy as np
from libc.math cimport pow, pi

def quartic_spline_kernel(double[:] r, double[:] h):
    cdef int n = len(r)
    cdef double[:] result = np.zeros(n, dtype=np.float64)
    cdef double q, q3, h3
    cdef int i
    for i in range(n):
        q = r[i] / h[i]
        if q < 1.0:
            result[i] = (1 - 1.5 * pow(q, 2) + 0.75 * pow(q, 3)) / (pi * pow(h[i], 3))
        elif q < 2.0:
            result[i] = (0.25 * pow((2 - q), 3)) / (pi * pow(h[i], 3))
    return np.array(result)
