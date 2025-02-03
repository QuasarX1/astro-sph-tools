# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def calculate_pixel_value(double xi, double yi, double[:, :] positions, double[:] smoothing_lengths, double[:] particle_properties, 
                          char projection_axis, double x_min, double x_max, double y_min, double y_max, int image_size, object kernel_func):
    cdef double pixel_size_x = (x_max - x_min) / image_size
    cdef double pixel_size_y = (y_max - y_min) / image_size
    cdef double x = x_min + xi * pixel_size_x
    cdef double y = y_min + yi * pixel_size_y
    cdef double[:] dx, dy
    cdef double[:] r2, r, weights
    cdef double result = 0.0
    cdef int i
    
    if projection_axis == b'x':
        dx = positions[:, 1] - x
        dy = positions[:, 2] - y
    elif projection_axis == b'y':
        dx = positions[:, 0] - x
        dy = positions[:, 2] - y
    else:  # projection_axis == 'z'
        dx = positions[:, 0] - x
        dy = positions[:, 1] - y

    r2 = dx**2 + dy**2
    mask = r2 < (2.0 * smoothing_lengths)**2
    r = np.sqrt(r2[mask])
    weights = kernel_func(r, smoothing_lengths[mask])
    result = np.sum(particle_properties[mask] * weights)

    return result
