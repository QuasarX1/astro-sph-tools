# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

import numpy as np
from typing import Callable
from QuasarCode import Console

from ..._CoordinateAxes import CoordinateAxes
from ._kernels import quartic_spline_kernel
from ._pixel_calculation import calculate_pixel_value

def process_chunk(
    xi_chunk: int,
    yi_chunk: int,
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    particle_properties: np.ndarray,
    image_size: tuple[int, int],
    chunk_size: int,
    projection_axis: CoordinateAxes,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    kernel_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> tuple[int, int, np.ndarray]:
    Console.print_debug(f"Processing chunk: xi_chunk={xi_chunk}, yi_chunk={yi_chunk}")
    img_chunk = np.zeros((chunk_size, chunk_size))
    
    xi_end = min(xi_chunk + chunk_size, image_size[0])
    yi_end = min(yi_chunk + chunk_size, image_size[1])
    
    pixel_size_x = (x_max - x_min) / image_size[0]
    pixel_size_y = (y_max - y_min) / image_size[1]

    # Calculate the mask for particles that overlap with the chunk
    if projection_axis == CoordinateAxes.X:
        x_mask = (positions[:, 1] >= x_min + xi_chunk * pixel_size_x - 2 * smoothing_lengths) & (positions[:, 1] < x_min + xi_end * pixel_size_x + 2 * smoothing_lengths)
        y_mask = (positions[:, 2] >= y_min + yi_chunk * pixel_size_y - 2 * smoothing_lengths) & (positions[:, 2] < y_min + yi_end * pixel_size_y + 2 * smoothing_lengths)
    elif projection_axis == CoordinateAxes.Y:
        x_mask = (positions[:, 0] >= x_min + xi_chunk * pixel_size_x - 2 * smoothing_lengths) & (positions[:, 0] < x_min + xi_end * pixel_size_x + 2 * smoothing_lengths)
        y_mask = (positions[:, 2] >= y_min + yi_chunk * pixel_size_y - 2 * smoothing_lengths) & (positions[:, 2] < y_min + yi_end * pixel_size_y + 2 * smoothing_lengths)
    else:  # projection_axis == CoordinateAxes.Z:
        x_mask = (positions[:, 0] >= x_min + xi_chunk * pixel_size_x - 2 * smoothing_lengths) & (positions[:, 0] < x_min + xi_end * pixel_size_x + 2 * smoothing_lengths)
        y_mask = (positions[:, 1] >= y_min + yi_chunk * pixel_size_y - 2 * smoothing_lengths) & (positions[:, 1] < y_min + yi_end * pixel_size_y + 2 * smoothing_lengths)
    
    mask = x_mask & y_mask
    masked_positions = positions[mask]
    masked_smoothing_lengths = smoothing_lengths[mask]
    masked_particle_properties = particle_properties[mask]

    for xi in range(xi_chunk, xi_end):
        for yi in range(yi_chunk, yi_end):
            if xi_chunk == 0 and yi_chunk == 0:
                Console.print_debug(f"Processing pixel: xi={xi}, yi={yi}")
            value = calculate_pixel_value(
                xi,
                yi,
                masked_positions,
                masked_smoothing_lengths,
                masked_particle_properties,
                str(projection_axis).encode(),
                x_min,
                x_max,
                y_min,
                y_max,
                image_size[0],
                kernel_func
            )
            img_chunk[xi - xi_chunk, yi - yi_chunk] = value

    return xi_chunk, yi_chunk, img_chunk

def create_image(
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    particle_properties: np.ndarray,
    image_size: tuple[int, int],
    chunk_size: int,
    projection_axis: CoordinateAxes,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    kernel_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = quartic_spline_kernel
) -> np.ndarray:
    img = np.zeros((image_size[0], image_size[1]))
    chunks = [
        (
            xi_chunk,
            yi_chunk,
            positions,
            smoothing_lengths,
            particle_properties,
            image_size,
            chunk_size,
            projection_axis,
            x_min,
            x_max,
            y_min,
            y_max,
            kernel_func
        )
        for xi_chunk in range(0, image_size[0], chunk_size)
        for yi_chunk in range(0, image_size[1], chunk_size)
    ]
    
    Console.print_debug("Starting processing")

    results = [process_chunk(*chunk) for chunk in chunks]

    for result in results:
        xi_chunk, yi_chunk, img_chunk = result
        xi_end = min(xi_chunk + chunk_size, image_size[0])
        yi_end = min(yi_chunk + chunk_size, image_size[1])
        img[xi_chunk:xi_end, yi_chunk:yi_end] = img_chunk[:(xi_end - xi_chunk), :(yi_end - yi_chunk)]
        Console.print_debug(f"Finished chunk: xi_chunk={xi_chunk}, yi_chunk={yi_chunk}")
    
    return img

'''
# Generate a large number of particles
n_particles = 1000000  # One million particles for a large dataset
positions = np.random.rand(n_particles, 3) * 100  # Random positions in a 100x100x100 cube
smoothing_lengths = np.random.rand(n_particles) * 10  # Random smoothing lengths between 0 and 10
particle_properties = np.random.rand(n_particles)  # Random particle properties

image_size = (200, 300)  # Separate sizes for x and y dimensions
chunk_size = 50
projection_axis = CoordinateAxes.Z  # Can be CoordinateAxes.X, CoordinateAxes.Y, or CoordinateAxes.Z
x_min, x_max = 0, 100  # Example physical space x range
y_min, y_max = 0, 100  # Example physical space y range

Console.print_debug("Creating image")
image = create_image(
    positions,
    smoothing_lengths,
    particle_properties,
    image_size,
    chunk_size,
    projection_axis,
    x_min,
    x_max,
    y_min,
    y_max
)
Console.print_debug("Image created")

# Plot the image using matplotlib
plt.imshow(np.log10(image + 1e-10), origin='lower', cmap='viridis')
plt.colorbar()
plt.title('SPH Smoothed Image')
plt.savefig('sph_smoothed_image.png')
'''
