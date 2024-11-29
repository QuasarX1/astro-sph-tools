# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from functools import singledispatch

import numpy as np
from unyt import unyt_array, unyt_quantity

def calculate_wrapped_displacement(from_positions: np.ndarray, to_positions: np.ndarray, box_width: float) -> np.ndarray:
    """
    Calculates the true displacement vector between two points in a periodic box.
    This assumes that the true displacement is always the shortest distance between two points.
    If the two points are part of a timeseries dataset, the value of dt must be sufficiently small such that the particles cannot travel more than half the length of the box.
    """
    position_deltas = to_positions - from_positions
    deltas_needing_wrapping = np.abs(position_deltas) > box_width / 2
    position_deltas[deltas_needing_wrapping] = position_deltas[deltas_needing_wrapping] - (np.sign(position_deltas[deltas_needing_wrapping]) * box_width)
    del deltas_needing_wrapping
    return position_deltas

def calculate_wrapped_distance(from_position: np.ndarray, to_positions: np.ndarray, box_width: float, do_squared_distance = False) -> np.ndarray:
    """
    Calculates the length of the true displacement vector between two points in a periodic box.
    This assumes that the true displacement is always the shortest distance between two points.
    If the two points are part of a timeseries dataset, the value of dt must be sufficiently small such that the particles cannot travel more than half the length of the box.
    """
    displacement = calculate_wrapped_displacement(from_position, to_positions, box_width)
    squared_distance = (displacement**2).sum(axis = 1 if ((len(from_position.shape) > 1) or (len(to_positions.shape) > 1)) else 0)
    del displacement
    if do_squared_distance:
        return squared_distance
    else:
        return np.sqrt(squared_distance)

def make_periodic(positions: np.ndarray, box_width: float, origin_is_centre: bool = False):
    if origin_is_centre:
        half_box_width = box_width / 2
        wrap = (positions < -half_box_width) | (positions >= half_box_width)
        positions[wrap] = -np.sign(positions[wrap] + half_box_width) * box_width + positions[wrap]
    else:
        wrap = (positions < 0.0) | (positions >= box_width)
        positions[wrap] = -np.sign(positions[wrap]) * box_width + positions[wrap]
@singledispatch
def calculate_periodic(start_positions: np.ndarray, box_width: float, origin_is_centre: bool = False) -> np.ndarray:
    final_positions = np.copy(start_positions)
    make_periodic(final_positions, box_width, origin_is_centre)
    return final_positions
@calculate_periodic.register(unyt_array)
def _(start_positions: unyt_array, box_width: unyt_quantity, origin_is_centre: bool = False) -> unyt_array:
    return unyt_array(calculate_periodic(start_positions.value, box_width.to(start_positions.units).value, origin_is_centre), units = start_positions.units)

@singledispatch
def shift_origin(start_positions: np.ndarray, new_origin: np.ndarray, box_width: float, origin_is_centre: bool = False) -> np.ndarray:
    new_positions = start_positions - new_origin
    make_periodic(new_positions, box_width, origin_is_centre)
    return new_positions
@shift_origin.register(unyt_array)
def _(start_positions: unyt_array, new_origin: unyt_array, box_width: unyt_quantity, origin_is_centre: bool = False) -> unyt_array:
    return unyt_array(shift_origin(start_positions.to(new_origin.units).value, new_origin.value, box_width.to(new_origin.units).value, origin_is_centre), units = new_origin.units)

@singledispatch
def shift_centre(start_positions: np.ndarray, new_centre: np.ndarray, box_width: float, origin_is_centre: bool = False) -> np.ndarray:
    if origin_is_centre:
        return shift_origin(start_positions, new_centre, box_width, origin_is_centre) 
    else:
        new_positions = start_positions + ((box_width / 2) - new_centre)
        make_periodic(new_positions, box_width, origin_is_centre)
        return new_positions
@shift_centre.register(unyt_array)
def _(start_positions: unyt_array, new_centre: unyt_array, box_width: unyt_quantity, origin_is_centre: bool = False) -> unyt_array:
    return unyt_array(shift_centre(start_positions.to(new_centre.units).value, new_centre.value, box_width.to(new_centre.units).value, origin_is_centre), units = new_centre.units)
