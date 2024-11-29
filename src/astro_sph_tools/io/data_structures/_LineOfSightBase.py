# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

import numpy as np
from unyt import unyt_array
from abc import ABC, abstractmethod
from typing import TypeVar, ParamSpec
from collections.abc import Callable
import h5py as h5
import os

T = TypeVar("T")
P = ParamSpec("P")

class LineOfSightFileBase(ABC):
    def __init__(self, filepath: str, number_of_sightlines: int, number_of_sightline_particles: np.ndarray, sightline_start_positions: unyt_array, sightline_direction_vectors: np.ndarray, redshift: float, expansion_factor: float, hubble_param: float):
        self.__filepath = filepath
        self.__file_name = os.path.split(self.__filepath)[1]
        self.__number_of_sightlines = number_of_sightlines
        self.__number_of_sightline_particles = number_of_sightline_particles
        self.__sightline_start_positions = sightline_start_positions
        self.__sightline_direction_vectors = sightline_direction_vectors
        self.__redshift = redshift
        self.__expansion_factor = expansion_factor
        self.__hubble_param = hubble_param

    @property
    def filepath(self) -> str:
        return self.__filepath

    @property
    def file_name(self) -> str:
        return self.__file_name
    
    def get_readonly_file_handle(self) -> h5.File:
        return h5.File(self.filepath, "r")

    @property
    def redshift(self) -> float:
        return self.__redshift
    @property
    def z(self) -> float:
        return self.__redshift

    @property
    def expansion_factor(self) -> float:
        return self.__expansion_factor
    @property
    def a(self) -> float:
        return self.__expansion_factor

    @property
    def hubble_param(self) -> float:
        return self.__hubble_param
    @property
    def h(self) -> float:
        return self.__hubble_param

    @property
    def number_of_sightlines(self) -> int:
        return self.__number_of_sightlines
    def __len__(self) -> int:
        return self.__number_of_sightlines
    
    def get_sightline_length(self, sightline_index: int) -> int:
        return self.__number_of_sightline_particles[sightline_index]
    
    def get_sightline_start_position(self, sightline_index: int) -> unyt_array:
        return self.__sightline_start_positions[sightline_index, :]
    
    def get_sightline_direction_vector(self, sightline_index: int) -> np.ndarray:
        return self.__sightline_direction_vectors[sightline_index, :]
    
    @abstractmethod
    def get_sightline(self, index: int, cache_data: bool = True) -> "LineOfSightBase":
        raise NotImplementedError("Attempted to call an abstract method.")

    @staticmethod
    @abstractmethod
    def get_files(directory: str, prefix: str = ...) -> tuple[str, ...]:
        raise NotImplementedError("Attempted to call an abstract method.")

class LineOfSightBase(ABC):
    def __init__(self, file_object: LineOfSightFileBase, number_of_particles: int, start_position: unyt_array, direction_vector: np.ndarray, cache_data: bool = True):
        self.__file = file_object
        self.__number_of_particles = number_of_particles
        self.__start_position = start_position
        self.__direction = direction_vector

        self.__caching_enabled: bool = cache_data
        self.__cached_data: dict[str, unyt_array] = {}

    @property
    def file(self) -> LineOfSightFileBase:
        return self.__file

    @property
    def number_of_particles(self) -> int:
        return self.__number_of_particles
    def __len__(self) -> int:
        return self.__number_of_particles
    
    @property
    def start_position(self) -> unyt_array:
        return self.__start_position
    
    @property
    def direction(self) -> np.ndarray:
        return self.__direction
    
    def delete_cache(self):
        self.__cached_data = {}

    @property
    def cache_data(self) -> bool:
        return self.__caching_enabled
    @cache_data.setter
    def _(self, value: bool):
        self.__caching_enabled = value

    def __check_cache_before_read(self, key: str, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        if self.__caching_enabled:
            if key not in self.__cached_data:
                self.__cached_data[key] = func(*args, **kwargs)
            return self.__cached_data[key]
        else:
            return func(*args, **kwargs)

    @property
    def positions_comoving(self) -> unyt_array:
        return self.__check_cache_before_read("positions_comoving", self._read_positions)
    @property
    def positions_proper(self) -> unyt_array:
        return self.__check_cache_before_read("positions_proper", self._read_positions, comoving = False)
    @abstractmethod
    def _read_positions(self, comoving = True) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @property
    def velocities_comoving(self) -> unyt_array:
        return self.__check_cache_before_read("velocities_comoving", self._read_velocities)
    @property
    def velocities_proper(self) -> unyt_array:
        return self.__check_cache_before_read("velocities_proper", self._read_velocities, comoving = False)
    @abstractmethod
    def _read_velocities(self, comoving = True) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @property
    def masses(self) -> unyt_array:
        return self.__check_cache_before_read("masses", self._read_masses)
    @abstractmethod
    def _read_masses(self) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @property
    def metallicities(self) -> unyt_array:
        return self.__check_cache_before_read("metallicities", self._read_metallicities)
    @abstractmethod
    def _read_metallicities(self) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @property
    def temperatures(self) -> unyt_array:
        return self.__check_cache_before_read("temperatures", self._read_temperatures)
    @abstractmethod
    def _read_temperatures(self) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @property
    def densities_comoving(self) -> unyt_array:
        return self.__check_cache_before_read("densities_comoving", self._read_densities)
    @property
    def densities_proper(self) -> unyt_array:
        return self.__check_cache_before_read("densities_proper", self._read_densities, comoving = False)
    @abstractmethod
    def _read_densities(self, comoving = True) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @property
    def smoothing_lengths_comoving(self) -> unyt_array:
        return self.__check_cache_before_read("smoothing_lengths_comoving", self._read_smoothing_lengths)
    @property
    def smoothing_lengths_proper(self) -> unyt_array:
        return self.__check_cache_before_read("smoothing_lengths_proper", self._read_smoothing_lengths, comoving = False)
    @abstractmethod
    def _read_smoothing_lengths(self, comoving = True) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")
