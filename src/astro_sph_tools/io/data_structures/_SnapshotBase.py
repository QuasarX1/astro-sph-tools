# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from abc import abstractmethod
from collections.abc import Awaitable
import asyncio
from concurrent.futures import ThreadPoolExecutor
from scipy.constants import gravitational_constant
import os

import numpy as np
from unyt import unyt_array, unyt_quantity

from ...data_structures._ParticleType import ParticleType
from ...data_structures._SimulationData import SimulationDataBase, T_ISimulation

class SnapshotBase(SimulationDataBase[T_ISimulation]):
    """
    Base class type for snapshot data reader types.
    """

    def __init__(
                    self,
                    filepath: str,
                    number: str,
                    redshift: float,
                    hubble_param: float,
                    omega_baryon: float,
                    expansion_factor: float,
                    box_size: float,
                    snipshot: bool
                ) -> None:
        self.__filepath: str = filepath
        self.__file_name: str = os.path.split(self.__filepath)[1]
        self.__snap_num: str = number
        self.__redshift: float = redshift
        self.__hubble_param: float = hubble_param
        self.__omega_baryon: float = omega_baryon
        self.__expansion_factor: float = expansion_factor
        self.__box_size: unyt_array = box_size
        self.__is_snipshot: unyt_array = snipshot

        self.__n_parts: dict[ParticleType, int] = self._get_number_of_particles()
        self.__n_parts_this_rank: dict[ParticleType, int] = self._get_number_of_particles_this_rank()

    @property
    def filepath(self) -> str:
        return self.__filepath

    @property
    def file_name(self) -> str:
        return self.__file_name

    @property
    def number(self) -> str:
        return self.__snap_num

    @property
    def redshift(self) -> float:
        return self.__redshift
    @property
    def z(self) -> float:
        return self.redshift

    @property
    def hubble_param(self) -> float:
        return self.__hubble_param
    @property
    def h(self) -> float:
        return self.hubble_param

    @property
    def expansion_factor(self) -> float:
        return self.__expansion_factor
    @property
    def a(self) -> float:
        return self.expansion_factor

    @property
    def box_size(self) -> unyt_array:
        return self.__box_size

    @property
    def snipshot(self) -> bool:
        return self.__is_snipshot

    def remove_h_factor(self, data: np.ndarray) -> np.ndarray:
        return data / self.h

    def make_h_less(self, data: np.ndarray) -> np.ndarray:
        return data * self.h

    def to_physical(self, data: np.ndarray|unyt_array) -> np.ndarray|unyt_array:
        return data * self.a

    def to_comoving(self, data: np.ndarray|unyt_array) -> np.ndarray|unyt_array:
        return data / self.a
    
    def calculate_comoving_critical_density(self):
        return (self.h**2 * unyt_quantity(100.0, units = "km/s/Mpc")**2 * 3.0 / 8.0 / np.pi / unyt_quantity(gravitational_constant, units = "N*m**2/kg**2")).to("Msun/Mpc**3")

    def calculate_proper_critical_density(self, redshift: float):
        return (1.0 + redshift)**3 * self.calculate_comoving_critical_density()

    @property
    def proper_critical_density(self):
        return self.calculate_proper_critical_density(self.z)

    def calculate_comoving_critical_gas_density(self):
        return self.__omega_baryon * self.calculate_comoving_critical_density()
    
    def calculate_proper_critical_gas_density(self, redshift: float):
        return self.__omega_baryon * self.calculate_proper_critical_density(redshift)

    @property
    def proper_critical_gas_density(self):
        return self.calculate_proper_critical_gas_density(self.z)

    @abstractmethod
    def _get_number_of_particles(self) -> dict[ParticleType, int]:
        """
        Called by constructor.
        """
    @abstractmethod
    def _get_number_of_particles_this_rank(self) -> dict[ParticleType, int]:
        """
        Called by constructor.
        """
        raise NotImplementedError("Attempted to call an abstract method.")
    def number_of_particles(self, particle_type: ParticleType) -> int:
        return self.__n_parts[particle_type]
    def number_of_particles_this_rank(self, particle_type: ParticleType) -> int:
        return self.__n_parts_this_rank[particle_type]

    @abstractmethod
    def get_IDs(self, particle_type: ParticleType) -> np.ndarray:
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_smoothing_lengths(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_masses(self, particle_type: ParticleType) -> unyt_array:
        if particle_type == ParticleType.black_hole:
            raise ValueError("get_masses is not supported for black hole particle as they lack a simple mass field.")
        return self._get_masses(particle_type)
    @abstractmethod
    def _get_masses(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_black_hole_subgrid_masses(self) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_black_hole_dynamical_masses(self) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_positions(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_velocities(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_sfr(self, particle_type: ParticleType) -> unyt_array:
        if particle_type != ParticleType.gas:
            raise ValueError("get_sfr is not supported for particle type other than gas.")
        return self._get_sfr(particle_type)
    @abstractmethod
    def _get_sfr(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")
    
    def get_volumes(self, particle_type: ParticleType) -> unyt_array:
        return self.get_smoothing_lengths(particle_type)**3 * (np.pi * (4/3))

    def get_metalicities(self, particle_type: ParticleType) -> unyt_array:
        if particle_type != ParticleType.gas and particle_type != ParticleType.star:
            raise ValueError("get_metalicities is not supported for particle types other than gas and star.")
        return self._get_metalicities(particle_type)
    @abstractmethod
    def _get_metalicities(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_mean_enrichment_redshift(self, particle_type: ParticleType) -> unyt_array:
        if particle_type != ParticleType.gas and particle_type != ParticleType.star:
            raise ValueError("get_mean_enrichment_redshift is not supported for particle types other than gas and star.")
        return self._get_mean_enrichment_redshift(particle_type)
    @abstractmethod
    def _get_mean_enrichment_redshift(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_densities(self, particle_type: ParticleType) -> unyt_array:
        if particle_type != ParticleType.gas and particle_type != ParticleType.star:
            raise ValueError("get_densities is not supported for particle types other than gas and star.")
        return self._get_densities(particle_type)
    @abstractmethod
    def _get_densities(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_number_densities(self, particle_type: ParticleType, element: str) -> unyt_array:
        if particle_type != ParticleType.gas and particle_type != ParticleType.star:
            raise ValueError("get_number_densities is not supported for particle types other than gas and star.")
        return self._get_number_densities(particle_type, element)
    @abstractmethod
    def _get_number_densities(self, particle_type: ParticleType, element: str) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_temperatures(self, particle_type: ParticleType) -> unyt_array:
        if particle_type != ParticleType.gas and particle_type != ParticleType.star:
            raise ValueError("get_temperatures is not supported for particle types other than gas and star.")
        return self._get_temperatures(particle_type)
    @abstractmethod
    def _get_temperatures(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    # async versions

    async def get_IDs_async(self, particle_type: ParticleType) -> Awaitable[np.ndarray]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_IDs, particle_type)
#        return self.get_IDs(particle_type)

    async def get_smoothing_lengths_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_smoothing_lengths, particle_type)
#        return self.get_smoothing_lengths(particle_type)

    async def get_masses_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_masses, particle_type)
#        return self.get_masses(particle_type)

    async def get_black_hole_subgrid_masses_async(self) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_black_hole_subgrid_masses)
#        return self.get_black_hole_subgrid_masses()

    async def get_black_hole_dynamical_masses_async(self) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_black_hole_dynamical_masses)
#        return self.get_black_hole_dynamical_masses()

    async def get_positions_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_positions, particle_type)
#        return self.get_positions(particle_type)

    async def get_velocities_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_velocities, particle_type)
#        return self.get_velocities(particle_type)

    async def get_sfr_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_sfr, particle_type)
#        return self.get_sfr(particle_type)

    async def get_metalicities_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_metalicities, particle_type)
#        return self.get_metalicities(particle_type)

    @staticmethod
    @abstractmethod
    def generate_filepaths(
       *snapshot_number_strings: str,
        directory: str,
        basename: str,
        file_extension: str = "hdf5",
        parallel_ranks: list[int]|None = None
    ) -> dict[
            str,
            str|dict[int, str]
         ]:
        raise NotImplementedError("Attempted to call an abstract method.")

    @staticmethod
    @abstractmethod
    def scrape_filepaths(
        catalogue_directory: str
    ) -> tuple[
            tuple[
                str,
                tuple[str, ...],
                tuple[int, ...]|None,
                str
            ],
            ...
         ]:
        raise NotImplementedError("Attempted to call an abstract method.")

    @staticmethod
    @abstractmethod
    def generate_filepaths_from_partial_info(
        directory: str,
        basename: str|None = None,
        snapshot_number_strings: list[str]|None = None,
        file_extension: str|None = None,
        parallel_ranks: list[int]|None = None
    ) -> dict[
            str,
            str|dict[int, str]
         ]:
        raise NotImplementedError("Attempted to call an abstract method.")

    @staticmethod
    @abstractmethod
    def get_snapshot_order(snapshot_file_info: list[str], reverse = False) -> list[str]:
        raise NotImplementedError("Attempted to call an abstract method.")
