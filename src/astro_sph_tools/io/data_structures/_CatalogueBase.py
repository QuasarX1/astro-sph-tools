# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from abc import abstractmethod
from collections.abc import Awaitable
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from unyt import unyt_array

from ._SnapshotBase import SnapshotBase
from ...data_structures._ParticleType import ParticleType
from ...data_structures._SimulationData import SimulationDataBase, T_ISimulation

class CatalogueBase(SimulationDataBase[T_ISimulation]):
    """
    Base class type for catalogue data reader types.
    """

    def __init__(
                    self,
                    membership_filepath: str,
                    properties_filepath: str,
                    snapshot: SnapshotBase,
                ) -> None:
        self.__membership_filepath: str = membership_filepath
        self.__properties_filepath: str = properties_filepath
        self.__snapshot: SnapshotBase = snapshot

        self.__n_haloes = self.get_number_of_haloes()
#        self.__halo_ids = self.get_halo_IDs()
#        self.__parent_ids = self.get_halo_parent_IDs()

        self.__snapshot_to_halo_particle_sorting_indexes: dict[ParticleType, np.ndarray] = {}
        self.__halo_to_snapshot_particle_sorting_indexes: dict[ParticleType, np.ndarray] = {}
        self.__snapshot_avalible_particle_filter: dict[ParticleType, np.ndarray] = {} # array of booleans of snapshot particle type length

#        self.__direct_children, self.__total_decendants = CatalogueBase._calculate_n_children(self.__halo_ids, self.__parent_ids)
        self.__direct_children, self.__total_decendants = CatalogueBase._calculate_n_children(*self._get_highrarchy_IDs())

    @property
    def membership_filepath(self) -> str:
        return self.__membership_filepath

    @property
    def properties_filepath(self) -> str:
        return self.__properties_filepath

    @property
    def snapshot(self) -> SnapshotBase:
        return self.__snapshot
    
    @property
    def redshift(self) -> float:
        return self.__snapshot.redshift
    @property
    def z(self) -> float:
        return self.redshift

    @property
    def hubble_param(self) -> float:
        return self.__snapshot.hubble_param
    @property
    def h(self) -> float:
        return self.hubble_param

    @property
    def expansion_factor(self) -> float:
        return self.__snapshot.expansion_factor
    @property
    def a(self) -> float:
        return self.expansion_factor
    
    @property
    def number_of_children(self) -> np.ndarray:
        return self.__direct_children

    @property
    def number_of_decendants(self) -> np.ndarray:
        return self.__total_decendants

    @property
    def number_of_haloes(self) -> int:
        return self.__n_haloes
    def __len__(self) -> int:
        return self.number_of_haloes
    @abstractmethod
    def get_number_of_haloes(self, particle_type: ParticleType|None = None) -> int:
        """
        Return the number of haloes in the catalogue.
        Optionally, specify a particle type to get the number of haloes containing those particles.
        """
        raise NotImplementedError("Attempted to call an abstract method.")
    
    @abstractmethod
    def get_halo_IDs(self, particle_type: ParticleType|None = None) -> np.ndarray:
        """
        Get a unique list of halo IDs.
        Optionally, specify a particle type to get only haloes containing those particles.
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_halo_parent_IDs(self, particle_type: ParticleType|None = None) -> np.ndarray:
        """
        Get a list halo IDs that indicate the parent of a particular halo.
        Length is identical to that returned by get_halo_IDs for the same arguments.
        Optionally, specify a particle type to get only haloes containing those particles.
        """
        raise NotImplementedError("Attempted to call an abstract method.")
    
    @abstractmethod
    def get_halo_top_level_parent_IDs(self, particle_type: ParticleType|None = None) -> np.ndarray:
        """
        Get a list halo IDs that indicate the top-most parent of a particular halo's higherarchy.
        Length is identical to that returned by get_halo_IDs for the same arguments.
        Optionally, specify a particle type to get only haloes containing those particles.
        """
        raise NotImplementedError("Attempted to call an abstract method.")
    
    @abstractmethod
    def get_halo_IDs_by_snapshot_particle(self, particle_type: ParticleType, include_unbound: bool = True, snapshot_particle_ids: np.ndarray|None = None) -> np.ndarray:
        """
        Get list of halo IDs - one for each particle in the snapshot.
        Particles with no associated halo recive an ID of -1.
        Optionally, specify a particle type to get only haloes containing those particles.
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_particle_IDs(self, particle_type: ParticleType, include_unbound: bool = True) -> np.ndarray:
        """
        Get a list of particle IDs that are included in the catalogue.
        Set 'include_unbound' to False to retrive only bound particles.
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_halo_centres(self, particle_type: ParticleType|None = None) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_halo_masses(self, particle_type: ParticleType|None = None) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

#    def __create_order_conversion(self, particle_type: ParticleType) -> None:
#        # Get the particle IDs for both the snapshot and catalogue
#        # By definition, the snapshot ID set must be at most idenstical to but more likley a subset of the snapshot ID set
#        halo_particle_ids = self.get_IDs(particle_type)
#        snapshot_particle_ids = self.__snapshot.get_IDs(particle_type)
#
#        # Get the indexes for each array in the order that sorts them
#        halo_sorted_indexes = halo_particle_ids.argsort()
#        snapshot_sorted_indexes = snapshot_particle_ids.argsort()
#
#        # Reverse the sorting opperation to get the indexes that will undo a sorted array back to the original order
#        halo_undo_sorted_indexes = halo_sorted_indexes.argsort()
#        snapshot_undo_sorted_indexes = snapshot_sorted_indexes.argsort()
#
#        # Sort both lists of IDs to make membership check easier, then undo the sort on the final boolean array
#        self.__snapshot_avalible_particle_filter[particle_type] = np.isin(snapshot_particle_ids[snapshot_sorted_indexes], halo_particle_ids[halo_sorted_indexes])[snapshot_undo_sorted_indexes]
#
#        # Sorting operations will only be done on matching particles, so re-compute the sort and unsort arrays for the snapshot IDs for only matched IDs
#        reduced_snapshot_sorted_indexes = snapshot_particle_ids[self.__snapshot_avalible_particle_filter[particle_type]].argsort()
#        reduced_snapshot_undo_sorted_indexes = reduced_snapshot_sorted_indexes.argsort()
#
#        # Define the correct translation ordering for each direction (for only matching IDs)
#        self.__snapshot_to_halo_particle_sorting_indexes[particle_type] = snapshot_particle_ids[self.__snapshot_avalible_particle_filter[particle_type]].argsort()[halo_undo_sorted_indexes]
#        self.__halo_to_snapshot_particle_sorting_indexes[particle_type] = halo_sorted_indexes[reduced_snapshot_undo_sorted_indexes]
#    
#    def halo_orderby_snapshot(self, particle_type: ParticleType, data: Union[np.ndarray, unyt_array], default_value: float = np.nan) -> Union[np.ndarray, unyt_array]:
#        if particle_type not in self.__halo_to_snapshot_particle_sorting_indexes:
#            self.__create_order_conversion(particle_type)
#        result = np.empty()
#        result[self.__snapshot_avalible_particle_filter] = data[self.__halo_to_snapshot_particle_sorting_indexes[particle_type]]
#        result[~self.__snapshot_avalible_particle_filter] = default_value
#        return result
#    
#    def snapshot_orderby_halo(self, particle_type: ParticleType, data: Union[np.ndarray, unyt_array]) -> Union[np.ndarray, unyt_array]:
#        if particle_type not in self.__snapshot_to_halo_particle_sorting_indexes:
#            self.__create_order_conversion(particle_type)
#        return data[self.__snapshot_to_halo_particle_sorting_indexes[particle_type]]

    @abstractmethod
    def _get_highrarchy_IDs(self) -> tuple[np.ndarray, np.ndarray]:#TODO: is this what is so slow?????
        """
        Return two signed integer arrays: ID of each halo and parent ID of each halo.
        A halo with no parent has a parent ID of -1.
        These IDs need only be self-consistent and may differ between implementations!
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @staticmethod
    def _calculate_n_children(halo_ids: np.ndarray, parent_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_direct_children = np.zeros_like(parent_ids, dtype = int)
        n_total_children = np.zeros_like(parent_ids, dtype = int)

        if (parent_ids != -1).sum() > 0 and (halo_ids != parent_ids).sum() > 0:

            null_index = -len(halo_ids)
            parent_indexes = np.empty_like(parent_ids, dtype = int)
            parent_indexes[parent_ids == -1] = null_index
            for index, id in enumerate(halo_ids):#TODO: looping over haloes in this way too slow?
                parent_indexes[parent_ids == id] = index

            for i in range(len(parent_indexes)):
                if parent_indexes[i] == null_index:
                    continue
                parent_index = parent_indexes[i]
                n_direct_children[parent_index] += 1 # Only increment direct children once for a given halo - each halo can be a direct child of only one halo
                while True:
                    n_total_children[parent_index] += 1
                    parent_index = parent_indexes[parent_index]
                    if parent_index == null_index:
                        break

        return n_direct_children, n_total_children

    # async versions

    async def get_number_of_haloes_async(self, particle_type: ParticleType|None = None) -> Awaitable[int]:
        """
        Return the number of haloes in the catalogue.
        Optionally, specify a particle type to get the number of haloes containing those particles.
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_number_of_haloes, particle_type)
    
    async def get_halo_IDs_async(self, particle_type: ParticleType|None = None) -> Awaitable[np.ndarray]:
        """
        Get a unique list of halo IDs.
        Optionally, specify a particle type to get only haloes containing those particles.
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_IDs, particle_type)

    async def get_halo_parent_IDs_async(self, particle_type: ParticleType|None = None) -> Awaitable[np.ndarray]:
        """
        Get a list halo IDs that indicate the parent of a particular halo.
        Length is identical to that returned by get_halo_IDs for the same arguments.
        Optionally, specify a particle type to get only haloes containing those particles.
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_parent_IDs, particle_type)
    
    async def get_halo_top_level_parent_IDs_async(self, particle_type: ParticleType|None = None) -> Awaitable[np.ndarray]:
        """
        Get a list halo IDs that indicate the top-most parent of a particular halo's higherarchy.
        Length is identical to that returned by get_halo_IDs for the same arguments.
        Optionally, specify a particle type to get only haloes containing those particles.
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_top_level_parent_IDs, particle_type)
    
    async def get_halo_IDs_by_snapshot_particle_async(self, particle_type: ParticleType, include_unbound: bool = True) -> Awaitable[np.ndarray]:
        """
        Get list of halo IDs - one for each particle in the snapshot.
        Particles with no associated halo recive an ID of -1.
        Optionally, specify a particle type to get only haloes containing those particles.
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_IDs_by_snapshot_particle, particle_type, include_unbound)

    async def get_particle_IDs_async(self, particle_type: ParticleType, include_unbound: bool = True) -> Awaitable[np.ndarray]:
        """
        Get a list of particle IDs that are included in the catalogue.
        Optionally, specify a particle type to get only those particles.
        Set 'include_unbound' to False to retrive only bound particles.
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_particle_IDs, particle_type, include_unbound)

    async def get_halo_centres_async(self, particle_type: ParticleType|None = None) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_centres, particle_type)

    async def get_halo_masses_async(self, particle_type: ParticleType|None = None) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_masses, particle_type)
    
    @staticmethod
    @abstractmethod
    def generate_filepaths(
       *snapshot_number_strings: str,
        directory: str,
        membership_basename: str,
        properties_basename: str,
        file_extension: str = "hdf5",
        parallel_ranks: list[int]|None = None
    ) -> dict[
            str,
            tuple[
                str|tuple[str, ...],
                str|tuple[str, ...]
            ]
         ]:
        raise NotImplementedError("Attempted to call an abstract method.")

    @staticmethod
    @abstractmethod
    def scrape_filepaths(
        directory: str,
        ignore_basenames: list[str]|None = None
    ) -> tuple[
            tuple[str, ...],
            str,
            str,
            str,
            tuple[int, ...]|None
         ]:
        raise NotImplementedError("Attempted to call an abstract method.")
    
    @staticmethod
    @abstractmethod
    def generate_filepaths_from_partial_info(
        directory: str,
        membership_basename: str|None = None,
        properties_basename: str|None = None,
        snapshot_number_strings: list[str]|None = None,
        file_extension: str|None = None,
        parallel_ranks: list[int]|None = None
    ) -> dict[
            str,
            tuple[
                str|tuple[str, ...],
                str|tuple[str, ...]
            ]
         ]:
        raise NotImplementedError("Attempted to call an abstract method.")
