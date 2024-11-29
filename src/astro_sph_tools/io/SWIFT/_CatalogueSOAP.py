# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed
from .._ParticleType import ParticleType
from ._CatalogueBase import CatalogueBase
from ._SnapshotSWIFT import SnapshotSWIFT

import os
from typing import Union, List, Tuple, Dict
import re

import numpy as np
from unyt import unyt_array
import h5py as h5
from swiftsimio.objects import cosmo_array
from QuasarCode import Settings, Console

class CatalogueSOAP(CatalogueBase):
    """
    SOAP catalogue data (SWIFT).

    Currently, Only VR SOAP catalogues are supported
    """

    def __init__(
                    self,
                    membership_filepath: str,
                    properties_filepath: str,
                    snapshot: SnapshotSWIFT,
                ) -> None:

        # File handles
        self.__membership_file = h5.File(membership_filepath, "r")
        self.__halo_data_file = h5.File(properties_filepath, "r")

        # Check for VR support
        if "VR" not in self.__halo_data_file:#TODO: support other SOAP catalogue types!
            self.__membership_file.close()
            self.__halo_data_file.close()
            raise NotImplementedError("SOAP catalogue not based on VELOCIraptor. Only VR catalogues are presently supported.")

        # Get halo ID information
        self.__halo_ids = self.__halo_data_file["VR/ID"][:]
        self.__halo_parent_ids = self.__halo_data_file["VR/ParentHaloID"][:]
        self.__halo_top_most_parent_ids = self.__halo_data_file["VR/HostHaloID"][:]
        self.__halo_top_most_parent_ids[self.__halo_top_most_parent_ids == -1] = self.__halo_ids[self.__halo_top_most_parent_ids == -1]

        assert (~(np.isin(self.__halo_parent_ids[self.__halo_parent_ids >= 0], self.__halo_ids))).sum() == 0
        assert (~(np.isin(self.__halo_top_most_parent_ids[self.__halo_top_most_parent_ids >= 0], self.__halo_ids))).sum() == 0

        # Convert halo IDs to indexes (SOAP membership info is stored as indexes)
        self.__halo_indexes = self.__halo_ids - 1 # np.arange(len(self.__halo_ids), dtype = int)
        self.__halo_parent_indexes = np.where(self.__halo_parent_ids != -1, self.__halo_parent_ids - 1, -1)
        self.__halo_top_most_parent_indexes = self.__halo_top_most_parent_ids - 1

#        self.__halo_parent_indexes = np.empty_like(self.__halo_indexes, dtype = int)
#        self.__halo_top_most_parent_indexes = np.empty_like(self.__halo_indexes, dtype = int)
#        sorted_halo_ids_order = np.argsort(self.__halo_ids)
#        self.__halo_parent_indexes = sorted_halo_ids_order[np.searchsorted(self.__halo_ids, self.__halo_parent_ids, sorter = sorted_halo_ids_order)]
#        print((self.__halo_parent_indexes != self.__halo_parent_ids).sum())
#        print(self.__halo_parent_indexes.shape, self.__halo_parent_ids.shape)
#        print((self.__halo_parent_indexes != self.__halo_parent_ids - 1).sum())
#        exit()
#        self.__halo_top_most_parent_indexes = sorted_halo_ids_order[np.searchsorted(self.__halo_ids, self.__halo_top_most_parent_ids, sorter = sorted_halo_ids_order)]

#        assert (self.__halo_parent_indexes < 0).sum() == 0
        assert (self.__halo_top_most_parent_indexes < 0).sum() == 0

        assert (~np.isin(self.__halo_parent_indexes[self.__halo_parent_indexes != -1], self.__halo_indexes)).sum() == 0
        assert (~np.isin(self.__halo_top_most_parent_indexes, self.__halo_indexes)).sum() == 0

        # Grab both index fields at the same time (reduces redundant IO)
        all_indexes = { part_type : self.__membership_file[part_type.common_hdf5_name]["GroupNr_all"][:] for part_type in ParticleType.get_all() }
        bound_indexes = { part_type : self.__membership_file[part_type.common_hdf5_name]["GroupNr_bound"][:] for part_type in ParticleType.get_all() }

        self.__halos_containing_parttypes = {}
        for part_type in ParticleType.get_all():
            indexes_by_particle = np.unique(all_indexes[part_type])
            self.__halos_containing_parttypes[part_type] = np.isin(self.__halo_indexes, indexes_by_particle[indexes_by_particle != -1])
        # Include a 'filter' for no specific type
        self.__halos_containing_parttypes[None] = np.full_like(self.__halo_ids, True, dtype = bool)

        # Pre-calculate the number of haloes for each option
        self.__n_haloes = { key : self.__halos_containing_parttypes[key].sum() for key in self.__halos_containing_parttypes }

        # Concatinate part type info and generate filters
        n_parts = { part_type: snapshot.number_of_particles(part_type) for part_type in ParticleType.get_all() }
        self.__halo_indexes_by_particle = np.empty(shape = sum(n_parts.values()), dtype = int)
        self.__particle_type_filters = { particle_type: np.full_like(self.__halo_indexes_by_particle, False, dtype = bool) for particle_type in ParticleType.get_all() }
        self.__member_filter = np.empty_like(self.__halo_indexes_by_particle, dtype = bool)
        self.__bound_filter = np.empty_like(self.__halo_indexes_by_particle, dtype = bool)
        offset = 0
        for particle_type in ParticleType.get_all():
            if Settings.debug:
                assert all_indexes[particle_type].shape[0] == n_parts[particle_type], f"Membership information contained a different number of {part_type.name} particles."
            endpoint = offset + n_parts[particle_type]
            self.__halo_indexes_by_particle[offset : endpoint] = all_indexes[particle_type]
            self.__particle_type_filters[particle_type][offset : endpoint] = True
            self.__member_filter[offset : endpoint] = all_indexes[particle_type] > -1
            self.__bound_filter[offset : endpoint] = bound_indexes[particle_type] > -1
            offset = endpoint

        super().__init__(
            membership_filepath = membership_filepath,
            properties_filepath = properties_filepath,
            snapshot = snapshot
        )

    @property
    def snapshot(self) -> SnapshotSWIFT:
        return super().snapshot

    def get_number_of_haloes(self, particle_type: Union[ParticleType, None] = None) -> int:
        """
        Return the number of haloes in the catalogue.
        Optionally, specify a particle type to get the number of haloes containing those particles.
        """
        return self.__n_haloes[particle_type]

    def get_halo_IDs(self, particle_type: Union[ParticleType, None] = None) -> np.ndarray:
        """
        Get a unique list of halo IDs.
        Optionally, specify a particle type to get only haloes containing those particles.
        """
        return self.__halo_indexes[self.__halos_containing_parttypes[particle_type]]

    def get_halo_parent_IDs(self, particle_type: Union[ParticleType, None] = None) -> np.ndarray:
        """
        Get a list halo IDs that indicate the parent of a particular halo.
        Length is identical to that returned by get_halo_IDs for the same arguments.
        Optionally, specify a particle type to get only haloes containing those particles.
        """
        return self.__halo_parent_indexes[self.__halos_containing_parttypes[particle_type]]
    
    def get_halo_top_level_parent_IDs(self, particle_type: Union[ParticleType, None] = None) -> np.ndarray:
        """
        Get a list halo IDs that indicate the top-most parent of a particular halo's higherarchy.
        Length is identical to that returned by get_halo_IDs for the same arguments.
        Optionally, specify a particle type to get only haloes containing those particles.
        """
        return self.__halo_top_most_parent_indexes[self.__halos_containing_parttypes[particle_type]]

    def get_halo_IDs_by_snapshot_particle(self, particle_type: ParticleType, include_unbound: bool = True) -> np.ndarray:
        """
        Get list of halo IDs - one for each particle in the snapshot.
        Particles with no associated halo recive an ID of -1.
        Optionally, specify a particle type to get only haloes containing those particles.
        """
        # SOAP membership data is already in catalogue order
        if include_unbound:
            return self.__halo_indexes_by_particle[self.__particle_type_filters[particle_type]]
        else:
            return np.where(self.__bound_filter, self.__halo_indexes_by_particle, -1)[self.__particle_type_filters[particle_type]]

    def get_particle_IDs(self, particle_type: ParticleType, include_unbound: bool = True) -> np.ndarray:
        """
        Get a list of particle IDs that are included in the catalogue.
        Optionally, specify a particle type to get only those particles.
        Set 'include_unbound' to False to retrive only bound particles.
        """
        return self.snapshot.get_IDs(particle_type)[(self.__member_filter if include_unbound else self.__bound_filter)[self.__particle_type_filters[particle_type]]]

    def _get_halo_property_propper_cgs_conversion(self, property_path: str) -> float:
        return self.__halo_data_file[property_path].attrs["Conversion factor to CGS (including cosmological corrections)"]
    def _get_halo_property_comoving_cgs_conversion(self, property_path: str) -> float:
        return self.__halo_data_file[property_path].attrs["Conversion factor to CGS (not including cosmological corrections)"]

    def get_halo_property(self, property_path: str, particle_type: Union[ParticleType, None] = None, convert_to_propper_units = False) -> np.ndarray:
        """
        Returns a field value converted to CGS units (in the requested length scheme).
        """
        return self.__halo_data_file[property_path][:][self.__halos_containing_parttypes[particle_type]] * (self._get_halo_property_propper_cgs_conversion if convert_to_propper_units else self._get_halo_property_comoving_cgs_conversion)(property_path)
    # TODO: figure out how to implement this as a cosmo_array object
#        return cosmo_array(
#            self.__halo_data_file[property_path][:][self.__halos_containing_parttypes[particle_type]],
#            unit,
#            cosmo_factor=cosmo_factor,
#            name=description,
#            compression=compression
#        )

    def get_halo_centres(self, particle_type: Union[ParticleType, None] = None, comoving = True) -> unyt_array:
        return unyt_array(self.get_halo_property("VR/CentreOfPotential", particle_type, convert_to_propper_units = not comoving), units = "cm").to("Mpc")

    def get_halo_masses(self, particle_type: Union[ParticleType, None] = None) -> unyt_array:
        return unyt_array(self.get_halo_property("SO/200_crit/TotalMass", particle_type), units = "g").to("Msun")
    
    def _get_highrarchy_IDs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return two signed integer arrays: ID of each halo and parent ID of each halo.
        A halo with no parent has a parent ID of -1.
        These IDs need only be self-consistent and may differ between implementations!
        """
        return (self.__halo_ids, self.__halo_parent_ids)
    
    @staticmethod
    def generate_filepaths(
       *snapshot_number_strings: str,
        directory: str,
        membership_basename: str,
        properties_basename: str,
        file_extension: str = "hdf5",
        parallel_ranks: Union[List[int], None] = None
    ) -> Dict[
            str,
            Tuple[
                Union[str, Tuple[str, ...]],
                Union[str, Tuple[str, ...]]
            ]
         ]:
#Union[Tuple[Tuple[Tuple[str, ...], ...], Tuple[Tuple[str, ...], ...]], Tuple[Tuple[str, ...], Tuple[str, ...]]]:

        file_extension = file_extension.strip(".")
        parallel_insert = "" if parallel_ranks is None else ".{{}}"
        template = os.path.join(directory, f"{{}}{{{{}}}}{parallel_insert}.{file_extension}")
        membership_template = template.format(membership_basename)
        properties_template = template.format(properties_basename)

        return {
            snapnum_string : (
                (
                    os.path.abspath(membership_template.format(snapnum_string)),
                    os.path.abspath(properties_template.format(snapnum_string))
                )
                if parallel_ranks is None else
                (
                    tuple([os.path.abspath(membership_template.format(snapnum_string, p)) for p in parallel_ranks]),
                    tuple([os.path.abspath(properties_template.format(snapnum_string, p)) for p in parallel_ranks])
                )
            )
            for i, snapnum_string
            in enumerate(snapshot_number_strings)
        }

#        result = [[None] * len(snapshot_number_strings),
#                  [None] * len(snapshot_number_strings)]
#        for i, snapnum_string in enumerate(snapshot_number_strings):
#            result[0][i] = membership_template.format(snapnum_string)
#            result[1][i] = properties_template.format(snapnum_string)
#            if parallel_ranks is not None:
#                result[0][i] = (result[0][i].format(p) for p in parallel_ranks)
#                result[1][i] = (result[1][i].format(p) for p in parallel_ranks)
#        result = (tuple(result[0]), tuple(result[1]))
#        return result

    @staticmethod
    def scrape_filepaths(
        directory: str,
        ignore_basenames: Union[list[str], None] = None
    ) -> Tuple[
            Tuple[str, ...],
            str,
            str,
            str,
            Union[Tuple[int, ...], None]
         ]:
        """
        Given the directory containing a SOAP catalogue, identify the file name information for the catalogue.

        Set ignore_basenames to a list of basenames in the same folder that should be ignored.
        """

        if ignore_basenames is None:
            ignore_basenames = []

        snap_number_components = set()
        parallel_components = set()
        basename_prefixes = set()
        file_extensions = set()
        pattern = re.compile(r"^(\D+)(\d+)(?:\.(\d+))?(\.\w+)$")
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                if match.group(1) not in ignore_basenames:
                    basename_prefixes.add(match.group(1))
                    snap_number_components.add(match.group(2))
                    if match.group(3):
                        parallel_components.add(int(match.group(3)))
                    file_extensions.add(match.group(4))

        # Check that the results are valid
        assert len(snap_number_components) > 0, f"No catalogue files found."
        assert len(file_extensions) == 1, f"Inconsistent file extensions in catalogue directory."
        assert len(basename_prefixes) == 2, f"Unexpected number of file formats for a SOAP catalogue. Expected 2 but found {len(basename_prefixes)}."

        # Format results
        snap_number_components = tuple(sorted(list(snap_number_components), key = lambda v: int(v)))
        parallel_components = tuple(sorted(list(parallel_components)))
        basename_prefixes = tuple(basename_prefixes)
        file_extension = list(file_extensions)[0]

        # Determine which basename is which
        test_file = CatalogueSOAP.generate_filepaths(
            snap_number_components[-1], # Use the final snap in case the first is empty due to having no haloes
            directory = directory,
            membership_basename = basename_prefixes[0],
            properties_basename = basename_prefixes[1],
            file_extension = file_extension,
            parallel_ranks = None if len(parallel_components) == 0 else parallel_components[0]
        )[snap_number_components[-1]][0]
        if len(parallel_components) > 0:
            test_file = test_file[0]
        with h5.File(test_file) as f:
            first_is_membership = ParticleType.gas.common_hdf5_name in f

        return (
            snap_number_components,
            os.path.abspath(directory),
            basename_prefixes[0 if first_is_membership else 1],
            basename_prefixes[1 if first_is_membership else 0],
            file_extension,
            None if len(parallel_components) == 0 else parallel_components
        )
    
    @staticmethod
    def generate_filepaths_from_partial_info(
        directory: str,
        membership_basename: Union[str, None] = None,
        properties_basename: Union[str, None] = None,
        snapshot_number_strings: Union[List[str], None] = None,
        file_extension: Union[str, None] = None,
        parallel_ranks: Union[List[int], None] = None
    ) -> Dict[
            str,
            Tuple[
                Union[str, Tuple[str, ...]],
                Union[str, Tuple[str, ...]]
            ]
         ]:

        scraped_info = CatalogueSOAP.scrape_filepaths(directory)

        # Generate filepaths
        return CatalogueSOAP.generate_filepaths(
           *snapshot_number_strings if snapshot_number_strings is not None else scraped_info[0],
            directory = directory,
            membership_basename = membership_basename if membership_basename is not None else scraped_info[2],
            properties_basename = properties_basename if properties_basename is not None else scraped_info[3],
            file_extension = file_extension if file_extension is not None else scraped_info[4],
            parallel_ranks = parallel_ranks if parallel_ranks is not None else scraped_info[5]
        )
