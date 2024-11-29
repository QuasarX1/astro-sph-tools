# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed
from .._ParticleType import ParticleType
from ._SnapshotBase import SnapshotBase

from typing import Union, List, Tuple, Dict
import os
import re

import numpy as np
from unyt import unyt_array, unyt_quantity
import swiftsimio as sw
from scipy.spatial import KDTree
from QuasarCode import Console

class SnapshotSWIFT(SnapshotBase):
    """
    SWIFT snapshot data.
    """

    @staticmethod
    def make_reader_object(filepath: str) -> object:#TODO: find type
        """
        Create an swiftsimio instance.
        """
        return sw.load(filepath)

    def __init__(self, filepath: str) -> None:
        Console.print_debug(f"Loading SWIFT snapshot from: {filepath}")

        pattern = re.compile(r'(?P<basename>[^/]+?)(?P<file_number>\d+)(?:\.(?P<parallel_id>(?:[1-9]\d*|0)))?\.(?P<extension>\w+)$')
        match = pattern.match(os.path.split(filepath)[1])
        if not match:
            raise ValueError(f"Snapshot file name \"{os.path.split(filepath)[1]}\" does not conform to the naming scheme of a SWIFT snapshot. SWIFT snapshot files must have a clear snapshot number component.")
        snap_num = match.group("file_number")

        self.__file_object = SnapshotSWIFT.make_reader_object(filepath)

        self.__cached_dm_smoothing_lengths: Union[unyt_array, None] = None

        super().__init__(
            filepath = filepath,
            number = snap_num,
            redshift = float(self.__file_object.metadata.header["Redshift"][0]),
            hubble_param = float(self.__file_object.metadata.cosmology["h"][0]),
            omega_baryon = float(self.__file_object.metadata.cosmology["Omega_b"][0]),
            expansion_factor = float(self.__file_object.metadata.header["Scale-factor"][0]),
            box_size = self.__file_object.metadata.boxsize
        )

    def _get_number_of_particles(self) -> Dict[ParticleType, int]:
        return { part_type : int(self.__file_object.metadata.header["NumPart_Total"][part_type.value]) for part_type in ParticleType.get_all() }

    def get_IDs(self, particle_type: ParticleType) -> np.ndarray:
        return particle_type.get_SWIFT_dataset(self.__file_object).particle_ids.value

    def get_smoothing_lengths(self, particle_type: ParticleType) -> unyt_array:
        """
        Note: dark matter particles have their smoothing lengths calculated using scipy KDTree
        """
        if particle_type == ParticleType.dark_matter:
            N_NABOURS = 32#TODO: move to settings and generalise with staticmethod on base class
            if self.__cached_dm_smoothing_lengths is None:
                n_part = self.number_of_particles(ParticleType.dark_matter)
                dm_part_positions = self.get_positions(ParticleType.dark_matter)
                unit = dm_part_positions.units
                dm_part_positions = dm_part_positions.value
                tree = KDTree(dm_part_positions)
                chunk_size = 10**4#TODO: move this to settings
                if n_part <= chunk_size:
                    # This causes a memory error
                    try:
                        self.__cached_dm_smoothing_lengths = unyt_array(tree.query(dm_part_positions, k = N_NABOURS)[0][:, N_NABOURS - 1], units = unit)
                    except MemoryError as e:
                        Console.print_warning("Not enough memory avalible to calculate particle smoothing lengths.\nDecrease the chunk size setting.")
                        raise e
                else:
                    self.__cached_dm_smoothing_lengths = unyt_array(np.zeros(n_part), units = unit)
                    for i in range(0, n_part, chunk_size):
                        selected_slice = slice(i, max(i + chunk_size, n_part))
                        self.__cached_dm_smoothing_lengths[selected_slice] = unyt_array(tree.query(dm_part_positions[selected_slice], k = N_NABOURS)[0][:, N_NABOURS - 1], units = unit).to("Mpc")
            return self.__cached_dm_smoothing_lengths
        else:
            return particle_type.get_SWIFT_dataset(self.__file_object).smoothing_lengths.to("Mpc")

    def _get_masses(self, particle_type: ParticleType) -> unyt_array:
        return particle_type.get_SWIFT_dataset(self.__file_object).masses.to("Msun")

    def get_black_hole_subgrid_masses(self) -> unyt_array:
        return self.__file_object.black_holes.subgrid_masses.to("Msun")

    def get_black_hole_dynamical_masses(self) -> unyt_array:
        return self.__file_object.black_holes.dynamical_masses.to("Msun")

    def get_positions(self, particle_type: ParticleType) -> unyt_array:
        return particle_type.get_SWIFT_dataset(self.__file_object).coordinates.to("Mpc")

    def get_velocities(self, particle_type: ParticleType) -> unyt_array:
        return particle_type.get_SWIFT_dataset(self.__file_object).velocities.to("km/s")

    def _get_sfr(self, particle_type: ParticleType) -> unyt_array:
        return particle_type.get_SWIFT_dataset(self.__file_object).star_formation_rates.to("Msun/yr")

    def _get_metalicities(self, particle_type: ParticleType) -> unyt_array:
        return particle_type.get_SWIFT_dataset(self.__file_object).metal_mass_fractions

    def _get_densities(self, particle_type: ParticleType) -> unyt_array:
        return particle_type.get_SWIFT_dataset(self.__file_object).densities.to("Msun/Mpc**3")

    def _get_temperatures(self, particle_type: ParticleType) -> unyt_array:
        return particle_type.get_SWIFT_dataset(self.__file_object).temperatures.to("K")
    
    @staticmethod
    def generate_filepaths(
       *snapshot_number_strings: str,
        directory: str,
        basename: str,
        file_extension: str = "hdf5",
        parallel_ranks: Union[List[int], None] = None
    ) -> Dict[
            str,
            Union[str, Dict[int, str]]
         ]:

        file_extension = file_extension.strip(".")
        parallel_insert = "" if parallel_ranks is None else ".{}"
        template = os.path.join(directory, f"{basename}{{}}{parallel_insert}.{file_extension}")

        results = {}
        for num in snapshot_number_strings:
            results[num] = os.path.abspath(template.format(num) if parallel_ranks is None else tuple([template.format(num, p) for p in parallel_ranks]))

        return results

    @staticmethod
    def scrape_filepaths(
        catalogue_directory: str
    ) -> Tuple[
            Tuple[
                str,
                Tuple[str, ...],
                Union[Tuple[int, ...], None],
                str
            ],
            ...
            ]:
        """
        Given the directory containing SWIFT snapshots, identify the file name information for the catalogue.
        """

        #pattern = re.compile(r'(?P<basename>[^/]+?)(?P<file_number>\d+)(?:\.(?P<parallel_id>\w+))?\.(?P<extension>\w+)$')
        pattern = re.compile(r'(?P<basename>[^/]+?)(?P<file_number>\d+)(?:\.(?P<parallel_id>(?:[1-9]\d*|0)))?\.(?P<extension>\w+)$')

        file_groups = {}

        for filename in os.listdir(catalogue_directory):
            match = pattern.match(filename)
            if match:
                basename = match.group("basename")
                file_number = match.group("file_number")
                parallel_id = match.group("parallel_id")
                extension = match.group("extension")

                # Handle edge cases
                if extension in ("siminfo", "units"):
                    continue

                if basename not in file_groups:
                    file_groups[basename] = { "extension" : extension, "number_strings" : [], "parallel_ids" : None }
                elif extension != file_groups[basename]["extension"]:
                    raise IOError("Inconsistent file extension for snapshots with the same basename.")

                if parallel_id is not None:
                    if len(file_groups[basename]["number_strings"]) == 0:
                        file_groups[basename]["parallel_ids"] = []
                    elif file_groups[basename]["parallel_ids"] is None:
                        raise IOError("Inconsistent parallel snapshot format for snapshots with the same basename.")

                    if parallel_id not in file_groups[basename]["parallel_ids"]:
                        file_groups[basename]["parallel_ids"].append(int(parallel_id))
                
                if file_number not in file_groups[basename]["number_strings"]:
                    file_groups[basename]["number_strings"].append(file_number)

        valid_basenames = []
        for basename in file_groups:
            parallel_component = "" if file_groups[basename]["parallel_ids"] is None else f".{file_groups[basename]['parallel_ids'][-1]}"
            test_file = os.path.join(catalogue_directory, f"{basename}{file_groups[basename]['number_strings'][-1]}{parallel_component}.{file_groups[basename]['extension']}")
            try:
                sw.load(test_file)
                valid_basenames.append(basename)
            except KeyError: pass # Ignore HDF5 files that aren't valid SWIFT snapshots
            except OSError:  pass # Ignore non-HDF5 files

        return tuple([(
            basename,
            tuple(file_groups[basename]["number_strings"]),
            tuple(file_groups[basename]["parallel_ids"]) if file_groups[basename]["parallel_ids"] is not None else None,
            file_groups[basename]["extension"]
        ) for basename in valid_basenames])
    
    @staticmethod
    def generate_filepaths_from_partial_info(
        directory: str,
        basename: Union[str, None] = None,
        snapshot_number_strings: Union[List[str], None] = None,
        file_extension: Union[str, None] = None,
        parallel_ranks: Union[List[int], None] = None
    ) -> Dict[
            str,
            Union[str, Dict[int, str]]
         ]:

        scraped_info = SnapshotSWIFT.scrape_filepaths(directory)

        valid_indexes = list(range(len(scraped_info)))

        if basename is not None:
            i = 0
            while i < len(valid_indexes):
                if scraped_info[valid_indexes[i]][0] != basename:
                    valid_indexes.pop(i)
                else:
                    i += 1

        if file_extension is not None:
            i = 0
            while i < len(valid_indexes):
                if scraped_info[valid_indexes[i]][3] != file_extension:
                    valid_indexes.pop(i)
                else:
                    i += 1

        if parallel_ranks is not None:
            i = 0
            while i < len(valid_indexes):
                if isinstance(scraped_info[valid_indexes[i]][2], str):
                    valid_indexes.pop(i)
                else:
                    i += 1

        if len(valid_indexes) == 0:
            raise FileNotFoundError("No snapshots match the partial specification.")
        if len(valid_indexes) > 1:
            raise IOError("Partial specification too general; more than one valid snapshot basename detected.")

        # Retain only correct set
        scraped_info = scraped_info[valid_indexes[0]]

        # Generate filepaths
        snapshot_file_locations: Dict[str, Union[str, Tuple[str, ...]]] = SnapshotSWIFT.generate_filepaths(
           *scraped_info[1],
            directory = directory,
            basename = scraped_info[0],
            file_extension = scraped_info[3],
            parallel_ranks = scraped_info[2]
        )

        if snapshot_number_strings is not None:
            for key in list(snapshot_file_locations.keys()):
                if key not in snapshot_number_strings:
                    del snapshot_file_locations[key]
            if len(snapshot_file_locations) != snapshot_number_strings:
                raise FileNotFoundError("Snapshot numbers provided not all present in directory.")
        
        if parallel_ranks is not None:
#            if isinstance(list(snapshot_file_locations.values())[0], str):
#                raise FileNotFoundError("Expected parallel snapshot files however no parallel components detected.")
            for key in snapshot_file_locations:
                for parallel_index in list(snapshot_file_locations[key].keys()):
                    if parallel_index not in parallel_ranks:
                        del snapshot_file_locations[key][parallel_index]
                if len(snapshot_file_locations[key]) != len(parallel_ranks):
                    raise FileNotFoundError("Snapshot parallel chunk indexes provided not all present in directory.")
        
        return snapshot_file_locations
    
    @staticmethod
    def get_snapshot_order(snapshot_file_info: List[str], reverse = False) -> List[str]:
        snapshot_file_info = list(snapshot_file_info)
        snapshot_file_info.sort(key = int, reverse = reverse)
        return snapshot_file_info