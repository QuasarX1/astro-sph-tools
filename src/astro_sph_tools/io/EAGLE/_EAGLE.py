# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

import os
import re
from collections.abc import Iterable, Iterator
from typing import TypeVar, Generic
from functools import singledispatchmethod

import numpy as np
from QuasarCode import Console

from ..data_structures._tree_structures import SimulationFileTreeBase, SimulationFileTreeLeafBase
from ._SnapshotEAGLE import SnapshotEAGLE
from ._CatalogueSUBFIND import CatalogueSUBFIND



class SnapOrSnipFiles_EAGLE(SimulationFileTreeLeafBase[SnapshotEAGLE]):
    def __init__(self, number: str, tag: str, filepath: str, all_filepaths: Iterable[str]):
        self.__number = number
        self.__tag = tag
        self.__filepath = filepath
        self.__filepaths = tuple(all_filepaths)
        self.__approximate_redshift = float(".".join(self.__tag.split("z")[1].split("p")))
    def __len__(self) -> int:
        return len(self.__filepaths)
    def load(self) -> SnapshotEAGLE:
        return SnapshotEAGLE(self.__filepath)
    @property
    def number(self) -> str:
        return self.__number
    @property
    def number_numerical(self) -> int:
        return int(self.__number)
    @property
    def tag(self) -> str:
        return self.__tag
    @property
    def tag_redshift(self) -> float:
        return self.__approximate_redshift
    @property
    def tag_expansion_factor(self) -> float:
        return 1 / (1 + self.tag_redshift)
    @property
    def filepaths(self) -> tuple[str, ...]:
        return self.__filepaths
    @property
    def filepath(self) -> str:
        return self.__filepath
T = TypeVar("T", bound = SnapOrSnipFiles_EAGLE)

class SimulationSnapOrSnipFiles_EAGLE(SimulationFileTreeBase[SnapshotEAGLE], Generic[T]):
    _snapshot_pattern = re.compile(r'.*snapshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]snap_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
    _snipshot_pattern = re.compile(r'.*snipshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]snip_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
    def __init__(self, directory: str, snipshots: bool, file_type: type[T], skip_numbers: list[str]|None = None):
        super().__init__(directory)

        if skip_numbers is None:
            skip_numbers = []

        self.__file_type_string = "snapshot" if not snipshots else "snipshot"

        pattern = self._snapshot_pattern if not snipshots else self._snipshot_pattern

        self.__scraped_file_info: dict[str, tuple[str, str, list[int], str]] = {}

        for root, _, files in os.walk(self.directory):
            for filename in files:
                match = pattern.match(os.path.join(root, filename))
                if match:
                    number = match.group("number")
                    if number in skip_numbers:
                        continue

                    redshift_int = match.group("redshift_int")
                    redshift_dec = match.group("redshift_dec")
                    parallel_index = int(match.group("parallel_index"))
                    extension = match.group("extension")

                    tag = f"{number}_z{redshift_int}p{redshift_dec}"
                    basename = os.path.join(f"snapshot_{tag}", f"snap_{tag}") if not snipshots else os.path.join(f"snipshot_{tag}", f"snip_{tag}")

                    if tag not in self.__scraped_file_info:
                        self.__scraped_file_info[tag] = (basename, number, [parallel_index], extension)
                    else:
                        assert basename == self.__scraped_file_info[tag][0]
                        assert number == self.__scraped_file_info[tag][1]
                        assert extension == self.__scraped_file_info[tag][3]
                        self.__scraped_file_info[tag][2].append(parallel_index)

        for tag in self.__scraped_file_info:
            self.__scraped_file_info[tag][2].sort()

        filepaths: dict[str, list[str]] = { tag : [os.path.join(self.directory, f"{self.__scraped_file_info[tag][0]}.{i}.{self.__scraped_file_info[tag][3]}") for i in self.__scraped_file_info[tag][2]] for tag in self.__scraped_file_info }
        self.__files: list[T] = [file_type(self.__scraped_file_info[tag][1], tag, filepaths[tag][0], filepaths[tag]) for tag in self.__scraped_file_info]

        self.__files.sort(key = lambda v: v.number_numerical)
        self.__file_lookup_by_number = { f.number : f for f in self.__files }
        self.__file_lookup_by_tag = { f.tag : f for f in self.__files }

    def __iter__(self) -> Iterator[T]:
        return iter(self.__files)

    def __len__(self) -> int:
        return len(self.__files)

    @singledispatchmethod
    def __get_item__(self, key: int) -> T:
        return self.__files[key]

    __get_item__.register(slice)
    def _(self, key: slice) -> tuple[T, ...]:
        return tuple(self.__files[key])

    def get_info(self) -> tuple[T, ...]:
        return tuple(self.__files)

    def get_numbers(self) -> tuple[str, ...]:
        return tuple([f.number for f in self.__files])

    def get_tags(self) -> tuple[str, ...]:
        return tuple([f.number for f in self.__files])

    def get_by_number(self, number: str) -> T:
        if number not in self.__file_lookup_by_number:
            raise KeyError(f"{self.__file_type_string.title()} number \"{number}\" not avalible (make sure the input datatype is a string).")
        return self.__file_lookup_by_number[number]

    def get_by_tag(self, tag: str) -> T:
        if tag not in self.__file_lookup_by_tag:
            raise KeyError(f"{self.__file_type_string.title()} tag \"{tag}\" not avalible.")
        return self.__file_lookup_by_tag[tag]
    
    def find_file_number_from_redshift(self, redshift: float) -> str:
        file_numbers = np.array(self.get_numbers(), dtype = str)
        file_numbers = file_numbers[np.array([float(v) for v in file_numbers], dtype = float).argsort()]
        file_redshifts = np.array([self.get_by_number(file_number).tag_redshift for file_number in file_numbers], dtype = float)
        prior_files_mask = file_redshifts >= redshift
        if prior_files_mask.sum() == 0:
            raise FileNotFoundError(f"Unable to find search data for a file with a redshift of (or exceding) {redshift}.\nThe first file has a redshift of {file_redshifts[0]}.")
        selected_redshift = file_redshifts[prior_files_mask][-1]
        if redshift >= 1.0 and redshift - selected_redshift > 0.5 or redshift < 1.0 and redshift - selected_redshift > 0.1:
            Console.print_verbose_warning(f"Attempted to find data at z={redshift} but only managed to retrive data for z=~{selected_redshift}.")
        return str(file_numbers[prior_files_mask][-1])



class SnapshotFiles_EAGLE(SnapOrSnipFiles_EAGLE):
    pass

class SimulationSnapshotFiles_EAGLE(SimulationSnapOrSnipFiles_EAGLE[SnapshotFiles_EAGLE]):
    def __init__(self, directory: str, skip_numbers: list[str]|None = None):
        super().__init__(
            directory,
            snipshots = False,
            file_type = SnapshotFiles_EAGLE,
            skip_numbers = skip_numbers
        )

    def __iter__(self) -> Iterator[SnapshotFiles_EAGLE]:
        return super().__iter__()



class SnipshotFiles_EAGLE(SnapOrSnipFiles_EAGLE):
    pass

class SimulationSnipshotFiles_EAGLE(SimulationSnapOrSnipFiles_EAGLE[SnipshotFiles_EAGLE]):
    def __init__(self, directory: str, skip_numbers: list[str]|None = None):
        super().__init__(
            directory,
            snipshots = True,
            file_type = SnipshotFiles_EAGLE,
            skip_numbers = skip_numbers
        )

    def __iter__(self) -> Iterator[SnipshotFiles_EAGLE]:
        return super().__iter__()



class SnapOrSnipCatalogueFiles_EAGLE(SimulationFileTreeLeafBase[CatalogueSUBFIND]):
    def __init__(self, number: str, tag: str, membership_filepaths: list[str], properties_filepaths: list[str], raw_particlre_data_info: SnapOrSnipFiles_EAGLE):
        self.__number = number
        self.__tag = tag
        self.__membership_filepaths = membership_filepaths
        self.__properties_filepaths = properties_filepaths
        self.__snapshot_info: SnapOrSnipFiles_EAGLE = raw_particlre_data_info
        self.__approximate_redshift = float(".".join(self.__tag.split("z")[1].split("p")))
    def __len__(self) -> int:
        return len(self.__properties_filepaths)
    def load(self) -> CatalogueSUBFIND:
        return CatalogueSUBFIND(self.__membership_filepaths, self.__properties_filepaths, self.__snapshot_info.load())
    @property
    def number(self) -> str:
        return self.__number
    @property
    def number_numerical(self) -> int:
        return int(self.__number)
    @property
    def tag(self) -> str:
        return self.__tag
    @property
    def tag_redshift(self) -> float:
        return self.__approximate_redshift
    @property
    def tag_expansion_factor(self) -> float:
        return 1 / (1 + self.tag_redshift)
    @property
    def filepaths(self) -> tuple[str, ...]:
        return tuple(self.__properties_filepaths)
    @property
    def filepath(self) -> str:
        return self.__properties_filepaths[0]
    @property
    def membership_filepath(self) -> str:
        return self.__membership_filepaths[0]
    @property
    def properties_filepath(self) -> str:
        return self.__properties_filepaths[0]
    @property
    def membership_filepaths(self) -> tuple[str, ...]:
        return tuple(self.__membership_filepaths)
    @property
    def properties_filepaths(self) -> tuple[str, ...]:
        return tuple(self.__properties_filepaths)

U = TypeVar("U", bound = SnapOrSnipCatalogueFiles_EAGLE)
class SimulationSnapOrSnipCatalogueFiles_EAGLE(SimulationFileTreeBase[CatalogueSUBFIND], Generic[U, T]):
    _snapshot_catalogue_membership_pattern = re.compile(r'.*particledata_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]eagle_subfind_particles_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
    _snipshot_catalogue_membership_pattern = re.compile(r'.*particledata_snip_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]eagle_subfind_snip_particles_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
    _snapshot_catalogue_properties_pattern = re.compile(r'.*groups_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]eagle_subfind_tab_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
    _snipshot_catalogue_properties_pattern = re.compile(r'.*groups_snip_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]eagle_subfind_snip_tab_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
    def __init__(self, directory: str, snipshots: bool, file_type: type[U], raw_particle_data_info: SimulationSnapOrSnipFiles_EAGLE[T], skip_numbers: list[str]|None = None):
        super().__init__(directory)

        if skip_numbers is None:
            skip_numbers = []

        self.__file_type_string = "snapshot catalogue" if not snipshots else "snipshot catalogue"

        membership_pattern = self._snapshot_catalogue_membership_pattern if not snipshots else self._snipshot_catalogue_membership_pattern
        properties_pattern = self._snapshot_catalogue_properties_pattern if not snipshots else self._snipshot_catalogue_properties_pattern
        
        self.__scraped_properties_info: dict[str, tuple[str, str, str, list[int]]] = {}
        self.__scraped_membership_info: dict[str, tuple[str, str, str, list[int]]] = {}

        for root, _, files in os.walk(directory):
            for filename in files:
                filepath = os.path.join(root, filename)
                membership_match = membership_pattern.match(filepath)
                properties_match = properties_pattern.match(filepath)

                if membership_match or properties_match:
                    match = membership_match if membership_match else properties_match
                    is_properties = bool(properties_match)

                    number = match.group("number")
                    if number in skip_numbers:
                        continue

                    redshift_int = match.group("redshift_int")
                    redshift_dec = match.group("redshift_dec")
                    parallel_index = int(match.group("parallel_index"))
                    extension = match.group("extension")

                    tag = f"{number}_z{redshift_int}p{redshift_dec}"
                    if is_properties:
                        basename = os.path.join(f"groups_{tag}", f"eagle_subfind_tab_{tag}") if not snipshots else os.path.join(f"groups_snip_{tag}", f"eagle_subfind_snip_tab_{tag}")
                        if tag not in self.__scraped_properties_info:
                            self.__scraped_properties_info[tag] = (number, basename, extension, [parallel_index])
                        else:
                            assert basename == self.__scraped_properties_info[tag][1]
                            assert extension == self.__scraped_properties_info[tag][2]
                            self.__scraped_properties_info[tag][3].append(parallel_index)
                    else:
                        basename = os.path.join(f"particledata_{tag}", f"eagle_subfind_particles_{tag}") if not snipshots else os.path.join(f"particledata_snip_{tag}", f"eagle_subfind_snip_particles_{tag}")
                        if tag not in self.__scraped_membership_info:
                            self.__scraped_membership_info[tag] = (number, basename, extension, [parallel_index])
                        else:
                            assert basename == self.__scraped_membership_info[tag][1]
                            assert extension == self.__scraped_membership_info[tag][2]
                            self.__scraped_membership_info[tag][3].append(parallel_index)

        for tag in self.__scraped_properties_info:
            self.__scraped_properties_info[tag][3].sort()
            self.__scraped_membership_info[tag][3].sort()

        self.__files: list[U] = [
            file_type(
                self.__scraped_properties_info[tag][0],
                tag,
                [os.path.join(self.directory, f"{self.__scraped_membership_info[tag][1]}.{i}.{self.__scraped_membership_info[tag][2]}") for i in self.__scraped_membership_info[tag][3]],
                [os.path.join(self.directory, f"{self.__scraped_properties_info[tag][1]}.{i}.{self.__scraped_properties_info[tag][2]}") for i in self.__scraped_properties_info[tag][3]],
                raw_particle_data_info.get_by_tag(tag)
            )
            for tag
            in self.__scraped_properties_info
        ]

        self.__files.sort(key = lambda v: v.number_numerical)
        self.__file_lookup_by_number = { f.number : f for f in self.__files }
        self.__file_lookup_by_tag = { f.tag : f for f in self.__files }

    def __iter__(self) -> Iterator[U]:
        return iter(self.__files)
    
    def __len__(self) -> int:
        return len(self.__files)

    @singledispatchmethod
    def __get_item__(self, key: int) -> U:
        return self.__files[key]

    __get_item__.register(slice)
    def _(self, key: slice) -> tuple[U, ...]:
        return tuple(self.__files[key])

    def get_info(self) -> tuple[U, ...]:
        return tuple(self.__files)

    def get_numbers(self) -> tuple[str, ...]:
        return tuple([f.number for f in self.__files])

    def get_tags(self) -> tuple[str, ...]:
        return tuple([f.number for f in self.__files])

    def get_by_number(self, number: str) -> U:
        if number not in self.__file_lookup_by_number:
            raise KeyError(f"{self.__file_type_string.title()} number \"{number}\" not avalible.")
        return self.__file_lookup_by_number[number]

    def get_by_tag(self, tag: str) -> U:
        if tag not in self.__file_lookup_by_tag:
            raise KeyError(f"{self.__file_type_string.title()} tag \"{tag}\" not avalible.")
        return self.__file_lookup_by_tag[tag]
    
    def find_file_number_from_redshift(self, redshift: float) -> str:
        file_numbers = np.array(self.get_numbers(), dtype = str)
        file_numbers = file_numbers[np.array([float(v) for v in file_numbers], dtype = float).argsort()]
        file_redshifts = np.array([self.get_by_number(file_number).redshift for file_number in file_numbers], dtype = float)
        prior_files_mask = file_redshifts >= redshift
        if prior_files_mask.sum() == 0:
            raise FileNotFoundError(f"Unable to find search data for a file with a redshift of (or exceding) {redshift}.\nThe first file has a redshift of {file_redshifts[0]}.")
        selected_redshift = file_redshifts[prior_files_mask][-1]
        if redshift >= 1.0 and redshift - selected_redshift > 0.5 or redshift < 1.0 and redshift - selected_redshift > 0.1:
            Console.print_verbose_warning(f"Attempted to find data at z={redshift} but only managed to retrive data for z=~{selected_redshift}.")
        return str(file_numbers[prior_files_mask][-1])



class SnapshotCatalogueFiles_EAGLE(SnapOrSnipCatalogueFiles_EAGLE):
    pass

class SimulationSnapshotCatalogueFiles_EAGLE(SimulationSnapOrSnipCatalogueFiles_EAGLE[SnapshotCatalogueFiles_EAGLE, SnapshotFiles_EAGLE]):
    def __init__(self, directory: str, snapshot_info: SimulationSnapshotFiles_EAGLE, skip_numbers: list[str]|None = None):
        super().__init__(
            directory,
            snipshots = False,
            file_type = SnapshotCatalogueFiles_EAGLE,
            raw_particle_data_info = snapshot_info,
            skip_numbers = skip_numbers
        )

    def __iter__(self) -> Iterator[SnapshotCatalogueFiles_EAGLE]:
        return super().__iter__()



class SnipshotCatalogueFiles_EAGLE(SnapOrSnipCatalogueFiles_EAGLE):
    pass

class SimulationSnipshotCatalogueFiles_EAGLE(SimulationSnapOrSnipCatalogueFiles_EAGLE[SnipshotCatalogueFiles_EAGLE, SnipshotFiles_EAGLE]):
    def __init__(self, directory: str, snipshot_info: SimulationSnipshotFiles_EAGLE, skip_numbers: list[str]|None = None):
        super().__init__(
            directory,
            snipshots = True,
            file_type = SnipshotCatalogueFiles_EAGLE,
            raw_particle_data_info = snipshot_info,
            skip_numbers = skip_numbers
        )

    def __iter__(self) -> Iterator[SnipshotCatalogueFiles_EAGLE]:
        return super().__iter__()



class FileTreeScraper_EAGLE(object):
    def __init__(self, filepath: str, skip_snapshot_numbers: list[str]|None = None, skip_snipshot_numbers: list[str]|None = None) -> None:
        self.__root_directory = filepath
        self.__snapshots = SimulationSnapshotFiles_EAGLE(filepath, skip_numbers = skip_snapshot_numbers)
        self.__snipshots = SimulationSnipshotFiles_EAGLE(filepath, skip_numbers = skip_snipshot_numbers)
        self.__snapshot_catalogues = SimulationSnapshotCatalogueFiles_EAGLE(filepath, self.__snapshots, skip_numbers = skip_snapshot_numbers)
        self.__snipshot_catalogues = SimulationSnipshotCatalogueFiles_EAGLE(filepath, self.__snipshots, skip_numbers = skip_snipshot_numbers)

    @property
    def snapshots(self) -> SimulationSnapshotFiles_EAGLE:
        return self.__snapshots

    @property
    def snipshots(self) -> SimulationSnipshotFiles_EAGLE:
        return self.__snipshots

    @property
    def catalogues(self) -> SimulationSnapshotCatalogueFiles_EAGLE:
        return self.__snapshot_catalogues

    @property
    def snipshot_catalogues(self) -> SimulationSnipshotCatalogueFiles_EAGLE:
        return self.__snipshot_catalogues

    @staticmethod
    def split_filepath(filepath: str) -> tuple[str, str]:
        """
        Split the filepath of an EAGLE data file into the
        """
        absolute_filepath = os.path.abspath(filepath)
        folder, file = os.path.split(absolute_filepath)
        absolute_directory, folder = os.path.split(folder)
        return (absolute_directory, os.path.join(folder, file))

    @staticmethod
    def directory_from_filepath(filepath: str) -> str:
        """
        Gets the path of the EAGLE data directory from a path to a simulation data file.
        """
        return FileTreeScraper_EAGLE.split_filepath(filepath)[0]

    @staticmethod
    def relitive_filepath(filepath: str) -> str:
        """
        Get the component of a filepath relitive to the simulation data directory.
        """
        return FileTreeScraper_EAGLE.split_filepath(filepath)[1]

    @staticmethod
    def make_filepath_with_root(directory: str, relitive_filepath: str) -> str:
        """
        Get the path to a data file given the relitive location of the file to the root directory and the path to the simulation root.
        """
        return os.path.join(directory, relitive_filepath)
    
    def make_filepath(self, relitive_filepath: str) -> str:
        """
        Get the path to a data file given the relitive location of the file to the root directory.
        """
        return FileTreeScraper_EAGLE.make_filepath_with_root(self.__root_directory, relitive_filepath)

    @staticmethod
    def get_alternative_filepath_with_root(directory: str, filepath: str) -> str:
        """
        Get the path to a data file given the original location of the file and the new path to the simulation root.
        """
        return FileTreeScraper_EAGLE.make_filepath_with_root(directory, FileTreeScraper_EAGLE.relitive_filepath(filepath))

    def get_alternative_filepath(self, filepath: str) -> str:
        """
        Get the path to a data file given the original location of the file and the new path to the simulation root.
        """
        return FileTreeScraper_EAGLE.get_alternative_filepath_with_root(self.__root_directory, filepath)
