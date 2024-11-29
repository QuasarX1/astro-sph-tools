# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

import datetime
import time
from typing import Any, Union, Collection, List, Tuple, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import singledispatchmethod
import os

import numpy as np
import h5py as h5
from unyt import unyt_quantity
from QuasarCode import Console
from QuasarCode.Data import VersionInfomation
from QuasarCode.Tools import Struct, AutoProperty, TypedAutoProperty, NullableTypedAutoProperty, TypeShield, NestedTypeShield

from .. import VERSION
from .._ParticleType import ParticleType
from ._Output_Objects__backwards import ContraData

WAIT_SECONDS = 10*60 # 10 minutes



class SnapshotParticleFilter(Struct):
    """
    particle_type

    redshift

    snapshot_number

    filepath

    allowed_ids

    mask
    """

    particle_type: ParticleType = TypedAutoProperty[ParticleType](TypeShield(ParticleType),
                doc = "Particle type.")
    redshift: float = TypedAutoProperty[float](TypeShield(float),
                doc = "Redshift of the snapshot to which this filter applies.")
    snapshot_number: str = TypedAutoProperty[str](TypeShield(str),
                doc = "Number of the snapshot to which this filter applies. This is NOT the parallel file index!")
    filepath: str = TypedAutoProperty[str](TypeShield(str),
                doc = "The snapshot file to which this filter applies. The first file if snapshot is split into parallel components.")
    #allowed_ids: np.ndarray = TypedAutoProperty[np.ndarray](NestedTypeShield(np.ndarray, np.int64), # Nested type check fails if dimension is empty!
    allowed_ids: np.ndarray = AutoProperty[np.ndarray](
                doc = "IDs of selected particles.")
    mask: np.ndarray = TypedAutoProperty[np.ndarray](NestedTypeShield(np.ndarray, np.bool_),
                doc = "Numpy boolean mask for selected particles.")



class LineOfSightParticleFilter(Struct):
    """
    particle_type

    redshift

    file_name

    line_of_sight_index

    filepath

    allowed_ids

    mask
    """

    particle_type: ParticleType = TypedAutoProperty[ParticleType](TypeShield(ParticleType),
                doc = "Particle type.")
    redshift: float = TypedAutoProperty[float](TypeShield(float),
                doc = "Redshift of the line-of-sight to which this filter applies.")
    file_name: str = TypedAutoProperty[str](TypeShield(str),
                doc = "Name of the file including extension.")
    line_of_sight_index: int = TypedAutoProperty[float](TypeShield(int),
                doc = "Index of the line-of-sight within its file.")
    filepath: str = TypedAutoProperty[str](TypeShield(str),
                doc = "The line-of-sight file to which this filter applies.")
    #allowed_ids: np.ndarray = TypedAutoProperty[np.ndarray](NestedTypeShield(np.ndarray, np.int64), # Nested type check fails if dimension is empty!
    allowed_ids: np.ndarray = AutoProperty[np.ndarray](
                doc = "IDs of selected particles.")
    mask: np.ndarray = TypedAutoProperty[np.ndarray](NestedTypeShield(np.ndarray, np.bool_),
                doc = "Numpy boolean mask for selected particles.")



class ParticleFilterFile(Struct):
    """
    version

    date

    filepath

    description

    contra_file

    simulation_type

    snapshots_directory (nullable)

    line_of_sight_directory (nullable)

    #snapshot_filters[redshift: float][part_type: ParticleType]
    snapshot_filters[file: str][part_type: ParticleType]

    #line_of_sight_filters[redshift: float][index: int][part_type: ParticleType]
    line_of_sight_filters[file: str][index: int][part_type: ParticleType]
    """

    version: VersionInfomation = TypedAutoProperty[VersionInfomation](TypeShield(VersionInfomation),
                doc = "Contra version number.")
    date: datetime.date = TypedAutoProperty[datetime.date](TypeShield(datetime.date),
                doc = "Date of execution (start of program).")
    filepath: str = TypedAutoProperty[str](TypeShield(str),
                doc = "This output file location as generated.")
    description: str = TypedAutoProperty[str](TypeShield(str),
                doc = "Information about the file, such as how the filters were calculated.")
    contra_file: str = TypedAutoProperty[str](TypeShield(str),
                doc = "Contra output file used to generate this file.")
    simulation_type: str = TypedAutoProperty[str](TypeShield(str),
                doc = "Type of source simulation data.")
    snapshots_directory: str|None = NullableTypedAutoProperty[str](TypeShield(str),
                doc = "Directory containing associated snapshots.")
    line_of_sight_directory: str|None = NullableTypedAutoProperty[str](TypeShield(str),
                doc = "Directory containing associated line-of-sight files.")
    #snapshot_filters: Dict[float, Dict[ParticleType, SnapshotParticleFilter]] = TypedAutoProperty[Dict[float, Dict[ParticleType, SnapshotParticleFilter]]](TypeShield(dict),
    snapshot_filters: Dict[str, Dict[ParticleType, SnapshotParticleFilter]] = TypedAutoProperty[Dict[str, Dict[ParticleType, SnapshotParticleFilter]]](TypeShield(dict),
                doc = "Filters for snapshot particles.")
    #line_of_sight_filters: Dict[float, Dict[int, Dict[ParticleType, LineOfSightParticleFilter]]] = TypedAutoProperty[Dict[float, Dict[int, Dict[ParticleType, LineOfSightParticleFilter]]]](TypeShield(dict),
    line_of_sight_filters: Dict[str, Dict[int, Dict[ParticleType, LineOfSightParticleFilter]]] = TypedAutoProperty[Dict[str, Dict[int, Dict[ParticleType, LineOfSightParticleFilter]]]](TypeShield(dict),
                doc = "Filters for line-of-sight particles.")

    def __init__(self, filepath: str, allow_parallel_write: bool = False, **kwargs) -> None:
        date_specified = "date" in kwargs
        description_specified = "description" in kwargs
        super().__init__(
            filepath = filepath,
            snapshot_filters = kwargs.pop("snapshot_filters", {}),
            line_of_sight_filters = kwargs.pop("line_of_sight_filters", {}),
            **kwargs)
        if os.path.exists(filepath) and not allow_parallel_write:
            self.__writable = False
            self.__read()
        else:
            self.__writable = True
            self.version = VersionInfomation.from_string(VERSION)
            if not date_specified:
                self.date = datetime.date.today()
            if not description_specified:
                self.description = ""

        self.__parallel_write_mode = allow_parallel_write

    def get_contra_data(self) -> ContraData:
        return ContraData.load(self.contra_file)

    @property
    def has_snapshots(self) -> bool:
        return len(self.snapshot_filters) > 0

    @property
    def has_lines_of_sight(self) -> bool:
        return len(self.line_of_sight_filters) > 0
    
#    def get_snapshot_redshifts(self) -> Tuple[float, ...]:
#        return tuple(self.snapshot_filters.keys())
#    
#    def get_line_of_sight_redshifts(self) -> Tuple[float, ...]:
#        return tuple(self.line_of_sight_filters.keys())
#
#    def get_number_of_lines_of_sight(self, redshift: float) -> int:
#        if redshift not in self.line_of_sight_filters:
#            raise KeyboardInterrupt(f"No line-of-sight filters avalible for redshift {redshift}.")
#        return len(self.line_of_sight_filters[redshift]) # Lines of sight should have sequential indexes so we can just check the length of the dictionary
#
#    def get_snapshot_particle_types(self, redshift: float) -> Tuple[ParticleType, ...]:
#        if redshift not in self.snapshot_filters:
#            raise KeyboardInterrupt(f"No snapshot filters avalible for redshift {redshift}.")
#        return tuple(self.snapshot_filters[redshift].keys())
#
#    def get_line_of_sight_particle_types(self, redshift: float, index: int) -> Tuple[ParticleType, ...]:
#        if redshift not in self.line_of_sight_filters:
#            raise KeyboardInterrupt(f"No line-of-sight filters avalible for redshift {redshift}.")
#        if index not in self.line_of_sight_filters[redshift]:
#            raise KeyboardInterrupt(f"No line-of-sight filters avalible for index {index} at redshift {redshift}.")
#        return tuple(self.line_of_sight_filters[redshift][index].keys())
#
#    def get_snapshot_filter(self, redshift: float, part_type: ParticleType) -> SnapshotParticleFilter:
#        if redshift not in self.snapshot_filters:
#            raise KeyboardInterrupt(f"No snapshot filters avalible for redshift {redshift}.")
#        if part_type not in self.snapshot_filters[redshift]:
#            raise KeyboardInterrupt(f"No snapshot filters avalible for {part_type.name} particles at redshift {redshift}.")
#        return self.snapshot_filters[redshift][part_type]
#
#    def get_line_of_sight_filter(self, redshift: float, index: int, part_type: ParticleType = ParticleType.gas) -> LineOfSightParticleFilter:
#        if redshift not in self.line_of_sight_filters:
#            raise KeyboardInterrupt(f"No line-of-sight filters avalible for redshift {redshift}.")
#        if index not in self.line_of_sight_filters[redshift]:
#            raise KeyboardInterrupt(f"No line-of-sight filters avalible for index {index} at redshift {redshift}.")
#        if part_type not in self.line_of_sight_filters[redshift][index]:
#            raise KeyboardInterrupt(f"No line-of-sight filters avalible for {part_type.name} particles at redshift {redshift}.")
#        return self.line_of_sight_filters[redshift][index][part_type]

    def get_snapshot_file_names(self) -> Tuple[str, ...]:
        return tuple(self.snapshot_filters.keys())

    def get_line_of_sight_file_names(self) -> Tuple[str, ...]:
        return tuple(self.line_of_sight_filters.keys())

    def get_number_of_lines_of_sight(self, file_name: str) -> int:
        if file_name not in self.line_of_sight_filters:
            raise KeyboardInterrupt(f"No line-of-sight filters avalible for file {file_name}.")
        return len(self.line_of_sight_filters[file_name]) # Lines of sight should have sequential indexes so we can just check the length of the dictionary

    def get_snapshot_particle_types(self, file_name: str) -> Tuple[ParticleType, ...]:
        if file_name not in self.snapshot_filters:
            raise KeyboardInterrupt(f"No snapshot filters avalible for file {file_name}.")
        return tuple(self.snapshot_filters[file_name].keys())

    def get_line_of_sight_particle_types(self, file_name: str, index: int) -> Tuple[ParticleType, ...]:
        if file_name not in self.line_of_sight_filters:
            raise KeyboardInterrupt(f"No line-of-sight filters avalible for file {file_name}.")
        if index not in self.line_of_sight_filters[file_name]:
            raise KeyboardInterrupt(f"No line-of-sight filters avalible for LOS{index} in file {file_name}.")
        return tuple(self.line_of_sight_filters[file_name][index].keys())

    def get_snapshot_filter(self, file_name: str, part_type: ParticleType) -> SnapshotParticleFilter:
        if file_name not in self.snapshot_filters:
            raise KeyboardInterrupt(f"No snapshot filters avalible for file {file_name}.")
        if part_type not in self.snapshot_filters[file_name]:
            raise KeyboardInterrupt(f"No snapshot filters avalible for {part_type.name} particles in file {file_name}.")
        return self.snapshot_filters[file_name][part_type]

    def get_line_of_sight_filter(self, file_name: str, index: int, part_type: ParticleType = ParticleType.gas) -> LineOfSightParticleFilter:
        if file_name not in self.line_of_sight_filters:
            raise KeyboardInterrupt(f"No line-of-sight filters avalible for file {file_name}.")
        if index not in self.line_of_sight_filters[file_name]:
            raise KeyboardInterrupt(f"No line-of-sight filters avalible for LOS{index} in file {file_name}.")
        if part_type not in self.line_of_sight_filters[file_name][index]:
            raise KeyboardInterrupt(f"No line-of-sight filters avalible for {part_type.name} particles for LOS{index} in file {file_name}.")
        return self.line_of_sight_filters[file_name][index][part_type]

    def __read(self) -> None:
        with h5.File(self.filepath, "r") as file:
            header = file["Header"]
            snapshot_datasets = file["Snapshots"] if "Snapshots" in file else None
            los_datasets = file["LinesOfSight"] if "LinesOfSight" in file else None

            version_number_from_file = VersionInfomation.from_string(header.attrs["Version"])
            if version_number_from_file != VersionInfomation.from_string(VERSION):
                Console.print_verbose_warning(f"Particle filter file at \"{self.filepath}\" reported creation from version {version_number_from_file} which is older than the current version({VersionInfomation.from_string(VERSION)}).\nThis file may not be compattible with this version and if loaded, may contain erronius data or cause errors.")
            self.version = version_number_from_file
            self.date = datetime.datetime.fromtimestamp((header.attrs["Date"])).date()
            self.description = str(header.attrs["Description"])
            self.contra_file = str(header.attrs["ContraFile"])
            self.simulation_type = str(header.attrs["SimulationType"])
            if snapshot_datasets is not None:
                self.snapshots_directory = str(header.attrs["SnapshotsDirectory"])
#                snapshot_redshifts = header["SnapshotRedshifts"][:]
                snapshot_files = header["SnapshotFiles"][:]
            if los_datasets is not None:
                self.line_of_sight_directory = str(header.attrs["LinesOfSightDirectory"])
#                los_redshifts = header["LineOfSightRedshifts"][:]
                los_files = header["LineOfSightFiles"][:]

            if snapshot_datasets is not None:
#                for z in snapshot_redshifts:
                for snap_file in snapshot_files:
#                    z_float = float(z)
#                    self.snapshot_filters[z_float] = {}
                    self.snapshot_filters[snap_file] = {}
#                    snap_part_datasets = snapshot_datasets[z]
                    snap_part_datasets = snapshot_datasets[snap_file]
                    part_types = [ParticleType(p) for p in snap_part_datasets["ParticleTypes"][:]]
                    for part_type in part_types:
                        part_type_dataset = snap_part_datasets[part_type.common_hdf5_name]
#                        self.snapshot_filters[z_float][part_type] = SnapshotParticleFilter(
                        self.snapshot_filters[snap_file][part_type] = SnapshotParticleFilter(
                            particle_type = part_type,
#                            redshift = z_float,
                            redshift = snap_part_datasets.attrs["Redshift"],
                            snapshot_number = snap_part_datasets.attrs["SnapshotNumber"],
                            filepath = snap_part_datasets.attrs["SnapshotFile"],
                            allowed_ids = part_type_dataset["ParticleIDs"][:],
                            mask = np.array(part_type_dataset["Mask"][:], dtype = np.bool_)
                        )

            if los_datasets is not None:
#                for z in los_redshifts:
                for los_file in los_files:
#                    z_float = float(z)
#                    self.line_of_sight_filters[z_float] = {}
                    self.line_of_sight_filters[los_file] = {}
#                    los_file_group = los_datasets[z]
                    los_file_group = los_datasets[los_file]
                    los_indexes = los_file_group["Indexes"][:]
                    for los_index in los_indexes:
#                        self.line_of_sight_filters[z_float][los_index] = {}
                        self.line_of_sight_filters[los_file][los_index] = {}
                        los_part_datasets = los_file_group[f"LOS{los_index}"]
                        part_types = [ParticleType(p) for p in los_part_datasets["ParticleTypes"][:]]
                        for part_type in part_types:
                            part_type_dataset = los_part_datasets[part_type.common_hdf5_name]
#                            self.line_of_sight_filters[z_float][los_index][part_type] = LineOfSightParticleFilter(
                            self.line_of_sight_filters[los_file][los_index][part_type] = LineOfSightParticleFilter(
                                particle_type = part_type,
#                                redshift = z_float,
                                redshift = los_file_group.attrs["Redshift"],
#                                file_name = los_file_group.attrs["LineOfSightFileName"],
                                file_name = los_file,
                                line_of_sight_index = los_part_datasets.attrs["LineOfSightIndex"],
                                filepath = los_file_group.attrs["LineOfSightFile"],
                                allowed_ids = part_type_dataset["ParticleIDs"][:],
                                mask = np.array(part_type_dataset["Mask"][:], dtype = np.bool_)
                            )

    def save(self, filepath: str|None = None, use_mask_datatype = type|None) -> None:
        writing_for_first_time = False
        if not self.__writable and filepath is None:
            raise FileExistsError("Unable to save ParticleFilterFile object as a file already exists at the default filepath.\nRename or delete the existing file first, or specify a new filename.")
        elif filepath is not None:
            if os.path.exists(filepath) and not self.__parallel_write_mode:
                raise FileExistsError(f"Unable to save ParticleFilterFile object as a file already exists at \"{filepath}\".\nRename or delete the existing file first, or specify a new filename.")
        else:
            writing_for_first_time = True
            filepath = self.filepath

        file: h5.File|None = None
        if self.__parallel_write_mode:
            file_open_start_time = time.time()
            while time.time() - file_open_start_time < WAIT_SECONDS:
                try:
                    file = h5.File(filepath, "a")
                    break
                except (BlockingIOError, OSError):#TODO: figure out exactly which exception this is!
                    file = None
                    time.sleep(10)
            if file is None:
                Console.print_error(f"Unable to aquire write access to output file within the allotted time ({WAIT_SECONDS}s). Moving to next line-of-sight.")
                raise IOError("Unable to access file to write data.")
        else:
            file = h5.File(filepath, "w")

        with file:

            # Check that the header group hasn't already been created - don't overwrite if already present!
            write_header = True
            try:
                header = file.create_group("Header")
            except:
                write_header = False

            if self.has_snapshots:
                snapshot_datasets = file.create_group("Snapshots")
            if self.has_lines_of_sight:
                los_datasets = file.create_group("LinesOfSight")

            if write_header:
                header.attrs["Version"] =  str(self.version)
                header.attrs["Date"] = datetime.datetime(year = self.date.year, month = self.date.month, day = self.date.day).timestamp()
                header.attrs["Description"] = self.description
                header.attrs["ContraFile"] = self.contra_file
                header.attrs["SimulationType"] = self.simulation_type
                if self.has_snapshots:
                    header.attrs["SnapshotsDirectory"] = self.snapshots_directory
    #                header.create_dataset("SnapshotRedshifts", data = np.array([str(v) for v in self.get_snapshot_redshifts()], dtype = object), dtype = h5.special_dtype(vlen = str))
                    header.create_dataset("SnapshotFiles", data = np.array([str(v) for v in self.get_snapshot_file_names()], dtype = object), dtype = h5.special_dtype(vlen = str))
                if self.has_lines_of_sight:
                    header.attrs["LinesOfSightDirectory"] = self.line_of_sight_directory
    #                header.create_dataset("LineOfSightRedshifts", data = np.array([str(v) for v in self.get_line_of_sight_redshifts()], dtype = object), dtype = h5.special_dtype(vlen = str))
                    header.create_dataset("LineOfSightFiles", data = np.array([str(v) for v in self.get_line_of_sight_file_names()], dtype = object), dtype = h5.special_dtype(vlen = str))

            if self.has_snapshots:
#                for z in self.get_snapshot_redshifts():
                for snap_file_name in self.get_snapshot_file_names():
#                    z_str = str(z)
#                    snap_dataset = snapshot_datasets.create_group(z_str)
                    snap_dataset = snapshot_datasets.create_group(snap_file_name)
#                    snap_dataset.attrs["SnapshotNumber"] = self.snapshot_filters[z][self.get_snapshot_particle_types(z)[0]].snapshot_number
                    snap_dataset.attrs["SnapshotNumber"] = self.snapshot_filters[snap_file_name][self.get_snapshot_particle_types(snap_file_name)[0]].snapshot_number
#                    snap_dataset.attrs["SnapshotFile"] = self.snapshot_filters[z][self.get_snapshot_particle_types(z)[0]].filepath
                    snap_dataset.attrs["SnapshotFile"] = self.snapshot_filters[snap_file_name][self.get_snapshot_particle_types(snap_file_name)[0]].filepath
                    snap_dataset.attrs["Redshift"] = self.snapshot_filters[snap_file_name][self.get_snapshot_particle_types(snap_file_name)[0]].redshift
#                    snap_dataset.create_dataset("ParticleTypes", data = np.array([p.value for p in self.get_snapshot_particle_types(z)], dtype = int))
                    snap_dataset.create_dataset("ParticleTypes", data = np.array([p.value for p in self.get_snapshot_particle_types(snap_file_name)], dtype = np.int8))
#                    for part_type in self.get_snapshot_particle_types(z):
                    for part_type in self.get_snapshot_particle_types(snap_file_name):
                        part_type_dataset = snap_dataset.create_group(part_type.common_hdf5_name)
#                        part_type_dataset.attrs["TotalNumberOfParticles"] = self.snapshot_filters[z][part_type].mask.shape[0]
#                        part_type_dataset.attrs["NumberOfParticles"] = self.snapshot_filters[z][part_type].mask.sum()
#                        part_type_dataset.create_dataset("ParticleIDs", data = self.snapshot_filters[z][part_type].allowed_ids)
#                        part_type_dataset.create_dataset("Mask", data = self.snapshot_filters[z][part_type].mask)
                        part_type_dataset.attrs["TotalNumberOfParticles"] = self.snapshot_filters[snap_file_name][part_type].mask.shape[0]
                        part_type_dataset.attrs["NumberOfParticles"] = self.snapshot_filters[snap_file_name][part_type].mask.sum()
                        part_type_dataset.create_dataset("ParticleIDs", data = self.snapshot_filters[snap_file_name][part_type].allowed_ids)
                        part_type_dataset.create_dataset("Mask", data = self.snapshot_filters[snap_file_name][part_type].mask if use_mask_datatype is None else np.array(self.snapshot_filters[snap_file_name][part_type].mask, dtype = use_mask_datatype))

            if self.has_lines_of_sight:
#                for z in self.get_line_of_sight_redshifts():
                for los_file_name in self.get_line_of_sight_file_names():
#                    z_str = str(z)
#                    los_file_dataset = los_datasets.create_group(z_str)
#                    los_file_dataset.attrs["LineOfSightFileName"] = self.line_of_sight_filters[z][0][self.get_line_of_sight_particle_types(z, 0)[0]].file_name
#                    los_file_dataset.attrs["LineOfSightFile"] = self.line_of_sight_filters[z][0][self.get_line_of_sight_particle_types(z, 0)[0]].filepath
#                    los_indexes = list(range(self.get_number_of_lines_of_sight(z)))
                    los_file_dataset = los_datasets.create_group(los_file_name)
#                    los_file_dataset.attrs["LineOfSightFileName"] = self.line_of_sight_filters[los_file_name][0][self.get_line_of_sight_particle_types(los_file_name, 0)[0]].file_name
                    los_file_dataset.attrs["LineOfSightFile"] = self.line_of_sight_filters[los_file_name][0][self.get_line_of_sight_particle_types(los_file_name, 0)[0]].filepath
                    los_file_dataset.attrs["Redshift"] = self.line_of_sight_filters[los_file_name][0][self.get_line_of_sight_particle_types(los_file_name, 0)[0]].redshift
                    los_indexes = list(range(self.get_number_of_lines_of_sight(los_file_name)))
                    los_file_dataset.create_dataset("Indexes", data = np.array(los_indexes, dtype = int))
                    for los_index in los_indexes:
                        los_dataset = los_file_dataset.create_group(f"LOS{los_index}")
#                        los_dataset.attrs["LineOfSightIndex"] = self.line_of_sight_filters[z][0][self.get_line_of_sight_particle_types(z, 0)[0]].line_of_sight_index
#                        los_dataset.create_dataset("ParticleTypes", data = np.array([p.value for p in self.get_line_of_sight_particle_types(z, los_index)], dtype = int))
                        los_dataset.attrs["LineOfSightIndex"] = self.line_of_sight_filters[los_file_name][0][self.get_line_of_sight_particle_types(los_file_name, 0)[0]].line_of_sight_index
                        los_dataset.create_dataset("ParticleTypes", data = np.array([p.value for p in self.get_line_of_sight_particle_types(los_file_name, los_index)], dtype = np.int8))
#                        for part_type in self.get_line_of_sight_particle_types(z, los_index):
                        for part_type in self.get_line_of_sight_particle_types(los_file_name, los_index):
                            part_type_dataset = los_dataset.create_group(part_type.common_hdf5_name)
#                            part_type_dataset.attrs["TotalNumberOfParticles"] = self.line_of_sight_filters[z][los_index][part_type].mask.shape[0]
#                            part_type_dataset.attrs["NumberOfParticles"] = self.line_of_sight_filters[z][los_index][part_type].mask.sum()
#                            part_type_dataset.create_dataset("ParticleIDs", data = self.line_of_sight_filters[z][los_index][part_type].allowed_ids)
#                            part_type_dataset.create_dataset("Mask", data = self.line_of_sight_filters[z][los_index][part_type].mask)
                            part_type_dataset.attrs["TotalNumberOfParticles"] = self.line_of_sight_filters[los_file_name][los_index][part_type].mask.shape[0]
                            part_type_dataset.attrs["NumberOfParticles"] = self.line_of_sight_filters[los_file_name][los_index][part_type].mask.sum()
                            part_type_dataset.create_dataset("ParticleIDs", data = self.line_of_sight_filters[los_file_name][los_index][part_type].allowed_ids)
                            part_type_dataset.create_dataset("Mask", data = self.line_of_sight_filters[los_file_name][los_index][part_type].mask if use_mask_datatype is None else np.array(self.line_of_sight_filters[los_file_name][los_index][part_type].mask, dtype = use_mask_datatype))

        if writing_for_first_time:
            # Mark the default filepath as being written to
            self.__writable = False
