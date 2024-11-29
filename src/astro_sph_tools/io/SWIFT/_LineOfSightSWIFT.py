# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed
#TODO: this is just the same as the EAGLE logic - update for SWIFT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from ._LineOfSightBase import LineOfSightFileBase, LineOfSightBase

import numpy as np
from unyt import unyt_array
from typing import Any, Union, Tuple, Dict, Callable
import h5py as h5
import os

class LineOfSightFileSWIFT(LineOfSightFileBase):
    def __init__(self, filepath: str):
        with h5.File(filepath, "r") as file:
            number_of_sightlines = file["Header"].attrs["Number_of_sight_lines"]
            number_of_sightline_particles = np.array([file[f"LOS{i}"].attrs["Number_of_part_this_los"] for i in range(number_of_sightlines)], dtype = int)
            redshift = file["Header"].attrs["Redshift"]
            expansion_factor = file["Header"].attrs["ExpansionFactor"]
            hubble_param = file["Header"].attrs["HubbleParam"]

#            self.__cgs_unit_conversion_factor_density: float = file["Units"].attrs["UnitDensity_in_cgs"]
#            self.__cgs_unit_conversion_factor_energy: float = file["Units"].attrs["UnitEnergy_in_cgs"]
#            self.__cgs_unit_conversion_factor_length: float = file["Units"].attrs["UnitLength_in_cm"]
#            self.__cgs_unit_conversion_factor_mass: float = file["Units"].attrs["UnitMass_in_g"]
#            self.__cgs_unit_conversion_factor_pressure: float = file["Units"].attrs["UnitPressure_in_cgs"]
#            self.__cgs_unit_conversion_factor_time: float = file["Units"].attrs["UnitTime_in_s"]
#            self.__cgs_unit_conversion_factor_velocity: float = file["Units"].attrs["UnitVelocity_in_cm_per_s"]

        super().__init__(
            filepath = filepath,
            number_of_sightlines = number_of_sightlines,
            number_of_sightline_particles = number_of_sightline_particles,
            redshift = redshift,
            expansion_factor = expansion_factor,
            hubble_param = hubble_param
        )

    def get_sightline(self, index: int, cache_data: bool = True) -> "LineOfSightSWIFT":
        return LineOfSightSWIFT(self, self.get_sightline_length(index), index)

    @staticmethod
    def get_files(directory: str, prefix: str = "los_") -> Tuple[str, ...]:
        los_files = [os.path.join(directory, file) for file in list(*os.walk(directory))[2] if file[:len(prefix)] == "los_"]
        los_files.sort(key = lambda v: int(v[len(prefix):].rsplit(".", maxsplit = 1)[0]))
        return tuple(los_files)

class LineOfSightSWIFT(LineOfSightBase):
    def __init__(self, file_object: LineOfSightFileBase, number_of_particles: int, sightline_index: int):
        super().__init__(file_object = file_object, number_of_particles = number_of_particles)
        self.__sightline_index = sightline_index

    @property
    def file(self) -> LineOfSightFileSWIFT:
        return self.file
    
    def _read_cgs_field(self, field: str, cgs_units: Union[str, None], comoving = True) -> unyt_array:
        with self.file.get_readonly_file_handle() as file:
            field_object = file[f"LOS{self.__sightline_index}/{field}"]
            return unyt_array(field_object[:] * (self.file.h ** field_object.attrs["h-scale-exponent"]) * (self.file.a ** (1.0 if comoving else field_object.attrs["aexp-scale-exponent"])) * field_object.attrs["CGSConversionFactor"], units = cgs_units)

    def _read_positions(self, comoving = True) -> unyt_array:
        return self._read_cgs_field("Positions", "cm", comoving).to("Mpc")
    def _read_velocities(self, comoving = True) -> unyt_array:
        return self._read_cgs_field("Velocity", "cm/s", comoving).to("km/s")
    def _read_masses(self) -> unyt_array:
        return self._read_cgs_field("Mass", "g").to("Msun")
    def _read_metallicities(self) -> unyt_array:
        return self._read_cgs_field("Metallicity", None)
    def _read_temperatures(self) -> unyt_array:
        return self._read_cgs_field("Temperature", "K")
    def _read_densities(self, comoving = True) -> unyt_array:
        return self._read_cgs_field("Density", "g/(cm**3)", comoving).to("Msun/(Mpc**3)")
    def _read_smoothing_lengths(self, comoving = True) -> unyt_array:
        return self._read_cgs_field("SmoothingLength", "cm", comoving).to("Mpc")
