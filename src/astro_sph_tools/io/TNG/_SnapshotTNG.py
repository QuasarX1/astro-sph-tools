# SPDX-FileCopyrightText: 2025-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

import builtins
from collections.abc import Sequence, Callable
import os
import re
from typing import Any, TypeVar, cast as typing_cast

from mendeleev.models import Element
from mendeleev import H, He, C, N, O, Ne, Mg, Si, Fe
import numpy as np
import numpy.typing as npt
from unyt import unyt_array, unyt_quantity
from h5py import File as HDF5_File
import illustris_python as il
from QuasarCode import Settings, Console
from QuasarCode.MPI import MPI_Config, mpi_get_slice, mpi_gather_array, mpi_scatter_array, mpi_redistribute_array_evenly
import atomic_weights

from ..errors import SnipshotError, SnipshotOperationError, SnipshotFieldError
from ...data_structures._ParticleType import ParticleType
from ..data_structures._SnapshotBase import SnapshotBase
from ._sim_type import SimType_TNG
from ...tools._ArrayReorder import ArrayReorder_MPI

ATOMIC_MASS_UNIT = unyt_quantity(1.661e-24, units = "g")



def _read_particles(tng_base_directory: str, snap_num: str, particle_type: ParticleType, field: str, subset = None, table_index: int|None = None) -> np.ndarray:
    return il.snapshot.loadSubset(tng_base_directory, int(snap_num), particle_type.value, field, subset = subset, mdi = [table_index if table_index is not None else None])



T = TypeVar("T", bound=np.generic)

class SnapshotTNG(SnapshotBase[SimType_TNG]):
    """
    TNG snapshot data.
    """

    # Usefull constants

    EAGLE_MAX_GROUP_NUMBER = 1073741824 # 2**30

    # snapshot reader function

    @staticmethod
    def make_reader_object(root_directory: str, snapshot_number: str, subset: Any|None = None) -> Callable[[ParticleType, str], np.ndarray[tuple[int, ...], np.dtype[np.floating]]]:
        """
        Create a partial function for snapshot reading.
        """
        return lambda particle_type, field, table_index = None: _read_particles(root_directory, snapshot_number, particle_type, field, subset = subset, table_index = table_index)

    @property
    def _reader(self) -> Callable[[ParticleType, str], np.ndarray[tuple[int, ...], np.dtype[np.floating]]]:
        return self.__reader

    # Constructor

    def __init__(self, root_directory: str, snapshot_number: str) -> None:
        filepath_template = os.path.join(root_directory, f"snapdir_{snapshot_number}", f"snap_{snapshot_number}.{{}}.hdf5")
        first_file_path = filepath_template.format(0)
        Console.print_debug(f"Loading TNG snapshot from: \"{filepath_template.format("*")}\".")

        with HDF5_File(first_file_path, "r") as hdf5_reader:
            redshift = hdf5_reader["Header"].attrs["Redshift"]
            hubble_param = hdf5_reader["Header"].attrs["HubbleParam"]
            omega_baryon = hdf5_reader["Header"].attrs["OmegaBaryon"]
            self.__number_of_particles = hdf5_reader["Header"].attrs["NumPart_Total"]
            self.__dm_mass_internal_units = hdf5_reader["Header"].attrs["MassTable"][1]
            self.__box_size_internal_units = hdf5_reader["Header"].attrs["BoxSize"]
            self.__length_h_exp: float = hdf5_reader["PartType1/Coordinates"].attrs["h_scaling"]
            self.__length_a_exp: float = hdf5_reader["PartType1/Coordinates"].attrs["a_scaling"]
            self.__length_cgs_conversion_factor: float = hdf5_reader["PartType1/Coordinates"].attrs["to_cgs"]
            try:
                self.__mass_h_exp: float = hdf5_reader["PartType0/Masses"].attrs["h_scaling"]
                #self.__mass_a_exp: float = hdf5_reader["PartType0/Masses"].attrs["a_scaling"]
                self.__mass_cgs_conversion_factor: float = hdf5_reader["PartType0/Masses"].attrs["to_cgs"]
            except:
                # Just in case there aren't any gas particles, use the expected values for TNG
                self.__mass_h_exp: float = -1.0
                #self.__mass_a_exp: float = 0.0
                self.__mass_cgs_conversion_factor: float = 1.989E43
            self.__velocity_h_exp: float = hdf5_reader["PartType1/Velocities"].attrs["h_scaling"]
            self.__velocity_a_exp: float = hdf5_reader["PartType1/Velocities"].attrs["a_scaling"]
            self.__velocity_cgs_conversion_factor: float = hdf5_reader["PartType1/Velocities"].attrs["to_cgs"]

            self.__cgs_unit_conversion_factor_length: float = hdf5_reader["Units"].attrs["UnitLength_in_cm"]
            self.__cgs_unit_conversion_factor_mass: float = hdf5_reader["Units"].attrs["UnitMass_in_g"]
            self.__cgs_unit_conversion_factor_velocity: float = hdf5_reader["Units"].attrs["UnitVelocity_in_cm_per_s"]
            #self.__cgs_unit_conversion_factor_density: float = hdf5_reader["Units"].attrs["UnitDensity_in_cgs"]
            #self.__cgs_unit_conversion_factor_energy: float = hdf5_reader["Units"].attrs["UnitEnergy_in_cgs"]
            #self.__cgs_unit_conversion_factor_pressure: float = hdf5_reader["Units"].attrs["UnitPressure_in_cgs"]
            #self.__cgs_unit_conversion_factor_time: float = hdf5_reader["Units"].attrs["UnitTime_in_s"]

            assert self.__length_cgs_conversion_factor == self.__cgs_unit_conversion_factor_length
            assert self.__mass_cgs_conversion_factor == self.__cgs_unit_conversion_factor_mass
            assert self.__velocity_cgs_conversion_factor == self.__cgs_unit_conversion_factor_velocity

        self.__solar_metallicity: float =  0.0127
        self.__cgs_unit_conversion_factor_density: float = self.__cgs_unit_conversion_factor_mass / (self.__cgs_unit_conversion_factor_length**3)#TODO: check this!
        #self.__cgs_unit_conversion_factor_energy: float = hdf5_reader["Units"].attrs["UnitEnergy_in_cgs"]
        #self.__cgs_unit_conversion_factor_pressure: float = hdf5_reader["Units"].attrs["UnitPressure_in_cgs"]
        #self.__cgs_unit_conversion_factor_time: float = hdf5_reader["Units"].attrs["UnitTime_in_s"]

        subset: Any|None = None

        self.__number_of_particles_this_rank: np.ndarray[tuple[int], np.dtype[np.int64]]
        if MPI_Config.comm_size == 1:
            self.__number_of_particles_this_rank = self.__number_of_particles
        else:
            # If MPI is in use, select only a portion of the particles
            Console.print_debug("Calculating particle subset.")

            subset = il.snapshot.getSnapOffsets(root_directory, int(snapshot_number), 0, "Subhalo")#TODO: what values can this 'type' field have? https://github.com/illustristng/illustris_python/blob/master/illustris_python/snapshot.py#L161

            self.__number_of_particles_this_rank = np.zeros(shape = self.__number_of_particles.shape, dtype = np.int64)

            for particle_type in ParticleType.get_all():
                subset_slice: slice = mpi_get_slice(self.__number_of_particles[particle_type.value])
                subset["offsetType"][particle_type.value] = subset_slice.start
                subset["lenType"][particle_type.value] = subset_slice.stop - subset_slice.start
                self.__number_of_particles_this_rank[particle_type.value] = subset["lenType"][particle_type.value]

        self.__reader = SnapshotTNG.make_reader_object(root_directory, snapshot_number, subset)

        Console.print_debug("Initialising base class.")

        super().__init__(
            filepath = first_file_path,
            number = snapshot_number,
            redshift = redshift,
            hubble_param = hubble_param,
            omega_baryon = omega_baryon,
            expansion_factor = 1 / (1 + redshift),
            box_size = unyt_array(np.array([self.__box_size_internal_units, self.__box_size_internal_units, self.__box_size_internal_units], dtype = float) * (hubble_param ** self.__length_h_exp) * self.__length_cgs_conversion_factor, units = "cm").to("Mpc"),
            tracked_elements = (H, He, C, N, O, Ne, Mg, Si, Fe),
            snipshot = False # No snipshots in TNG
        )

        self.__dark_matter_particle_mass = self._convert_to_cgs_mass(self.__dm_mass_internal_units).to("Msun")

        Console.print_debug("Done creating TNG snapshot reader.")

    # Unyt handlers

    def make_cgs_data(self, cgs_units: str, data: np.ndarray[tuple[int, ...], np.dtype[np.float64]], h_exp: float, cgs_conversion_factor: float, a_exp: float = 0.0) -> unyt_array:
        """
        Convert raw data to a unyt_array object with the correct units.
        To retain data in co-moving space, omit the value for "a_exp".
        """
        return unyt_array(data * (self.h ** h_exp) * (self.a ** a_exp) * cgs_conversion_factor, units = cgs_units)
    
    def _convert_to_cgs_length(self, data: np.ndarray[tuple[int, ...], np.dtype[np.float64]], proper = False) -> unyt_array:
        return self.make_cgs_data(
            "cm",
            data,
            h_exp = self.__length_h_exp,
            cgs_conversion_factor = self.__length_cgs_conversion_factor,
            a_exp = self.__length_a_exp if proper else 0.0
        )
    
    def _convert_to_cgs_mass(self, data: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> unyt_array:
        return self.make_cgs_data(
            "g",
            data,
            h_exp = self.__mass_h_exp,
            cgs_conversion_factor = self.__mass_cgs_conversion_factor
        )
    
    def _convert_to_cgs_velocity(self, data: np.ndarray[tuple[int, ...], np.dtype[np.float64]], proper = False) -> unyt_array:
        return self.make_cgs_data(
            "cm/s",
            data,
            h_exp = self.__velocity_h_exp,
            cgs_conversion_factor = self.__velocity_cgs_conversion_factor,
            a_exp = self.__velocity_a_exp if proper else 0.0
        )

#    # Particle number accessors for internal use
#
#    def _get_number_of_particles(self) -> dict[ParticleType, int]:
#        return { p : int(self.__number_of_particles[p.value]) for p in ParticleType.get_all() }
#    def _get_number_of_particles_this_rank(self) -> dict[ParticleType, int]:
#        #return { p : (limits:=mpi_get_slice(n_parts:=int(self.__number_of_particles[p.value])).indices(n_parts))[1] - limits[0] for p in ParticleType.get_all() }
#        return { p : int(self.__number_of_particles_this_rank[p.value]) for p in ParticleType.get_all() }
#    def __check_number_of_particles_this_rank(self, particle_type_index: int) -> int:
#        try:
#            return self.__file_object.count_particles(particle_type_index)
#        except:
#            return 0
#    def __update_particle_count(self) -> None:
#        """
#        Used to update the per-rank and total particle numbers.
#        """
#        n_parttypes = len(self.__number_of_particles_this_rank)
#        for i in range(n_parttypes):
#            self.__number_of_particles_this_rank[i] = self.__check_number_of_particles_this_rank(i)
#        if MPI_Config.comm_size == 1:
#            self.__number_of_particles = self.__number_of_particles_this_rank
#        else:
#            self.__number_of_particles = sum(MPI_Config.comm.allgather(self.__number_of_particles_this_rank), start = np.zeros_like(self.__number_of_particles_this_rank))

    # Data reading methods

    def __read_dataset(self, particle_type: ParticleType, field_name: str, expected_dtype: type[T], expected_shape_after_first_dimension: tuple[int, ...] = tuple(), table_index: int|None = None) -> np.ndarray[tuple[int, ...], np.dtype[T]]:
        #return self.__data_read_reorder[particle_type](loaded_data if (loaded_data:=self.__file_object.read_dataset(particle_type.value, field_name)) is not None else np.empty((0,), dtype = expected_dtype))
        Console.print_verbose_info(f"Reading snapshot {particle_type.name} particle dataset \"{field_name}\".")
        Console.print_debug(f"Expecting data of type {expected_dtype}.")
        loaded_data = self.__reader(particle_type, field_name, table_index)
        processed_object = loaded_data if loaded_data is not None else np.empty((0, *expected_shape_after_first_dimension), dtype = expected_dtype)
        #Console.print_debug("Redistributing data.")
        #redistributed_data =  self.__data_read_reorder[particle_type](processed_object)
        Console.print_debug("Done reading data.")
        #return redistributed_data
        return processed_object

    def _get_IDs(self, particle_type: ParticleType) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        return typing_cast(np.ndarray[tuple[int], np.dtype[np.int64]], self.__read_dataset(particle_type, "ParticleIDs", expected_dtype = np.int64))

    def _get_smoothing_lengths(self, particle_type: ParticleType, use_proper_units: bool = False) -> unyt_array:
        return self._convert_to_cgs_length(self.__read_dataset(particle_type, "SubfindHsml", expected_dtype = np.float64), proper = use_proper_units).to("Mpc")

    def _get_masses(self, particle_type: ParticleType) -> unyt_array:
        if particle_type == ParticleType.dark_matter:
            return np.full(self.number_of_particles_this_rank(ParticleType.dark_matter), self.__dark_matter_particle_mass)
        return self._convert_to_cgs_mass(self.__read_dataset(particle_type, "Masses", expected_dtype = np.float64)).to("Msun")

    #TODO: this is a bug when MPI is used (total or rank-total mass???). Also an issue in EAGLE!
    def _get_total_mass(self, particle_type: ParticleType|None) -> unyt_quantity:
        if particle_type is None:
            return self._get_total_mass(ParticleType.dark_matter) + self._get_total_mass(ParticleType.gas) + self._get_total_mass(ParticleType.star) + self.get_total_black_hole_dynamical_mass()
        elif particle_type == ParticleType.dark_matter:
            return self.__dark_matter_particle_mass * self.number_of_particles(ParticleType.dark_matter)
        else:
            return self._get_masses(particle_type).sum()

    def _get_black_hole_subgrid_masses(self) -> unyt_array:
        return self._convert_to_cgs_mass(self.__read_dataset(ParticleType.black_hole, "BH_Mass", expected_dtype = np.float64)).to("Msun")

    def _get_black_hole_dynamical_masses(self) -> unyt_array:
        return self._convert_to_cgs_mass(self.__read_dataset(ParticleType.black_hole, "Masses", expected_dtype = np.float64)).to("Msun")

    def _get_total_black_hole_subgrid_mass(self) -> unyt_array:
        return self.get_black_hole_subgrid_masses().sum()

    def _get_total_black_hole_dynamical_mass(self) -> unyt_array:
        return self.get_black_hole_dynamical_masses().sum()

    def _get_positions(self, particle_type: ParticleType, use_proper_units: bool = False) -> unyt_array:
        return self._convert_to_cgs_length(self.__read_dataset(particle_type, "Coordinates", expected_dtype = np.float64, expected_shape_after_first_dimension = (3,)), proper = use_proper_units).to("Mpc")

    def _get_velocities(self, particle_type: ParticleType, use_proper_units: bool = False) -> unyt_array:
        return self._convert_to_cgs_velocity(self.__read_dataset(particle_type, "Velocities", expected_dtype = np.float64, expected_shape_after_first_dimension = (3,)), proper = use_proper_units).to("km/s")

    def _get_sfr(self) -> unyt_array:
        return unyt_array(self.__read_dataset(ParticleType.gas, "StarFormationRate", expected_dtype = np.float64), units = "Msun/yr")

    def _get_metallicities(self, particle_type: ParticleType, solar_units: bool, solar_metallicity: float|None) -> unyt_array:
        result = unyt_array(self.__read_dataset(particle_type, "GFM_Metallicity", expected_dtype = np.float64), units = None)
        if not solar_units:
            return result
        else:
            return result / (solar_metallicity if solar_metallicity is not None else self.__solar_metallicity)
    
    def _get_mean_enrichment_redshift(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("TNG does not compute mean enrichment redshift.")

    def _get_densities(self, particle_type: ParticleType, use_proper_units: bool) -> unyt_array:
        return self.make_cgs_data(
            "g/cm**3",
            self.__read_dataset(particle_type, "Density", expected_dtype = np.float64),
            h_exp = 2.0,
            cgs_conversion_factor = self.__cgs_unit_conversion_factor_density,
            a_exp = -3 if use_proper_units else 0
        ).to("Msun/Mpc**3")
    
    # default_abundance is not used as TNG does not have snipshots
    def _get_number_densities(self, particle_type: ParticleType, element: Element, use_proper_units: bool, default_abundance: float|None = None) -> unyt_array:
        if self.snipshot and default_abundance is None:
            raise SnipshotOperationError(
                operation_name = "get_number_densities",
                message = "Unable to read abundance data - snipshots do not contain this information. A \"default_abundance\" value must be specified."
            )

        table_index: int
        atomic_weight: unyt_quantity
        match element.atomic_number:
            case H.atomic_number:
                table_index = 0
                atomic_weight = atomic_weights.H  * ATOMIC_MASS_UNIT # Hydrogen
            case He.atomic_number:
                table_index = 1
                atomic_weight = atomic_weights.He * ATOMIC_MASS_UNIT # Helium
            case C.atomic_number:
                table_index = 2
                atomic_weight = atomic_weights.C  * ATOMIC_MASS_UNIT # Carbon
            case N.atomic_number:
                table_index = 3
                atomic_weight = atomic_weights.N  * ATOMIC_MASS_UNIT # Nitrogen
            case O.atomic_number:
                table_index = 4
                atomic_weight = atomic_weights.O  * ATOMIC_MASS_UNIT # Oxygen
            case Ne.atomic_number:
                table_index = 5
                atomic_weight = atomic_weights.Ne * ATOMIC_MASS_UNIT # Neon
            case Mg.atomic_number:
                table_index = 6
                atomic_weight = atomic_weights.Mg * ATOMIC_MASS_UNIT # Magnesium
            case Si.atomic_number:
                table_index = 7
                atomic_weight = atomic_weights.Si * ATOMIC_MASS_UNIT # Silicon
            case Fe.atomic_number:
                table_index = 8
                atomic_weight = atomic_weights.Fe * ATOMIC_MASS_UNIT # Iron
            case _:
                raise ValueError(f"Element \"{element}\" not tracked as part of TNG.")

        particle_densities = self.make_cgs_data(
            "g/cm**3",
            self.__read_dataset(particle_type, "Density", expected_dtype = np.float64),
            h_exp = 2.0,
            cgs_conversion_factor = self.__cgs_unit_conversion_factor_density,
            a_exp = -3 if use_proper_units else 0
        )

        return particle_densities * self.__read_dataset(particle_type, f"GFM_Metals", table_index = table_index, expected_dtype = np.float64) / atomic_weight

    #https://github.com/j-davies-astro/TNG_tools/blob/master/illustris_tools.py#L150
    def _get_temperatures(self, particle_type: ParticleType) -> unyt_array:

        m_p_cgs = 1.6726219e-24
        boltzmann_cgs = np.float64(1.38064852e-16)

        internal_energy = self.__read_dataset(particle_type, "InternalEnergy", expected_dtype = np.float64) * 1.0e10 # CGS conversion
        hydrogen_mass_fraction = self.__read_dataset(particle_type, "GFM_Metals", table_index = 0, expected_dtype = np.float64)
        electron_abundance = self.__read_dataset(particle_type, "ElectronAbundance", expected_dtype = np.float64)

        mu = (4.0 * m_p_cgs) / (1.0 + 3.0 * hydrogen_mass_fraction + 4.0 * hydrogen_mass_fraction * electron_abundance)
        #mu = (4.0 * m_p_cgs) / (1.0 + hydrogen_mass_fraction * (electron_abundance * 4.0 + 3.0))
        values = ((5.0 / 3.0) - 1.0) * mu * internal_energy / boltzmann_cgs

        return unyt_array(values, units = "K")#TODO: check that the unit is in fact Kelvin

    def _get_elemental_abundance(self, particle_type: ParticleType, element: Element) -> unyt_array:

        table_index: int
        match element.atomic_number:
            case H.atomic_number:  table_index = 0
            case He.atomic_number: table_index = 1
            case C.atomic_number:  table_index = 2
            case N.atomic_number:  table_index = 3
            case O.atomic_number:  table_index = 4
            case Ne.atomic_number: table_index = 5
            case Mg.atomic_number: table_index = 6
            case Si.atomic_number: table_index = 7
            case Fe.atomic_number: table_index = 8
            case _:
                raise ValueError(f"Element \"{element}\" not tracked as part of TNG.")

        return self.__read_dataset(particle_type, f"GFM_Metals", table_index = table_index, expected_dtype = np.float64)
