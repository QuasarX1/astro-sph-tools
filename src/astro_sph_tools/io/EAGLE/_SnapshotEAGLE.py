# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

import builtins
from collections.abc import Sequence
import os
import re
from typing import Any, TypeVar, cast as typing_cast

from mendeleev.models import Element
from mendeleev import H, He, C, N, O, Ne, Mg, Si, Fe
import numpy as np
import numpy.typing as npt
from unyt import unyt_array, unyt_quantity
from h5py import File as HDF5_File
from pyread_eagle import EagleSnapshot
from QuasarCode import Settings, Console
from QuasarCode.MPI import MPI_Config, mpi_get_slice, mpi_gather_array, mpi_scatter_array, mpi_redistribute_array_evenly
import atomic_weights

from ..errors import SnipshotError, SnipshotOperationError, SnipshotFieldError
from ...data_structures._ParticleType import ParticleType
from ..data_structures._SnapshotBase import SnapshotBase
from ._sim_type import SimType_EAGLE
from ...tools._ArrayReorder import ArrayReorder_MPI

ATOMIC_MASS_UNIT = unyt_quantity(1.661e-24, units = "g")



T = TypeVar("T", bound=np.generic)

class SnapshotEAGLE(SnapshotBase[SimType_EAGLE]):
    """
    EAGLE snapshot data.
    """

    # Usefull constants

    EAGLE_MAX_GROUP_NUMBER = 1073741824 # 2**30

    # pyread_eagle reader object

    __pyread_eagle_object_verbose: bool = False
    @staticmethod
    def set_pyread_eagle_to_verbose(verbose: bool = True) -> None:
        """
        Enable verbose output on pyread_eagle object for new SnapshotEAGLE objects.
        """
        SnapshotEAGLE.__pyread_eagle_object_verbose = verbose
    @staticmethod
    def make_reader_object(filepath: str) -> EagleSnapshot:
        """
        Create an EagleSnapshot instance.
        """
        Console.print_debug("Creating pyread_eagle object:")
        return EagleSnapshot(fname = filepath, verbose = SnapshotEAGLE.__pyread_eagle_object_verbose)

    @property
    def _file_object(self) -> EagleSnapshot:
        return self.__file_object

    # Constructor

    def __init__(self, filepath: str) -> None:
        Console.print_debug(f"Loading EAGLE snapshot from: \"{filepath}\".")

        pattern = re.compile(r'.*sn(?P<snap_type_letter>[ai])pshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]sn(?P=snap_type_letter)p_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
        match = pattern.match(filepath)
        if not match:
            raise ValueError(f"Snapshot filepath \"{filepath}\" does not conform to the naming scheme of an EAGLE snapshot. EAGLE snapshot files must have a clear snapshot number component in both the folder and file names.")
        snap_num = match.group("number")
        is_snipshot = match.group("snap_type_letter") == "i"

        hdf5_reader = HDF5_File(filepath)

        with HDF5_File(filepath, "r") as hdf5_reader:
            redshift = hdf5_reader["Header"].attrs["Redshift"]
            hubble_param = hdf5_reader["Header"].attrs["HubbleParam"]
            expansion_factor = hdf5_reader["Header"].attrs["ExpansionFactor"]
            omega_baryon = hdf5_reader["Header"].attrs["OmegaBaryon"]
            self.__number_of_particles = hdf5_reader["Header"].attrs["NumPart_Total"]
            self.__dm_mass_internal_units = hdf5_reader["Header"].attrs["MassTable"][1]
            self.__box_size_internal_units = hdf5_reader["Header"].attrs["BoxSize"]
            self.__solar_metallicity: float = hdf5_reader["Constants"].attrs["Z_Solar"]
            self.__length_h_exp: float = hdf5_reader["PartType1/Coordinates"].attrs["h-scale-exponent"]
            self.__length_a_exp: float = hdf5_reader["PartType1/Coordinates"].attrs["aexp-scale-exponent"]
            self.__length_cgs_conversion_factor: float = hdf5_reader["PartType1/Coordinates"].attrs["CGSConversionFactor"]
            try:
                self.__mass_h_exp: float = hdf5_reader["PartType0/Mass"].attrs["h-scale-exponent"]
                #self.__mass_a_exp: float = hdf5_reader["PartType0/Mass"].attrs["aexp-scale-exponent"]
                self.__mass_cgs_conversion_factor: float = hdf5_reader["PartType0/Mass"].attrs["CGSConversionFactor"]
            except:
                # Just in case there aren't any gas particles, use the expected values for EAGLE
                self.__mass_h_exp: float = -1.0
                #self.__mass_a_exp: float = 0.0
                self.__mass_cgs_conversion_factor: float = 1.989E43
            self.__velocity_h_exp: float = hdf5_reader["PartType1/Velocity"].attrs["h-scale-exponent"]
            self.__velocity_a_exp: float = hdf5_reader["PartType1/Velocity"].attrs["aexp-scale-exponent"]
            self.__velocity_cgs_conversion_factor: float = hdf5_reader["PartType1/Velocity"].attrs["CGSConversionFactor"]

            self.__cgs_unit_conversion_factor_density: float = hdf5_reader["Units"].attrs["UnitDensity_in_cgs"]
            self.__cgs_unit_conversion_factor_energy: float = hdf5_reader["Units"].attrs["UnitEnergy_in_cgs"]
            self.__cgs_unit_conversion_factor_length: float = hdf5_reader["Units"].attrs["UnitLength_in_cm"]
            self.__cgs_unit_conversion_factor_mass: float = hdf5_reader["Units"].attrs["UnitMass_in_g"]
            self.__cgs_unit_conversion_factor_pressure: float = hdf5_reader["Units"].attrs["UnitPressure_in_cgs"]
            self.__cgs_unit_conversion_factor_time: float = hdf5_reader["Units"].attrs["UnitTime_in_s"]
            self.__cgs_unit_conversion_factor_velocity: float = hdf5_reader["Units"].attrs["UnitVelocity_in_cm_per_s"]

            assert self.__length_cgs_conversion_factor == self.__cgs_unit_conversion_factor_length
            assert self.__mass_cgs_conversion_factor == self.__cgs_unit_conversion_factor_mass
            assert self.__velocity_cgs_conversion_factor == self.__cgs_unit_conversion_factor_velocity

        self.__file_object = SnapshotEAGLE.make_reader_object(filepath)
        Console.print_debug("Calling pyread_eagle object's 'select_region' method:")
        self.__file_object.select_region(0.0, self.__box_size_internal_units, 0.0, self.__box_size_internal_units, 0.0, self.__box_size_internal_units)

        self.__number_of_particles_this_rank: np.ndarray[tuple[int], np.dtype[np.int64]]
        if MPI_Config.comm_size == 1:
            self.__number_of_particles_this_rank = self.__number_of_particles
        else:
            # If MPI is in use, select only a portion of the particles
            Console.print_debug("Calling pyread_eagle object's 'split_selection' method.")
            self.__file_object.split_selection(MPI_Config.rank, MPI_Config.comm_size)

            #t = self.__file_object.read_dataset(ParticleType.gas.value, "ParticleIDs")
            self.__number_of_particles_this_rank = np.zeros(shape = self.__number_of_particles.shape, dtype = np.int64)
            for i in range(len(self.__number_of_particles)):
                self.__number_of_particles_this_rank[i] = self.__check_number_of_particles_this_rank(i)

#            # Calculate the MPI reorder opperation to evenly distribute read data between all ranks
#            # This is needed as pyread_eagle only splits data between ranks by chunk, meaning uneven particle numbers and complications involving more ranks than chunks
#            expected_lengths_total = self._get_number_of_particles()
#            expected_lengths_this_rank = self._get_number_of_particles_this_rank()
#            self.__data_read_reorder: dict[ParticleType, ArrayReorder_MPI] = {}
#            for part_type in ParticleType.get_all():
#
#                self.__data_read_reorder[part_type] = mpi_redistribute_array_evenly
#
#        else:
#            self.__data_read_reorder = { part_type : (lambda x: x) for part_type in ParticleType.get_all() }

        Console.print_debug("Initialising base class.")

        super().__init__(
            filepath = filepath,
            number = snap_num,
            redshift = redshift,
            hubble_param = hubble_param,
            omega_baryon = omega_baryon,
            expansion_factor = expansion_factor,
            box_size = unyt_array(np.array([self.__box_size_internal_units, self.__box_size_internal_units, self.__box_size_internal_units], dtype = float) * (hubble_param ** self.__length_h_exp) * self.__length_cgs_conversion_factor, units = "cm").to("Mpc"),
            tracked_elements = (H, He, C, N, O, Ne, Mg, Si, Fe),
            snipshot = is_snipshot
        )

        self.__dark_matter_particle_mass = self._convert_to_cgs_mass(self.__dm_mass_internal_units).to("Msun")

        Console.print_debug("Done creating EAGLE snapshot reader.")

    # Restrictions on spatial regions from which to read data

    #def _restrict_data_proper_loading_region(self, min_x: float|unyt_quantity|None = None, max_x: float|unyt_quantity|None = None, min_y: float|unyt_quantity|None = None, max_y: float|unyt_quantity|None = None, min_z: float|unyt_quantity|None = None, max_z: float|unyt_quantity|None = None, clear_existing_region = True, recalculate_split = True) -> None:
    #    self._restrict_data_comoving_loading_region(
    #        min_x / self.a if min_x is not None else None,
    #        max_x / self.a if max_x is not None else None,
    #        min_y / self.a if min_y is not None else None,
    #        max_y / self.a if max_y is not None else None,
    #        min_z / self.a if min_z is not None else None,
    #        max_z / self.a if max_z is not None else None,
    #        clear_existing_region = clear_existing_region,
    #        recalculate_split = recalculate_split
    #    )

    def _restrict_data_comoving_loading_region(
        self,
        min_x: float|unyt_quantity|None = None, max_x: float|unyt_quantity|None = None,
        min_y: float|unyt_quantity|None = None, max_y: float|unyt_quantity|None = None,
        min_z: float|unyt_quantity|None = None, max_z: float|unyt_quantity|None = None,
        clear_existing_region = True,
        recalculate_split = True
    ) -> None:
        # Get the conversion factor to go from comoving to h-less comoving
        # (a.k.a., multiply inputs by this value!)
        conversion_factor = 1 / ((self.hubble_param ** self.__length_h_exp) * self.__length_cgs_conversion_factor)

        # Convert values so each field is filled with a floating point value
        if min_x is not None:
            if not isinstance(min_x, unyt_quantity):
                min_x = unyt_quantity(min_x, units = "Mpc", dtype = float)
            min_x = min_x.to("cm").value * conversion_factor
        else:
            min_x = 0.0
        if max_x is not None:
            if not isinstance(max_x, unyt_quantity):
                max_x = unyt_quantity(max_x, units = "Mpc", dtype = float)
            max_x = max_x.to("cm").value * conversion_factor
        else:
            max_x = self.__box_size_internal_units
        if min_y is not None:
            if not isinstance(min_y, unyt_quantity):
                min_y = unyt_quantity(min_y, units = "Mpc", dtype = float)
            min_y = min_y.to("cm").value * conversion_factor
        else:
            min_y = 0.0
        if max_y is not None:
            if not isinstance(max_y, unyt_quantity):
                max_y = unyt_quantity(max_y, units = "Mpc", dtype = float)
            max_y = max_y.to("cm").value * conversion_factor
        else:
            max_y = self.__box_size_internal_units
        if min_z is not None:
            if not isinstance(min_z, unyt_quantity):
                min_z = unyt_quantity(min_z, units = "Mpc", dtype = float)
            min_z = min_z.to("cm").value * conversion_factor
        else:
            min_z = 0.0
        if max_z is not None:
            if not isinstance(max_z, unyt_quantity):
                max_z = unyt_quantity(max_z, units = "Mpc", dtype = float)
            max_z = max_z.to("cm").value * conversion_factor
        else:
            max_z = self.__box_size_internal_units

        # Check for regions where the min and max values are the wrong way around - i.e. wrap around the box
        # Correct these to allow negative values (to be handled later)
        if min_x > max_x:
            min_x = np.mod(min_x, self.__box_size_internal_units) # Move both endpoints inside the box
            max_x = np.mod(max_x, self.__box_size_internal_units) # Move both endpoints inside the box
            if min_x > max_x:                                     # If the endpoints are still the wrong way around:
                min_y = min_y - self.__box_size_internal_units    #     Move the min point into negative space so that it gets wrapped later
        if min_y > max_y:
            min_y = np.mod(min_y, self.__box_size_internal_units)
            max_y = np.mod(max_y, self.__box_size_internal_units)
            if min_y > max_y:
                min_y = min_y - self.__box_size_internal_units
        if min_z > max_z:
            min_z = np.mod(min_z, self.__box_size_internal_units)
            max_z = np.mod(max_z, self.__box_size_internal_units)
            if min_z > max_z:
                min_z = min_z - self.__box_size_internal_units

        # Check for regions larger than the box
        #     Truncate these to be the size of the box in that dimension
        # Also check for regions where the maximum is outside the box
        #     Shift these to the other side of the box (reduces wrapping conditions that need checking later)
        if self.__box_size_internal_units < max_x:
            if min_x < 0.0 or min_x + self.__box_size_internal_units < max_x:
                min_x = 0.0
                max_x = self.__box_size_internal_units
            else:
                min_x = min_x - self.__box_size_internal_units
                max_x = max_x - self.__box_size_internal_units
        if self.__box_size_internal_units < max_y:
            if min_y < 0.0 or min_y + self.__box_size_internal_units < max_y:
                min_y = 0.0
                max_y = self.__box_size_internal_units
            else:
                min_y = min_y - self.__box_size_internal_units
                max_y = max_y - self.__box_size_internal_units
        if self.__box_size_internal_units < max_z:
            if min_z < 0.0 or min_z + self.__box_size_internal_units < max_z:
                min_z = 0.0
                max_z = self.__box_size_internal_units
            else:
                min_z = min_z - self.__box_size_internal_units
                max_z = max_z - self.__box_size_internal_units

        # Store the chunk as a tuple in a list
        # This will be used to break up the chunk if it crosses a boundary
        wrapped_region_chunks: list[tuple[float, float, float, float, float, float]] = [
            (min_x, max_x, min_y, max_y, min_z, max_z)
        ]
        copy_region_chunks: list[tuple[float, float, float, float, float, float]]

        # Check for boundaries being crossed by the initial region and mutate the existing region(s) while creating new regions for the offending space
        # Only need to check 0 boundary as above changes should mandate the max value be within the box
        if min_x < 0.0:
            copy_region_chunks = wrapped_region_chunks.copy()
            wrapped_region_chunks = []
            for region in copy_region_chunks:
                wrapped_region_chunks.append((0.0, *region[1:]))
                wrapped_region_chunks.append((self.__box_size_internal_units + region[0], self.__box_size_internal_units, *region[2:]))
        if min_y < 0.0:
            copy_region_chunks = wrapped_region_chunks.copy()
            wrapped_region_chunks = []
            for region in copy_region_chunks:
                wrapped_region_chunks.append((*region[:2], 0.0, *region[3:]))
                wrapped_region_chunks.append((*region[:2], self.__box_size_internal_units + region[2], self.__box_size_internal_units, *region[4:]))
        if min_z < 0.0:
            copy_region_chunks = wrapped_region_chunks.copy()
            wrapped_region_chunks = []
            for region in copy_region_chunks:
                wrapped_region_chunks.append((*region[:4], 0.0, region[5]))
                wrapped_region_chunks.append((*region[:4], self.__box_size_internal_units + region[4], self.__box_size_internal_units))

        if clear_existing_region:
            self.__file_object.clear_selection()
        for region in wrapped_region_chunks:
            if region[0] == region[1] or region[2] == region[3] or region[4] == region[5]:
                continue # If any of the axes are of 0 length, don't bother
                         # Should only occur when a region is specified adjacent to but entirely outside of the box
            self.__file_object.select_region(*region)

        if recalculate_split:
            if MPI_Config.comm_size > 1:
                self.__file_object.split_selection(MPI_Config.rank, MPI_Config.comm_size)
            self.__update_particle_count()
            self._update_number_of_particles()

    def restrict_data_proper_loading_region(
        self,
        min_x: float|unyt_quantity|list[float|unyt_quantity]|None = None, max_x: float|unyt_quantity|list[float|unyt_quantity]|None = None,
        min_y: float|unyt_quantity|list[float|unyt_quantity]|None = None, max_y: float|unyt_quantity|list[float|unyt_quantity]|None = None,
        min_z: float|unyt_quantity|list[float|unyt_quantity]|None = None, max_z: float|unyt_quantity|list[float|unyt_quantity]|None = None
    ) -> None:
        self.restrict_data_comoving_loading_region(
            ([min_x / self.a for i in range(len(min_x))] if isinstance(min_x, Sequence) else min_x / self.a) if min_x is not None else None,
            ([max_x / self.a for i in range(len(max_x))] if isinstance(max_x, Sequence) else max_x / self.a) if max_x is not None else None,
            ([min_y / self.a for i in range(len(min_y))] if isinstance(min_y, Sequence) else min_y / self.a) if min_y is not None else None,
            ([max_y / self.a for i in range(len(max_y))] if isinstance(max_y, Sequence) else max_y / self.a) if max_y is not None else None,
            ([min_z / self.a for i in range(len(min_z))] if isinstance(min_z, Sequence) else min_z / self.a) if min_z is not None else None,
            ([max_z / self.a for i in range(len(max_z))] if isinstance(max_z, Sequence) else max_z / self.a) if max_z is not None else None
        )

    def restrict_data_comoving_loading_region(
        self,
        min_x: float|unyt_quantity|list[float|unyt_quantity]|None = None, max_x: float|unyt_quantity|list[float|unyt_quantity]|None = None,
        min_y: float|unyt_quantity|list[float|unyt_quantity]|None = None, max_y: float|unyt_quantity|list[float|unyt_quantity]|None = None,
        min_z: float|unyt_quantity|list[float|unyt_quantity]|None = None, max_z: float|unyt_quantity|list[float|unyt_quantity]|None = None
    ) -> None:
        min_x_iterable = isinstance(min_x, Sequence) if min_x is not None else False
        max_x_iterable = isinstance(max_x, Sequence) if max_x is not None else False
        min_y_iterable = isinstance(min_y, Sequence) if min_y is not None else False
        max_y_iterable = isinstance(max_y, Sequence) if max_y is not None else False
        min_z_iterable = isinstance(min_z, Sequence) if min_z is not None else False
        max_z_iterable = isinstance(max_z, Sequence) if max_z is not None else False

        if not (min_x_iterable or max_x_iterable or min_y_iterable or max_y_iterable or min_z_iterable or max_z_iterable):
            # Only single values or None so a single call will be sufficient
            self._restrict_data_comoving_loading_region(
                min_x = min_x,
                max_x = max_x,
                min_y = min_y,
                max_y = max_y,
                min_z = min_z,
                max_z = max_z,
                clear_existing_region = True,
                recalculate_split = True
            )

        else:
            n_iterables = sum([min_x_iterable, max_x_iterable, min_y_iterable, max_y_iterable, min_z_iterable, max_z_iterable])
            total_iterable_values = sum([
                (len(min_x) if min_x_iterable else 0),
                (len(min_x) if min_x_iterable else 0),
                (len(min_x) if min_x_iterable else 0),
                (len(min_x) if min_x_iterable else 0),
                (len(min_x) if min_x_iterable else 0),
                (len(min_x) if min_x_iterable else 0)
            ])
            if min_x_iterable and len(min_x) * n_iterables != total_iterable_values:
                raise IndexError("Number of element in sequence min_x inconsistent with other sequence arguments. All arguments given as sequences must have the same length.")
            if max_x_iterable and len(max_x) * n_iterables != total_iterable_values:
                raise IndexError("Number of element in sequence max_x inconsistent with other sequence arguments. All arguments given as sequences must have the same length.")
            if min_y_iterable and len(min_y) * n_iterables != total_iterable_values:
                raise IndexError("Number of element in sequence min_y inconsistent with other sequence arguments. All arguments given as sequences must have the same length.")
            if max_y_iterable and len(max_y) * n_iterables != total_iterable_values:
                raise IndexError("Number of element in sequence max_y inconsistent with other sequence arguments. All arguments given as sequences must have the same length.")
            if min_z_iterable and len(min_z) * n_iterables != total_iterable_values:
                raise IndexError("Number of element in sequence min_z inconsistent with other sequence arguments. All arguments given as sequences must have the same length.")
            if max_z_iterable and len(max_z) * n_iterables != total_iterable_values:
                raise IndexError("Number of element in sequence max_z inconsistent with other sequence arguments. All arguments given as sequences must have the same length.")

            # Guaranteed to be the same number per contributing parameter, so should be an exact integer!
            n_iterable_elements = int(total_iterable_values / n_iterables)

            if n_iterable_elements == 0:
                raise IndexError()

            # Always do the first one with clearing the existing selection
            self._restrict_data_comoving_loading_region(
                min_x = min_x[0] if min_x_iterable else min_x,
                max_x = max_x[0] if min_x_iterable else max_x,
                min_y = min_y[0] if min_y_iterable else min_y,
                max_y = max_y[0] if min_y_iterable else max_y,
                min_z = min_z[0] if min_z_iterable else min_z,
                max_z = max_z[0] if min_z_iterable else max_z,
                clear_existing_region = True,
                recalculate_split = n_iterable_elements == 1 # This evaluating to True isn't optimal, but technically permitted - please don't do this :'(
            )

            if n_iterable_elements > 2:
                for i in range(1, n_iterable_elements - 1):
                    self._restrict_data_comoving_loading_region(
                        min_x = min_x[i] if min_x_iterable else min_x,
                        max_x = max_x[i] if min_x_iterable else max_x,
                        min_y = min_y[i] if min_y_iterable else min_y,
                        max_y = max_y[i] if min_y_iterable else max_y,
                        min_z = min_z[i] if min_z_iterable else min_z,
                        max_z = max_z[i] if min_z_iterable else max_z,
                        clear_existing_region = False,
                        recalculate_split = False
                    )

            # The last one (if more than 1) should recalculate the particle distribution
            if n_iterable_elements >= 2:
                self._restrict_data_comoving_loading_region(
                    min_x = min_x[-1] if min_x_iterable else min_x,
                    max_x = max_x[-1] if min_x_iterable else max_x,
                    min_y = min_y[-1] if min_y_iterable else min_y,
                    max_y = max_y[-1] if min_y_iterable else max_y,
                    min_z = min_z[-1] if min_z_iterable else min_z,
                    max_z = max_z[-1] if min_z_iterable else max_z,
                    clear_existing_region = False,
                    recalculate_split = True
                )

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

    # Particle number accessors for internal use

    def _get_number_of_particles(self) -> dict[ParticleType, int]:
        return { p : int(self.__number_of_particles[p.value]) for p in ParticleType.get_all() }
    def _get_number_of_particles_this_rank(self) -> dict[ParticleType, int]:
        #return { p : (limits:=mpi_get_slice(n_parts:=int(self.__number_of_particles[p.value])).indices(n_parts))[1] - limits[0] for p in ParticleType.get_all() }
        return { p : int(self.__number_of_particles_this_rank[p.value]) for p in ParticleType.get_all() }
    def __check_number_of_particles_this_rank(self, particle_type_index: int) -> int:
        try:
            return self.__file_object.count_particles(particle_type_index)
        except:
            return 0
    def __update_particle_count(self) -> None:
        """
        Used to update the per-rank and total particle numbers.
        """
        n_parttypes = len(self.__number_of_particles_this_rank)
        for i in range(n_parttypes):
            self.__number_of_particles_this_rank[i] = self.__check_number_of_particles_this_rank(i)
        if MPI_Config.comm_size == 1:
            self.__number_of_particles = self.__number_of_particles_this_rank
        else:
            self.__number_of_particles = sum(MPI_Config.comm.allgather(self.__number_of_particles_this_rank), start = np.zeros_like(self.__number_of_particles_this_rank))

    # Data reading methods

    def __read_dataset(self, particle_type: ParticleType, field_name: str, expected_dtype: type[T], expected_shape_after_first_dimension: tuple[int, ...] = tuple()) -> np.ndarray[tuple[int, ...], np.dtype[T]]:
        #return self.__data_read_reorder[particle_type](loaded_data if (loaded_data:=self.__file_object.read_dataset(particle_type.value, field_name)) is not None else np.empty((0,), dtype = expected_dtype))
        Console.print_verbose_info(f"Reading snapshot {particle_type.name} particle dataset \"{field_name}\".")
        Console.print_debug(f"Expecting data of type {expected_dtype}.")
        loaded_data = self.__file_object.read_dataset(particle_type.value, field_name)
        processed_object = loaded_data if loaded_data is not None else np.empty((0, *expected_shape_after_first_dimension), dtype = expected_dtype)
        #Console.print_debug("Redistributing data.")
        #redistributed_data =  self.__data_read_reorder[particle_type](processed_object)
        Console.print_debug("Done reading data.")
        #return redistributed_data
        return processed_object

    def _get_IDs(self, particle_type: ParticleType) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        return typing_cast(np.ndarray[tuple[int], np.dtype[np.int64]], self.__read_dataset(particle_type, "ParticleIDs", expected_dtype = np.int64))

    def _get_smoothing_lengths(self, particle_type: ParticleType, use_proper_units: bool = False) -> unyt_array:
        return self._convert_to_cgs_length(self.__read_dataset(particle_type, "SmoothingLength", expected_dtype = np.float64), proper = use_proper_units).to("Mpc")

    def _get_masses(self, particle_type: ParticleType) -> unyt_array:
        if particle_type == ParticleType.dark_matter:
            #return self._convert_to_cgs_mass(np.full(self.number_of_particles(ParticleType.dark_matter), self.__dm_mass_internal_units)).to("Msun")
            return np.full(self.number_of_particles(ParticleType.dark_matter), self.__dark_matter_particle_mass)
        return self._convert_to_cgs_mass(self.__read_dataset(particle_type, "Mass", expected_dtype = np.float64)).to("Msun")

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
        return self._convert_to_cgs_mass(self.__read_dataset(ParticleType.black_hole, "Mass", expected_dtype = np.float64)).to("Msun")

    def _get_total_black_hole_subgrid_mass(self) -> unyt_array:
        return self.get_black_hole_subgrid_masses().sum()

    def _get_total_black_hole_dynamical_mass(self) -> unyt_array:
        return self.get_black_hole_dynamical_masses().sum()

    def _get_positions(self, particle_type: ParticleType, use_proper_units: bool = False) -> unyt_array:
        return self._convert_to_cgs_length(self.__read_dataset(particle_type, "Coordinates", expected_dtype = np.float64, expected_shape_after_first_dimension = (3,)), proper = use_proper_units).to("Mpc")

    def _get_velocities(self, particle_type: ParticleType, use_proper_units: bool = False) -> unyt_array:
        return self._convert_to_cgs_velocity(self.__read_dataset(particle_type, "Velocity", expected_dtype = np.float64, expected_shape_after_first_dimension = (3,)), proper = use_proper_units).to("km/s")

    def _get_sfr(self) -> unyt_array:
        return unyt_array(self.__read_dataset(ParticleType.gas, "StarFormationRate", expected_dtype = np.float64), units = "Msun/yr")

    def _get_metallicities(self, particle_type: ParticleType, solar_units: bool, solar_metallicity: float|None) -> unyt_array:
        result = unyt_array(self.__read_dataset(particle_type, "Metallicity", expected_dtype = np.float64), units = None)
        if not solar_units:
            return result
        else:
            return result / (solar_metallicity if solar_metallicity is not None else self.__solar_metallicity)
    
    def _get_mean_enrichment_redshift(self, particle_type: ParticleType) -> unyt_array:
        return unyt_array(self.__read_dataset(particle_type, "MetalMassWeightedRedshift", expected_dtype = np.float64), units = None)

    def _get_densities(self, particle_type: ParticleType, use_proper_units: bool) -> unyt_array:
        return self.make_cgs_data(
            "g/cm**3",
            self.__read_dataset(particle_type, "Density", expected_dtype = np.float64),
            h_exp = 2.0,
            cgs_conversion_factor = self.__cgs_unit_conversion_factor_density,
            a_exp = -3 if use_proper_units else 0
        ).to("Msun/Mpc**3")
    
    def _get_number_densities(self, particle_type: ParticleType, element: Element, use_proper_units: bool, default_abundance: float|None = None) -> unyt_array:
        if self.snipshot and default_abundance is None:
            raise SnipshotOperationError(
                operation_name = "get_number_densities",
                message = "Unable to read abundance data - snipshots do not contain this information. A \"default_abundance\" value must be specified."
            )

        atomic_weight: unyt_quantity
        match element.atomic_number:
            case H.atomic_number:  atomic_weight = atomic_weights.H  * ATOMIC_MASS_UNIT # Hydrogen
            case He.atomic_number: atomic_weight = atomic_weights.He * ATOMIC_MASS_UNIT # Helium
            case C.atomic_number:  atomic_weight = atomic_weights.C  * ATOMIC_MASS_UNIT # Carbon
            case N.atomic_number:  atomic_weight = atomic_weights.N  * ATOMIC_MASS_UNIT # Nitrogen
            case O.atomic_number:  atomic_weight = atomic_weights.O  * ATOMIC_MASS_UNIT # Oxygen
            case Ne.atomic_number: atomic_weight = atomic_weights.Ne * ATOMIC_MASS_UNIT # Neon
            case Mg.atomic_number: atomic_weight = atomic_weights.Mg * ATOMIC_MASS_UNIT # Magnesium
            case Si.atomic_number: atomic_weight = atomic_weights.Si * ATOMIC_MASS_UNIT # Silicon
            case Fe.atomic_number: atomic_weight = atomic_weights.Fe * ATOMIC_MASS_UNIT # Iron
            case _:
                raise ValueError(f"Element \"{element}\" not tracked as part of EAGLE.")

        particle_densities = self.make_cgs_data(
            "g/cm**3",
            self.__read_dataset(particle_type, "Density", expected_dtype = np.float64),
            h_exp = 2.0,
            cgs_conversion_factor = self.__cgs_unit_conversion_factor_density,
            a_exp = -3 if use_proper_units else 0
        )

        if self.snipshot:
            return particle_densities * default_abundance / atomic_weight
        else:
            return particle_densities * self.__read_dataset(particle_type, f"ElementAbundance/{element}", expected_dtype = np.float64) / atomic_weight

    def _get_temperatures(self, particle_type: ParticleType) -> unyt_array:
        return unyt_array(self.__read_dataset(particle_type, "Temperature", expected_dtype = np.float64), units = "K")

    def _get_elemental_abundance(self, particle_type: ParticleType, element: Element) -> unyt_array:
        if self.snipshot:
            raise SnipshotFieldError(
                field_name = f"{particle_type}ElementAbundance",
                message = "Unable to read abundance data - snipshots do not contain this information."
            )

        element_field_name: str
        match element.atomic_number:
            case H.atomic_number:  element_field_name = "Hydrogen"
            case He.atomic_number: element_field_name = "Helium"
            case C.atomic_number:  element_field_name = "Carbon"
            case N.atomic_number:  element_field_name = "Nitrogen"
            case O.atomic_number:  element_field_name = "Oxygen"
            case Ne.atomic_number: element_field_name = "Neon"
            case Mg.atomic_number: element_field_name = "Magnesium"
            case Si.atomic_number: element_field_name = "Silicon"
            case Fe.atomic_number: element_field_name = "Iron"
            case _:
                raise ValueError(f"Element \"{element}\" not tracked as part of EAGLE.")

        return self.__read_dataset(particle_type, f"ElementAbundance/{element_field_name}", expected_dtype = np.float64)



    def get_group_ID(self, particle_type: ParticleType, include_unbound: bool = True) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """
        EXPEREMENTAL
        """

        group_numbers = typing_cast(np.ndarray[tuple[int], np.dtype[np.int32]], self.__read_dataset(particle_type, "GroupNumber", expected_dtype = np.int32))

        unbound_mask = group_numbers < 0#TODO: is this the right meaning for a -ve group number, or is it like in the SUBFIND reader?????
        if include_unbound:
            group_numbers[unbound_mask] = -group_numbers[unbound_mask]
        else:
            group_numbers[unbound_mask] = self.EAGLE_MAX_GROUP_NUMBER

        return group_numbers

    def get_group_index(self, particle_type: ParticleType, include_unbound: bool = True) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """
        EXPEREMENTAL
        """
        group_numbers = self.get_group_ID(particle_type = particle_type, include_unbound = include_unbound)

        max_number = group_numbers.max() # This is used to indicate no associated halo
        group_numbers[group_numbers == self.EAGLE_MAX_GROUP_NUMBER] = 0

        group_indexes = group_numbers - 1

        return group_indexes







'''
    @staticmethod
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
        raise NotImplementedError("Not implemented for EAGLE. Update to generalise file path creation.")#TODO:

    @staticmethod
    def scrape_filepaths(#TODO: create file info objects that do this with a common interface for generating these objects to allow each subclass to keep the filepath formatting clear and obscured from the user
        catalogue_directory: str,
        snipshots: bool = False
    ) -> tuple[
            tuple[
                str,
                tuple[str, ...],
                tuple[int, ...],
                str
            ],
            ...
         ]:

#        pattern = re.compile(r'.*snapshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]snap_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
        if not snipshots:
            pattern = re.compile(r'.*snapshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]snap_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
        else:
            pattern = re.compile(r'.*snipshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]snip_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')

        snapshots: dict[str, list[str, list[str], list[int], str]] = {}

        for root, _, files in os.walk(catalogue_directory):
            for filename in files:
                match = pattern.match(os.path.join(root, filename))
                if match:
                    number = match.group("number")
                    redshift_int = match.group("redshift_int")
                    redshift_dec = match.group("redshift_dec")
                    parallel_index = int(match.group("parallel_index"))
                    extension = match.group("extension")

                    tag = f"{number}_z{redshift_int}p{redshift_dec}"
                    basename = os.path.join(f"snapshot_{tag}", f"snap_{tag}") if not snipshots else os.path.join(f"snipshot_{tag}", f"snip_{tag}")

                    if tag not in snapshots:
                        snapshots[tag] = [basename, [number], [parallel_index], extension]
                    else:
                        assert basename == snapshots[tag][0]
                        assert extension == snapshots[tag][3]
                        snapshots[tag][2].append(parallel_index)

        for tag in snapshots:
            snapshots[tag][2].sort()

        return tuple([
            tuple([
                snapshots[tag][0],
                tuple(snapshots[tag][1]),
                tuple(snapshots[tag][2]),
                snapshots[tag][3]
            ])
            for tag
            in snapshots
        ])

    @staticmethod
    def generate_filepaths_from_partial_info(
        directory: str,
        basename: str|None = None,
        snapshot_number_strings: list[str]|None = None,
        file_extension: str|None = None,
        parallel_ranks: list[int]|None = None,
        snipshots: bool = False
    ) -> dict[
            str,
            str|dict[int, str]
         ]:
        if basename is not None or file_extension is not None or parallel_ranks is not None:
            raise NotImplementedError("TODO: some fields not supported for EAGLE. Change API to use objects with file info specific to sim types.")#TODO:

        snap_file_info = { snap[1][0] : snap for snap in SnapshotEAGLE.scrape_filepaths(directory, snipshots = snipshots) }
        selected_files = {}
        for num in (snapshot_number_strings if snapshot_number_strings is not None else snap_file_info.keys()):
            if num not in snap_file_info:
                raise FileNotFoundError("Snapshot numbers provided not all present in directory.")
#            selected_files[num] = { i : os.path.join("/mnt/aridata1/users/aricrowe/replacement_EAGLE_snap/RefL0100N1504" if num == "006" else directory, f"{snap_file_info[num][0]}.{i}.{snap_file_info[num][3]}") for i in snap_file_info[num][2] }
            selected_files[num] = { i : os.path.join(directory, f"{snap_file_info[num][0]}.{i}.{snap_file_info[num][3]}") for i in snap_file_info[num][2] }

        return selected_files

    @staticmethod
    def get_snapshot_order(snapshot_file_info: list[str], reverse = False) -> list[str]:
        snapshot_file_info = list(snapshot_file_info)
        snapshot_file_info.sort(key = int, reverse = reverse)
        return snapshot_file_info



#class SnipshotEAGLE(SnapshotEAGLE):
#    def get_smoothing_lengths(self, particle_type: ParticleType) -> unyt_array:
#        raise NotImplementedError("Smoothing length not avalible from EAGLE snipshots.")
#    def _get_sfr(self, particle_type: ParticleType) -> unyt_array:
#        raise NotImplementedError("SFR not avalible from EAGLE snipshots.")
#
#    @staticmethod#TODO:
#    def scrape_filepaths(
#        catalogue_directory: str
#    ) -> tuple[
#            tuple[
#                str,
#                tuple[str, ...],
#                Union[tuple[int, ...], None],
#                str
#            ],
#            ...
#         ]:
#
#        pattern = re.compile(r'^snipshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)/snip_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
'''
