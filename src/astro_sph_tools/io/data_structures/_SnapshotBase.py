# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from abc import abstractmethod
import asyncio
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, TypeVar

from mendeleev.models import Element
import numpy as np
from scipy.constants import gravitational_constant
from unyt import unyt_array, unyt_quantity

#from ..errors import SnipshotError, SnipshotOperationError, SnipshotFieldError
from ...data_structures._ParticleType import ParticleType
from ...data_structures._SimulationData import SimulationDataBase, T_ISimulation



T = TypeVar("T", float, np.float64, unyt_quantity, np.ndarray[tuple[()], np.dtype[np.float64]], np.ndarray[tuple[int, ...], np.dtype[np.float64]], unyt_array)

class SnapshotBase(SimulationDataBase[T_ISimulation]):
    """
    Base class type for snapshot data reader types.

    Conventions for implementations:
        By convention, all user-facing methods should return data in co-moving coordinates with
        options to allow conversion to proper units where appropriate. Data should also have no
        factors of h removed (i.e. data should NOT be h-less) and should have physical units
        provided using the `unyt` library where appropriate (even if the unit is `dimensionless`).

        Particle IDs will be integers and not strings. Should the data use IDs that are not already
        integer values, the implementation MUST convert the underlying data and do so in a way that
        preserves the uniqueness between snapshots!

        Elements should be specified using the mendeleev library. To specify an element, use:
            `from mendeleev import H, He` etc.

    Constructor Parameters:
        `str` filepath:
            The filepath to the snapshot file (or the first of a set of sequential files).
        `str` number:
            String representation of the file number (including left-padded zeros).
        `float` redshift:
            The redshift of the snapshot.
        `float` hubble_param:
            The hubble param used for the simulation.
        `float` omega_baryon:
            The value of Ω_b used for the simulation.
        `float` expansion_factor:
            The expansion factor of the snapshot.
            This is almost certainly `1 / (1 + z)`.
        `unyt.unyt_array[Mpc, (3,), numpy.float64]` box_size:
            The lengths of each edge of the simulation volume in Mpc.
            Should be provided as using co-moving coordinates (i.e. the size at redshift 0).
        `bool` snipshot:
            Is this a snipshot (i.e. a snapshot with fewer fields).
    """

    # Constructor

    def __init__(
        self,
        filepath: str,
        number: str,
        redshift: float,
        hubble_param: float,
        omega_baryon: float,
        expansion_factor: float,
        box_size: unyt_array,
        tracked_elements: tuple[Element, ...],
        snipshot: bool
    ) -> None:
        super().__init__()

        self.__filepath: str = filepath
        self.__file_name: str = os.path.split(self.__filepath)[1]
        self.__snap_num: str = number
        self.__redshift: float = redshift
        self.__hubble_param: float = hubble_param
        self.__omega_baryon: float = omega_baryon
        self.__expansion_factor: float = expansion_factor
        self.__box_size: unyt_array = box_size
        self.__tracked_elements: tuple[Element, ...] = tracked_elements
        self.__is_snipshot: bool = snipshot

        self.__n_parts: dict[ParticleType, int]
        self.__n_parts_this_rank: dict[ParticleType, int]
        self._update_number_of_particles()

    # Abstract method calls required by the constructor

    @abstractmethod
    def _get_number_of_particles(self) -> dict[ParticleType, int]:
        """
        Called by constructor.
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def _get_number_of_particles_this_rank(self) -> dict[ParticleType, int]:
        """
        Called by constructor.
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    def _update_number_of_particles(self) -> None:
        """
        Internal method for updating the number of particles should it be changed by a child class.

        Called by constructor.
        """
        self.__n_parts = self._get_number_of_particles()
        self.__n_parts_this_rank = self._get_number_of_particles_this_rank()

    # File properties

    @property
    def filepath(self) -> str:
        """
        `str` The filepath to the snapshot file (or the first of a set of sequential files).
        """
        return self.__filepath

    @property
    def file_name(self) -> str:
        """
        `str` The filename (including extension) of the snapshot file (or the first of a set of
        sequential files).
        """
        return self.__file_name

    @property
    def number(self) -> str:
        """
        `str` String representation of the file number (including left-padded zeros).
        """
        return self.__snap_num

    @property
    def snipshot(self) -> bool:
        """
        `str` Is this a snipshot (i.e. a snapshot with fewer fields).
        """
        return self.__is_snipshot

    # Snapshot properties

    @property
    def redshift(self) -> float:
        """
        `float` The redshift of the snapshot.
        """
        return self.__redshift
    @property
    def z(self) -> float:
        """
        `float` The redshift of the snapshot.

        Alias for `redshift`.
        """
        return self.redshift

    @property
    def expansion_factor(self) -> float:
        """
        `float` The expansion factor of the snapshot.
        """
        return self.__expansion_factor
    @property
    def a(self) -> float:
        """
        `float` The expansion factor of the snapshot.

        Alias for `expansion_factor`.
        """
        return self.expansion_factor

    # Simulation properties

    @property
    def box_size(self) -> unyt_array:
        """
        `unyt.unyt_array[Mpc, (3,), numpy.float64]` The lengths of each edge of the simulation
        volume in Mpc using co-moving coordinates (i.e. the size at redshift 0).
        """
        return self.__box_size

    @property
    def hubble_param(self) -> float:
        """
        `float` The hubble param used for the simulation.
        """
        return self.__hubble_param
    @property
    def h(self) -> float:
        """
        `float` The hubble param used for the simulation.

        Alias for `hubble_param`.
        """
        return self.hubble_param

    @property
    def tracked_elements(self) -> tuple[Element]:
        """
        `tuple[mendeleev.models.Element]` Elements with abundances tracked by the simulation.
        """
        return self.__tracked_elements

    # Conversions between co-moving and proper coordinates

    def to_proper(self, data: T, length_dimensions_exponent: int) -> T:
        """
        Convert data in co-moving coordinates to proper coordinates.
        To do the reverse, use `to_comoving`.

        Parameters:
            `float-like|array-like` data:
                Data to convert.
            `int` length_dimensions_exponent:
                Exponent. Use this do describe the dimensions of the data.
                For area, use `2`. For density, `-3`. `0` Will have no effect.

        Returns -> `float-like|array-like`:
            `data` converted to proper coordinates using the properties of this snapshot.

        See also:
            `to_proper_length`
            `to_proper_area`
            `to_proper_volume`
            `to_proper_column_density`
            `to_proper_density`
            `to_comoving`
        """
        return data * (self.a**length_dimensions_exponent)

    def to_proper_length(self, data: T) -> T:
        """
        Convert length data in co-moving coordinates to proper coordinates.
        To do the reverse, use `to_comoving_length`.

        Parameters:
            `float-like|array-like` data:
                Length data to convert.

        Returns -> `float-like|array-like`:
            `data` converted to proper coordinates using the properties of this snapshot.

        See also:
            `to_proper`
            `to_proper_area`
            `to_proper_volume`
            `to_proper_column_density`
            `to_proper_density`
        """
        return self.to_proper(data = data, length_dimensions_exponent = 1)

    def to_proper_area(self, data: T) -> T:
        """
        Convert area data in co-moving coordinates to proper coordinates.
        To do the reverse, use `to_comoving_area`.

        Parameters:
            `float-like|array-like` data:
                Area data to convert.

        Returns -> `float-like|array-like`:
            `data` converted to proper coordinates using the properties of this snapshot.

        See also:
            `to_proper`
            `to_proper_length`
            `to_proper_volume`
            `to_proper_column_density`
            `to_proper_density`
        """
        return self.to_proper(data = data, length_dimensions_exponent = 2)

    def to_proper_volume(self, data: T) -> T:
        """
        Convert volume data in co-moving coordinates to proper coordinates.
        To do the reverse, use `to_comoving_volume`.

        Parameters:
            `float-like|array-like` data:
                Volume data to convert.

        Returns -> `float-like|array-like`:
            `data` converted to proper coordinates using the properties of this snapshot.

        See also:
            `to_proper`
            `to_proper_length`
            `to_proper_area`
            `to_proper_column_density`
            `to_proper_density`
        """
        return self.to_proper(data = data, length_dimensions_exponent = 3)

    def to_proper_column_density(self, data: T) -> T:
        """
        Convert column density data in co-moving coordinates to proper coordinates.
        To do the reverse, use `to_comoving_column_density`.

        Parameters:
            `float-like|array-like` data:
                Column density data to convert.

        Returns -> `float-like|array-like`:
            `data` converted to proper coordinates using the properties of this snapshot.

        See also:
            `to_proper`
            `to_proper_length`
            `to_proper_area`
            `to_proper_volume`
            `to_proper_density`
        """
        return self.to_proper(data = data, length_dimensions_exponent = -2)

    def to_proper_density(self, data: T) -> T:
        """
        Convert density data in co-moving coordinates to proper coordinates.
        To do the reverse, use `to_comoving_density`.

        Parameters:
            `float-like|array-like` data:
                Density data to convert.

        Returns -> `float-like|array-like`:
            `data` converted to proper coordinates using the properties of this snapshot.

        See also:
            `to_proper`
            `to_proper_length`
            `to_proper_area`
            `to_proper_volume`
            `to_proper_column_density`
        """
        return self.to_proper(data = data, length_dimensions_exponent = -3)

    def to_comoving(self, data: T, length_dimensions_exponent: int) -> T:
        """
        Convert data in proper coordinates to co-moving coordinates.
        To do the reverse, use `to_proper`.

        Parameters:
            `float-like|array-like` data:
                Data to convert.
            `int` length_dimensions_exponent:
                Exponent. Use this do describe the dimensions of the data.
                For area, use `2`. For density, `-3`. `0` Will have no effect.

        Returns -> `float-like|array-like`:
            `data` converted to co-moving coordinates assuming the data is described by the
            properties of this snapshot.

        See also:
            `to_comoving_length`
            `to_comoving_area`
            `to_comoving_volume`
            `to_comoving_column_density`
            `to_comoving_density`
            `to_proper`
        """
        return data / (self.a**length_dimensions_exponent)

    def to_comoving_length(self, data: T) -> T:
        """
        Convert length data in proper coordinates to co-moving coordinates.
        To do the reverse, use `to_proper_length`.

        Parameters:
            `float-like|array-like` data:
                Length data to convert.

        Returns -> `float-like|array-like`:
            `data` converted to co-moving coordinates assuming the data is described by the
            properties of this snapshot.

        See also:
            `to_comoving`
            `to_comoving_area`
            `to_comoving_volume`
            `to_comoving_column_density`
            `to_comoving_density`
        """
        return self.to_comoving(data = data, length_dimensions_exponent = 1)

    def to_comoving_area(self, data: T) -> T:
        """
        Convert area data in proper coordinates to co-moving coordinates.
        To do the reverse, use `to_proper_area`.

        Parameters:
            `float-like|array-like` data:
                Area data to convert.

        Returns -> `float-like|array-like`:
            `data` converted to co-moving coordinates assuming the data is described by the
            properties of this snapshot.

        See also:
            `to_comoving`
            `to_comoving_length`
            `to_comoving_volume`
            `to_comoving_column_density`
            `to_comoving_density`
        """
        return self.to_comoving(data = data, length_dimensions_exponent = 2)

    def to_comoving_volume(self, data: T) -> T:
        """
        Convert volume data in proper coordinates to co-moving coordinates.
        To do the reverse, use `to_proper_volume`.

        Parameters:
            `float-like|array-like` data:
                Volume data to convert.

        Returns -> `float-like|array-like`:
            `data` converted to co-moving coordinates assuming the data is described by the
            properties of this snapshot.

        See also:
            `to_comoving`
            `to_comoving_length`
            `to_comoving_area`
            `to_comoving_column_density`
            `to_comoving_density`
        """
        return self.to_comoving(data = data, length_dimensions_exponent = 3)

    def to_comoving_column_density(self, data: T) -> T:
        """
        Convert column density data in proper coordinates to co-moving coordinates.
        To do the reverse, use `to_proper_column_density`.

        Parameters:
            `float-like|array-like` data:
                Column density data to convert.

        Returns -> `float-like|array-like`:
            `data` converted to co-moving coordinates assuming the data is described by the
            properties of this snapshot.

        See also:
            `to_comoving`
            `to_comoving_length`
            `to_comoving_area`
            `to_comoving_volume`
            `to_comoving_density`
        """
        return self.to_comoving(data = data, length_dimensions_exponent = -2)

    def to_comoving_density(self, data: T) -> T:
        """
        Convert density data in proper coordinates to co-moving coordinates.
        To do the reverse, use `to_proper_density`.

        Parameters:
            `float-like|array-like` data:
                Density data to convert.

        Returns -> `float-like|array-like`:
            `data` converted to co-moving coordinates assuming the data is described by the
            properties of this snapshot.

        See also:
            `to_comoving`
            `to_comoving_length`
            `to_comoving_area`
            `to_comoving_volume`
            `to_comoving_column_density`
        """
        return self.to_comoving(data = data, length_dimensions_exponent = -3)

    # Critical density

    def calculate_comoving_critical_density(self) -> unyt_quantity:
        """
        Calculate the critical density in co-moving coordinates.

        Calculates `H^2 * 3 / (8 * π * G)`.

        Returns `unyt.unyt_quantity[Msun/Mpc**3, numpy.float64]`:
            Critical density in co-moving coordinates using the units `Msun Mpc^-3`.
        """
        return (self.h**2 * unyt_quantity(100.0, units = "km/s/Mpc")**2 * 3.0 / 8.0 / np.pi / unyt_quantity(gravitational_constant, units = "N*m**2/kg**2")).to("Msun/Mpc**3")

    def calculate_proper_critical_density(self) -> unyt_quantity:
        """
        Calculate the critical density in proper coordinates.

        Calculates `H^2 * 3 / (8 * π * G) / (a^3)`.

        Returns `unyt.unyt_quantity[Msun/Mpc**3, numpy.float64]`:
            Critical density in proper coordinates using the units `Msun Mpc^-3`.
        """
        #return (1.0 + redshift)**3 * self.calculate_comoving_critical_density()
        return self.to_proper_density(self.calculate_comoving_critical_density())

    @property
    def proper_critical_density(self) -> unyt_quantity:
        """
        `unyt.unyt_quantity[Msun/Mpc**3, numpy.float64]` Critical density in proper coordinates.

        `H^2 * 3 / (8 * π * G) / (a^3)`
        """
        return self.calculate_proper_critical_density()

    def calculate_comoving_critical_gas_density(self) -> unyt_quantity:
        """
        Calculate the critical baryon density in co-moving coordinates.

        Calculates `Ω_b * H^2 * 3 / (8 * π * G)`.

        Returns `unyt.unyt_quantity[Msun/Mpc**3, numpy.float64]`:
            Critical baryon density in co-moving coordinates using the units `Msun Mpc^-3`.
        """
        return self.__omega_baryon * self.calculate_comoving_critical_density()

    def calculate_proper_critical_gas_density(self) -> unyt_quantity:
        """
        Calculate the critical baryon density in proper coordinates.

        Calculates `Ω_b * H^2 * 3 / (8 * π * G) / (a^3)`.

        Returns `unyt.unyt_quantity[Msun/Mpc**3, numpy.float64]`:
            Critical density baryon in proper coordinates using the units `Msun Mpc^-3`.
        """
        return self.__omega_baryon * self.calculate_proper_critical_density()

    @property
    def proper_critical_gas_density(self) -> unyt_quantity:
        """
        `unyt.unyt_quantity[Msun/Mpc**3, numpy.float64]` Critical baryon density in proper coordinates.

        `Ω_b * H^2 * 3 / (8 * π * G) / (a^3)`
        """
        return self.calculate_proper_critical_gas_density()

    # Particle number information

    def number_of_particles(self, particle_type: ParticleType) -> int:
        """
        Get the total number of particles of a given type in the snapshot.

        NOTE: when using MPI, this will be different to the number of particles read by the current
        rank! For that, use `number_of_particles_this_rank`.

        Parameters:
            `ParticleType` particle_type:
                The target particle type.

        Returns `int`:
            The number of particles of the specified type in the entire snapshot.
        """
        return self.__n_parts[particle_type]

    def number_of_particles_this_rank(self, particle_type: ParticleType) -> int:
        """
        Get the number of particles of a given type that will be read by the current MPI rank.

        NOTE: when NOT using MPI, this will be identical to `number_of_particles`.

        Parameters:
            `ParticleType` particle_type:
                The target particle type.

        Returns `int`:
            The number of particles of the specified type assigned to this rank.
        """
        return self.__n_parts_this_rank[particle_type]

    # Methods for data loading
    # Those marked as abstract need to be implemented in a child class

    def get_IDs(self, particle_type: ParticleType) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """
        Read the field providing unique particle IDs.

        Parameters:
            `ParticleType` particle_type:
                The target particle type.

        Returns `numpy.ndarray[(N,), numpy.int64]`:
            The array of particle IDs using 64-bit integers.
        """
        return self._get_IDs(particle_type = particle_type)
    @abstractmethod
    def _get_IDs(self, particle_type: ParticleType) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_smoothing_lengths(self, particle_type: ParticleType, use_proper_units: bool = False) -> unyt_array:
        """
        Read the field providing particle smoothing lengths in Mpc.

        Parameters:
            `ParticleType` particle_type:
                The target particle type.
            (optional) `bool` use_proper_units:
                Convert the data to proper coordinates.
                Default is `False`.

        Returns `unyt.unyt_array[Mpc, (N,), numpy.float64]`:
            Using 64-bit floats.
        """
        return self._get_smoothing_lengths(particle_type = particle_type, use_proper_units = use_proper_units)
    @abstractmethod
    def _get_smoothing_lengths(self, particle_type: ParticleType, use_proper_units: bool) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_masses(self, particle_type: ParticleType) -> unyt_array:
        """
        Read the field providing particle masses in Msun.

        Unsupported Particle Types:
            `ParticleType.black_hole` (see alternative dynamical & subgrid methods)

        Parameters:
            `ParticleType` particle_type:
                The target particle type.

        Returns `unyt.unyt_array[Msun, (N,), numpy.float64]`:
            Using 64-bit floats.
        """
        if particle_type == ParticleType.black_hole:
            raise ValueError("get_masses is not supported for black hole particle as they lack a simple mass field.")
        return self._get_masses(particle_type = particle_type)
    @abstractmethod
    def _get_masses(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_total_mass(self, particle_type: ParticleType|None) -> unyt_quantity:
        """
        Read the field(s) providing particle masses in Msun and sum the result(s).

        Unsupported Particle Types:
            `ParticleType.black_hole` (see alternative dynamical & subgrid methods)

        Parameters:
            `ParticleType|None` particle_type:
                The target particle type or `None` to sum all particle fields (except black holes).

        Returns `unyt.unyt_quantity[Msun, numpy.float64]`:
            Using 64-bit float.
        """
        if particle_type == ParticleType.black_hole:
            raise ValueError("get_total_mass is not supported for black hole particle as they lack a simple mass field.")
        return self._get_total_mass(particle_type = particle_type)
    @abstractmethod
    def _get_total_mass(self, particle_type: ParticleType|None) -> unyt_quantity:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_black_hole_subgrid_masses(self) -> unyt_array:
        """
        Read the field providing black hole particle subgrid masses in Msun.

        Returns `unyt.unyt_array[Msun, (N,), numpy.float64]`:
            Using 64-bit floats.
        """
        return self._get_black_hole_subgrid_masses()
    @abstractmethod
    def _get_black_hole_subgrid_masses(self) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_black_hole_dynamical_masses(self) -> unyt_array:
        """
        Read the field providing black hole particle dynamical masses in Msun.

        Returns `unyt.unyt_array[Msun, (N,), numpy.float64]`:
            Using 64-bit floats.
        """
        return self._get_black_hole_dynamical_masses()
    @abstractmethod
    def _get_black_hole_dynamical_masses(self) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_total_black_hole_subgrid_mass(self) -> unyt_quantity:
        """
        Read the field providing black hole particle subgrid masses in Msun and sum the result.

        Returns `unyt.unyt_quantity[Msun, numpy.float64]`:
            Using 64-bit float.
        """
        return self._get_total_black_hole_subgrid_mass()
    @abstractmethod
    def _get_total_black_hole_subgrid_mass(self) -> unyt_quantity:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_total_black_hole_dynamical_mass(self) -> unyt_quantity:
        """
        Read the field providing black hole particle dynamical masses in Msun and sum the result.

        Returns `unyt.unyt_quantity[Msun, numpy.float64]`:
            Using 64-bit float.
        """
        return self._get_total_black_hole_dynamical_mass()
    @abstractmethod
    def _get_total_black_hole_dynamical_mass(self) -> unyt_quantity:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_positions(self, particle_type: ParticleType, use_proper_units: bool = False) -> unyt_array:
        """
        Read the field providing particle coordinates in Mpc.

        Parameters:
            `ParticleType` particle_type:
                The target particle type.
            (optional) `bool` use_proper_units:
                Convert the data to proper coordinates.
                Default is `False`.

        Returns `unyt.unyt_array[Mpc, (N,3), numpy.float64]`:
            Using 64-bit floats.
        """
        return self._get_positions(particle_type = particle_type, use_proper_units = use_proper_units)
    @abstractmethod
    def _get_positions(self, particle_type: ParticleType, use_proper_units: bool) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_velocities(self, particle_type: ParticleType, use_proper_units: bool = False) -> unyt_array:
        """
        Read the field providing particle velocities in km/s.

        Parameters:
            `ParticleType` particle_type:
                The target particle type.
            (optional) `bool` use_proper_units:
                Convert the data to proper coordinates.
                Default is `False`.

        Returns `unyt.unyt_array[km/s, (N,3), numpy.float64]`:
            Using 64-bit floats.
        """
        return self._get_velocities(particle_type = particle_type, use_proper_units = use_proper_units)
    @abstractmethod
    def _get_velocities(self, particle_type: ParticleType, use_proper_units: bool) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_sfr(self) -> unyt_array:
        """
        Read the gas particle field providing star formation rates in Msun per year.

        Returns `unyt.unyt_array[Msun/yr, (N,), numpy.float64]`:
            Using 64-bit floats.
        """
        return self._get_sfr()
    @abstractmethod
    def _get_sfr(self) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_volumes(self, particle_type: ParticleType, use_proper_units: bool = False) -> unyt_array:
        """
        Read the field providing (or calculate) particle smoothing lengths in Mpc.
        Where no explicit volume field exists, this will be calculated as the volume within the
        smoothing length.

        Parameters:
            `ParticleType` particle_type:
                The target particle type.
            (optional) `bool` use_proper_units:
                Convert the data to proper coordinates.
                Default is `False`.

        Returns `unyt.unyt_array[Msun, (N,), numpy.float64]`:
            Using 64-bit floats.
        """
        return self._get_volumes(particle_type = particle_type, use_proper_units = use_proper_units)
    #TODO: move this to be implementation specific and add abstractmethod
    def _get_volumes(self, particle_type: ParticleType, use_proper_units: bool) -> unyt_array:
        return self.get_smoothing_lengths(particle_type, use_proper_units = use_proper_units)**3 * (np.pi * (4/3))

    def get_metallicities(self, particle_type: ParticleType, solar_units: bool = False, solar_metallicity: float|None = None) -> unyt_array:
        """
        Read the field providing particle metallicities.
        Optionally, convert this to solar metallicity. NOTE: this does not use a unyt
        implementation of solar metallicity! The units of the returned array are always
        `dimensionless` By default, the simulation defined version of solar metallicity is used,
        however this can be overridden.

        Supported Particle Types:
            `ParticleType.gas`
            `ParticleType.star`

        Parameters:
            `ParticleType` particle_type:
                The target particle type.
            (optional) `bool` solar_units:
                Convert the values to units of solar metallicity.
                Default is `False`.
            (optional) `float|None` solar_metallicity:
                Override the value of solar metallicity used by the simulation.
                Default is `None`.

        Returns `unyt.unyt_array[dimensionless, (N,), numpy.float64]`:
            Using 64-bit floats.
        """
        if particle_type != ParticleType.gas and particle_type != ParticleType.star:
            raise ValueError("get_metalicities is not supported for particle types other than gas and star.")
        return self._get_metallicities(particle_type = particle_type, solar_units = solar_units, solar_metallicity = solar_metallicity)
    @abstractmethod
    def _get_metallicities(self, particle_type: ParticleType, solar_units: bool, solar_metallicity: float|None) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_mean_enrichment_redshift(self, particle_type: ParticleType) -> unyt_array:
        """
        Read the field providing particle mean enrichment redshift (z_Z).

        Supported Particle Types:
            `ParticleType.gas`
            `ParticleType.star`

        Parameters:
            `ParticleType` particle_type:
                The target particle type.

        Returns `unyt.unyt_array[dimensionless, (N,), numpy.float64]`:
            Using 64-bit floats.
        """
        if particle_type != ParticleType.gas and particle_type != ParticleType.star:
            raise ValueError("get_mean_enrichment_redshift is not supported for particle types other than gas and star.")
        return self._get_mean_enrichment_redshift(particle_type)
    @abstractmethod
    def _get_mean_enrichment_redshift(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_densities(self, particle_type: ParticleType, use_proper_units: bool = False) -> unyt_array:
        """
        Read the field providing particle mass densities.

        Supported Particle Types:
            `ParticleType.gas`
            `ParticleType.star`

        Parameters:
            `ParticleType` particle_type:
                The target particle type.
            (optional) `bool` use_proper_units:
                Convert the data to proper coordinates.
                Default is `False`.

        Returns `unyt.unyt_array[Mpc/Mpc**3, (N,), numpy.float64]`:
            Using 64-bit floats.
        """
        if particle_type != ParticleType.gas and particle_type != ParticleType.star:
            raise ValueError("get_densities is not supported for particle types other than gas and star.")
        return self._get_densities(particle_type, use_proper_units = use_proper_units)
    @abstractmethod
    def _get_densities(self, particle_type: ParticleType, use_proper_units: bool) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_number_densities(self, particle_type: ParticleType, element: Element, use_proper_units: bool = False, default_abundance: float|None = None) -> unyt_array:
        """
        Read the field providing (or calculate) particle elemental number densities.

        Supported Particle Types:
            `ParticleType.gas`
            `ParticleType.star`

        Parameters:
            `ParticleType` particle_type:
                The target particle type.
            `mendeleev.models.Element` element:
                Desired element specified by the mendeleev package (can be imported by the symbol).
            (optional) `bool` use_proper_units:
                Convert the data to proper coordinates.
                Default is `False`.
            (optional) `float|None` default_abundance:
                Provide an elemental abundance to assume in the event that the snapshot doesn't
                contain elemental abundance data.
                Default is `None`.

        Returns `unyt.unyt_array[Mpc**-3, (N,), numpy.float64]`:
            Using 64-bit floats.
        """
        if particle_type != ParticleType.gas and particle_type != ParticleType.star:
            raise ValueError("get_number_densities is not supported for particle types other than gas and star.")
        return self._get_number_densities(particle_type = particle_type, element = element, use_proper_units = use_proper_units, default_abundance = default_abundance)
    @abstractmethod
    def _get_number_densities(self, particle_type: ParticleType, element: Element, use_proper_units: bool, default_abundance: float|None = None) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_temperatures(self, particle_type: ParticleType) -> unyt_array:
        """
        Read the field providing particle temperatures.

        Supported Particle Types:
            `ParticleType.gas`
            `ParticleType.star`

        Parameters:
            `ParticleType` particle_type:
                The target particle type.

        Returns `unyt.unyt_array[K, (N,), numpy.float64]`:
            Using 64-bit floats.
        """
        if particle_type != ParticleType.gas and particle_type != ParticleType.star:
            raise ValueError("get_temperatures is not supported for particle types other than gas and star.")
        return self._get_temperatures(particle_type = particle_type)
    @abstractmethod
    def _get_temperatures(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_elemental_abundance(self, particle_type: ParticleType, element: Element) -> unyt_array:
        """
        Read the field providing particle elemental abundances.

        Supported Particle Types:
            `ParticleType.gas`
            `ParticleType.star`

        Parameters:
            `ParticleType` particle_type:
                The target particle type.
            `mendeleev.models.Element` element:
                Desired element specified by the mendeleev package (can be imported by the symbol).

        Returns `unyt.unyt_array[dimensionless, (N,), numpy.float64]`:
            Using 64-bit floats.
        """
        if particle_type != ParticleType.gas and particle_type != ParticleType.star:
            raise ValueError("get_elemental_abundance is not supported for particle types other than gas and star.")
        return self._get_elemental_abundance(particle_type = particle_type, element = element)
    @abstractmethod
    def _get_elemental_abundance(self, particle_type: ParticleType, element: Element) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    # async versions (experimental)

    async def get_IDs_async(self, particle_type: ParticleType) -> Awaitable[np.ndarray[tuple[int], np.dtype[np.int64]]]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_IDs, particle_type)
#        return self.get_IDs(particle_type)

    async def get_smoothing_lengths_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_smoothing_lengths, particle_type)
#        return self.get_smoothing_lengths(particle_type)

    async def get_masses_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_masses, particle_type)
#        return self.get_masses(particle_type)

    async def get_black_hole_subgrid_masses_async(self) -> Awaitable[unyt_array]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_black_hole_subgrid_masses)
#        return self.get_black_hole_subgrid_masses()

    async def get_black_hole_dynamical_masses_async(self) -> Awaitable[unyt_array]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_black_hole_dynamical_masses)
#        return self.get_black_hole_dynamical_masses()

    async def get_positions_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_positions, particle_type)
#        return self.get_positions(particle_type)

    async def get_velocities_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_velocities, particle_type)
#        return self.get_velocities(particle_type)

    async def get_sfr_async(self) -> Awaitable[unyt_array]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_sfr)
#        return self.get_sfr(particle_type)

    async def get_metalicities_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_metallicities, particle_type)
#        return self.get_metalicities(particle_type)






'''
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
'''
