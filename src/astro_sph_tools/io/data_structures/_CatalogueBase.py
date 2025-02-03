# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import cast as typing_cast, Any, Literal, TypeVar

import numpy as np
from unyt import unyt_quantity, unyt_array

from ._SnapshotBase import SnapshotBase
from ..errors import HaloDefinitionNotSupportedError
from ...data_structures._ParticleType import ParticleType
from ...data_structures._SimulationData import SimulationDataBase, T_ISimulation



T = TypeVar("T", bound = "type[CatalogueBase[Any]]")

class IHaloDefinition(ABC):
    """
    Interface for types used to specify the various ways to describe a halo.
    This serves as a base class for extensibility purposes.
    """
    @abstractmethod
    def is_match(self, value: "IHaloDefinition") -> bool:
        """
        Is the test definition of the same type and does it have a matching value (where
        appropriate)?
        """
        raise NotImplementedError("Attempted to call an abstract method.")
    @abstractmethod
    def get_details_for_error(self) -> str|None:
        """
        Information about the type of halo (e.g. radius) that should be printed when an error
        indicating lack of support is raised.
        """
        raise NotImplementedError("Attempted to call an abstract method.")
    def make_error(self, catalogue_type: T, message: str|None = None) -> HaloDefinitionNotSupportedError:
        """
        Create an error to be raised when this halo definition is not supported.
        """
        return HaloDefinitionNotSupportedError(
            definition_type = type(self),
            catalogue_type = catalogue_type,
            definition_detail = self.get_details_for_error(),
            message = message
        )
    def __eq__(self, value):
        return False if not isinstance(value, IHaloDefinition) else self.is_match(value)
    def __ne__(self, value):
        return not (self == value)
class FOFGroup(IHaloDefinition):
    """
    Halo definition: All particles that are linked as part of a 'Friends Of Friends' clustering.
    """
    def get_details_for_error(self) -> str|None:
        return None
    def is_match(self, value: "IHaloDefinition") -> bool:
        return isinstance(value, FOFGroup)
class IApertureHalo(IHaloDefinition):
    """
    Subtype of IHaloDefinition for haloes defined by a sphere.
    """
class FixedRadiusAperture(IApertureHalo):
    """
    Halo definition: All particles that fall within a fixed radius from the centre.
    """
    def __init__(self, radius: unyt_quantity) -> None:
        self.__radius = radius
    @property
    def radius(self) -> unyt_quantity:
        return self.__radius
    def get_details_for_error(self) -> str|None:
        return f"R = {self.radius.value} ({self.radius.units})"
    def is_match(self, value: "IHaloDefinition") -> bool:
        return isinstance(value, FixedRadiusAperture) and self.radius.to("kpc").value == value.radius.to("kpc").value
class SphericalOverdensityAperture(IApertureHalo):
    """
    Subtype of IHaloDefinition for haloes defined by a sphere with a co-moving radius set individually using
    cosmological properties.
    """
    def __init__(self, overdensity_multiple: int) -> None:
        self.__overdensity_multiple = overdensity_multiple
    @property
    def overdensity_limit(self) -> int:
        return self.__overdensity_multiple
class CriticalSphericalOverdensityAperture(SphericalOverdensityAperture):
    """
    Halo definition: All particles that fall within a co-moving radius within which the density is a multiple
    of the critical density of the universe (according to the simulation cosmogony).
    """
    def get_details_for_error(self) -> str|None:
        return f"R = {self.overdensity_limit} * rho_c"
    def is_match(self, value: "IHaloDefinition") -> bool:
        return isinstance(value, CriticalSphericalOverdensityAperture) and self.overdensity_limit == value.overdensity_limit
class MeanSphericalOverdensityAperture(SphericalOverdensityAperture):
    """
    Halo definition: All particles that fall within a co-moving radius within which the density is a multiple
    of the mean density of the universe (according to the simulation cosmogony).
    """
    def get_details_for_error(self) -> str|None:
        return f"R = {self.overdensity_limit} * <rho>"
    def is_match(self, value: "IHaloDefinition") -> bool:
        return isinstance(value, MeanSphericalOverdensityAperture) and self.overdensity_limit == value.overdensity_limit
class TopHatSphericalOverdensityAperture(SphericalOverdensityAperture):
    """
    Halo definition: From EAGLE "Group_R_TopHat200"
        Co-moving radius within which density is 200 times (18 * pi^2 + 82 * (Omega_m(z)-1) - 39 * (Omega_m(z)-1)^2); Physical radius = radius h^-1 a U_L[cm]
    """
    def get_details_for_error(self) -> str|None:
        return f"R = {self.overdensity_limit} * (18 * pi**2 + 82 * (Omega_m(z) - 1) - 39 * (Omega_m(z) - 1)**2)"
    def is_match(self, value: "IHaloDefinition") -> bool:
        return isinstance(value, TopHatSphericalOverdensityAperture) and self.overdensity_limit == value.overdensity_limit

_1kpc = unyt_quantity(1, units = "kpc")
class BasicHaloDefinitions(Enum):
    """
    Enumeration of built-in definitions of a halo.
    """
    FOF_GROUP = FOFGroup()
    SO_200_CRIT  = CriticalSphericalOverdensityAperture( 200)
    SO_500_CRIT  = CriticalSphericalOverdensityAperture( 500)
    SO_2500_CRIT = CriticalSphericalOverdensityAperture(2500)
    SO_200_MEAN  = MeanSphericalOverdensityAperture( 200)
    SO_500_MEAN  = MeanSphericalOverdensityAperture( 500)
    SO_2500_MEAN = MeanSphericalOverdensityAperture(2500)
    SO_200_TOP_HAT = TopHatSphericalOverdensityAperture(200)
    FIXED_5_KPC   = FixedRadiusAperture(_1kpc *   5)
    FIXED_10_KPC  = FixedRadiusAperture(_1kpc *  10)
    FIXED_30_KPC  = FixedRadiusAperture(_1kpc *  30)
    FIXED_50_KPC  = FixedRadiusAperture(_1kpc *  50)
    FIXED_100_KPC = FixedRadiusAperture(_1kpc * 100)



class CatalogueBase(SimulationDataBase[T_ISimulation]):
    """
    Base class type for catalogue data reader types.

    Built-in halo definitions are found under the static attribute `BasicHaloDefinitions` (Enum).

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

        Raw halo IDs should be made accessible, however it is encouraged to make use of halo
        indexes wherever possible to make code that iterates over haloes as portable as possible.

    Constructor Parameters:
        `str` membership_filepath:
            The filepath to the catalogue particle membership file (or the first of a set of
            sequential files).
        `str` properties_filepath:
            The filepath to the catalogue halo properties file (or the first of a set of sequential
            files).
        `SnapshotBase[ISimulation]` snapshot:
            Snapshot object for snapshot data associated with this catalogue.
    """

    BasicHaloDefinitions = BasicHaloDefinitions

    def __init__(
        self,
        membership_filepath: str,
        properties_filepath: str,
        snapshot: SnapshotBase[T_ISimulation],
    ) -> None:
        super().__init__()

        self.__membership_filepath: str = membership_filepath
        self.__properties_filepath: str = properties_filepath
        self.__snapshot: SnapshotBase[T_ISimulation] = snapshot

        self.__n_haloes = self.get_number_of_haloes()

        self.__descendants_need_calculating: bool = False
        self.__direct_children:   np.ndarray[tuple[int], np.dtype[np.int64]]|None = None
        self.__total_descendants: np.ndarray[tuple[int], np.dtype[np.int64]]|None = None

    # Abstract method calls required by the constructor

    @abstractmethod
    def _get_hierarchy_IDs(self) -> tuple[np.ndarray[tuple[int], np.dtype[np.int64]], np.ndarray[tuple[int], np.dtype[np.int64]]]:#TODO: is this what is so slow?????
        """
        Return two signed integer arrays: ID of each halo and parent ID of each halo.
        A halo with no parent has a parent ID of -1.
        These IDs need only be self-consistent and may differ between implementations!

        Returns `tuple[np.ndarray[(n,), numpy.int64], np.ndarray[(n,), numpy.int64]]`:

            [0] -> Halo IDs

            [1] -> Halo parent IDs
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    # File properties

    @property
    def membership_filepath(self) -> str:
        """
        `str` The filepath to the catalogue particle membership file (or the first of a set of
        sequential files).
        """
        return self.__membership_filepath

    @property
    def properties_filepath(self) -> str:
        """
        `str` The filepath to the catalogue halo properties file (or the first of a set of
        sequential files).
        """
        return self.__properties_filepath

    @property
    def snapshot(self) -> SnapshotBase[T_ISimulation]:
        """
        `SnapshotBase[ISimulation]` Snapshot object for snapshot data associated with this
        catalogue.

        Override this method when implementing a child class to alter the return type.
        """
        return self.__snapshot

    # Snapshot properties
    # These are added for convenience when not needing to utilise the snapshot data directly.

    @property
    def redshift(self) -> float:
        """
        `float` The redshift of the catalogue.
        """
        return self.__snapshot.redshift
    @property
    def z(self) -> float:
        """
        `float` The redshift of the catalogue.

        Alias for `redshift`.
        """
        return self.redshift

    @property
    def expansion_factor(self) -> float:
        """
        `float` The expansion factor of the catalogue.
        """
        return self.__snapshot.expansion_factor
    @property
    def a(self) -> float:
        """
        `float` The expansion factor of the catalogue.

        Alias for `expansion_factor`.
        """
        return self.expansion_factor

    # Simulation properties
    # These are added for convenience when not needing to utilise the snapshot data directly.

    @property
    def box_size(self) -> unyt_array:
        """
        `unyt.unyt_array[Mpc, (3,), numpy.float64]` The lengths of each edge of the simulation
        volume in Mpc using co-moving coordinates (i.e. the size at redshift 0).
        """
        return self.__snapshot.box_size

    @property
    def hubble_param(self) -> float:
        """
        `float` The hubble param used for the simulation.
        """
        return self.__snapshot.hubble_param
    @property
    def h(self) -> float:
        """
        `float` The hubble param used for the simulation.

        Alias for `hubble_param`.
        """
        return self.hubble_param

    # General halo properties

    @property
    def number_of_haloes(self) -> int:
        """
        `int` The total number of haloes in the catalogue.

        Can also be retrieved with `len(<instance>)`.
        """
        return self.__n_haloes
    def __len__(self) -> int:
        """
        `int` The total number of haloes in the catalogue.

        Can also be retrieved with `number_of_haloes`.
        """
        return self.number_of_haloes

    @property
    def number_of_children(self) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """
        `np.ndarray[(N,), numpy.int64]` The number of direct descendants (children) of each halo.

        WARNING: This property is lazily calculated and evaluating it for the first time may incur
        a performance penalty.
        """
        if self.__descendants_need_calculating:
            self.__calculate_descendant_info()
        return typing_cast(np.ndarray[tuple[int], np.dtype[np.int64]], self.__direct_children)

    @property
    def number_of_descendants(self) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """
        `np.ndarray[(N,), numpy.int64]` The total number of descendants of each halo.

        On a tree diagram, this is the total number of nodes that link back to this node.

        WARNING: This property is lazily calculated and evaluating it for the first time may incur
        a performance penalty.
        """
        if self.__descendants_need_calculating:
            self.__calculate_descendant_info()
        return typing_cast(np.ndarray[tuple[int], np.dtype[np.int64]], self.__total_descendants)

    # Methods for data loading
    # Those marked as abstract need to be implemented in a child class

    @abstractmethod
    def get_number_of_haloes(self, particle_type: ParticleType|None = None) -> int:
        """
        Return the number of haloes in the catalogue.
        Optionally, specify a particle type to get the number of haloes containing those particles.

        Parameters:
            (optional) `ParticleType|None` particle_type:
                Get results for only haloes that have at least one particle of this type.

        Returns `int`:
            The number of haloes in the catalogue with at least one particle of the specified type.
        """
        raise NotImplementedError("Attempted to call an abstract method.")
    
    @abstractmethod
    def get_halo_IDs(self, particle_type: ParticleType|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """
        Get a unique list of halo IDs.
        Optionally, specify a particle type to get only haloes containing those particles.

        Parameters:
            (optional) `ParticleType|None` particle_type:
                Get results for only haloes that have at least one particle of this type.

        Returns `np.ndarray[(n,), numpy.int64]`:
            The IDs of haloes in the catalogue with at least one particle of the specified type.
        """
        raise NotImplementedError("Attempted to call an abstract method.")
    @abstractmethod
    def get_halo_indexes(self, particle_type: ParticleType|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """
        Get a list of halo indexes.
        Optionally, specify a particle type to get only haloes containing those particles.

        Parameters:
            (optional) `ParticleType|None` particle_type:
                Get results for only haloes that have at least one particle of this type.

        Returns `np.ndarray[(n,), numpy.int64]`:
            The indexes of haloes in the catalogue with at least one particle of the specified type.
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_halo_parent_IDs(self, particle_type: ParticleType|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """
        Get a list of halo IDs that indicate the parent of a particular halo.
        Length is identical to that returned by `get_halo_IDs` for the same arguments.
        Optionally, specify a particle type to get only get parents of haloes containing those
        particles (the parents do not necessarily contain particles of the specified type).

        Parameters:
            (optional) `ParticleType|None` particle_type:
                Get results for only haloes that have at least one particle of this type.

        Returns `np.ndarray[(n,), numpy.int64]`:
            The parent IDs of haloes in the catalogue with at least one particle of the specified
            type.
        """
        raise NotImplementedError("Attempted to call an abstract method.")
    @abstractmethod
    def get_halo_parent_indexes(self, particle_type: ParticleType|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """
        Get a list of halo indexes that indicate the parent of a particular halo.
        Length is identical to that returned by `get_halo_indexes` for the same arguments.
        Optionally, specify a particle type to get only get parents of haloes containing those
        particles (the parents do not necessarily contain particles of the specified type).

        Parameters:
            (optional) `ParticleType|None` particle_type:
                Get results for only haloes that have at least one particle of this type.

        Returns `np.ndarray[(n,), numpy.int64]`:
            The parent indexes of haloes in the catalogue with at least one particle of the specified
            type.
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_halo_top_level_parent_IDs(self, particle_type: ParticleType|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """
        Get a list of halo IDs that indicate the top-most parent of a particular halo's hierarchy.
        Length is identical to that returned by `get_halo_IDs` for the same arguments.
        Optionally, specify a particle type to get only get parents of haloes containing those
        particles (the parents do not necessarily contain particles of the specified type).

        Parameters:
            (optional) `ParticleType|None` particle_type:
                Get results for only haloes that have at least one particle of this type.

        Returns `np.ndarray[(n,), numpy.int64]`:
            The top-most parent IDs of haloes in the catalogue with at least one particle of the
            specified type.
        """
        raise NotImplementedError("Attempted to call an abstract method.")
    @abstractmethod
    def get_halo_top_level_parent_indexes(self, particle_type: ParticleType|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """
        Get a list of halo indexes that indicate the top-most parent of a particular halo's hierarchy.
        Length is identical to that returned by `get_halo_indexes` for the same arguments.
        Optionally, specify a particle type to get only get parents of haloes containing those
        particles (the parents do not necessarily contain particles of the specified type).

        Parameters:
            (optional) `ParticleType|None` particle_type:
                Get results for only haloes that have at least one particle of this type.

        Returns `np.ndarray[(n,), numpy.int64]`:
            The top-most parent indexes of haloes in the catalogue with at least one particle of
            the specified type.
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_halo_centres_of_mass(self, particle_type: ParticleType|None = None, use_proper_units: bool = False) -> unyt_array:
        """
        Read the coordinates for the centre of mass for each halo.

        Parameters:
            (optional) `ParticleType|None` particle_type:
                Get results for only haloes that have at least one particle of this type.
            (optional) `bool` use_proper_units:
                Convert the data to proper coordinates.
                Default is `False`.

        Returns `unyt.unyt_array[Mpc, (n,3), numpy.float64]`:
            Using 64-bit floats.

        See also:
            `get_halo_centres_of_potential`
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_halo_centres_of_potential(self, particle_type: ParticleType|None = None, use_proper_units: bool = False) -> unyt_array:
        """
        Read the coordinates for the centre of potential for each halo.

        Parameters:
            (optional) `ParticleType|None` particle_type:
                Get results for only haloes that have at least one particle of this type.
            (optional) `bool` use_proper_units:
                Convert the data to proper coordinates.
                Default is `False`.

        Returns `unyt.unyt_array[Mpc, (n,3), numpy.float64]`:
            Using 64-bit floats.

        See also:
            `get_halo_centres_of_mass`
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_halo_masses(self, halo_type: IHaloDefinition, particle_type: ParticleType|None = None) -> unyt_array:
        """
        Get the total mass of each halo according to a particular halo definition.

        Parameters:
            `IHaloDefinition` halo_type:
                Specification for how a halo is defined.
                See `BasicHaloDefinitions` for some built-in halo definitions.
            (optional) `ParticleType|None` particle_type:
                Get results for only haloes that have at least one particle of this type.

        Returns `unyt.unyt_array[Msun, (n,), numpy.float64]`:
            Using 64-bit floats.
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_halo_radii(self, halo_type: IHaloDefinition, particle_type: ParticleType|None = None, use_proper_units: bool = False) -> unyt_array:
        """
        Get the radius of each halo according to a particular halo definition.

        Parameters:
            `IHaloDefinition` halo_type:
                Specification for how a halo is defined.
                See `BasicHaloDefinitions` for some built-in halo definitions.
            (optional) `ParticleType|None` particle_type:
                Get results for only haloes that have at least one particle of this type.
            (optional) `bool` use_proper_units:
                Convert the data to proper coordinates.
                Default is `False`.

        Returns `unyt.unyt_array[Mpc, (n,), numpy.float64]`:
            Using 64-bit floats.
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_halo_IDs_by_snapshot_particle(self, particle_type: ParticleType, include_unbound: bool = True, snapshot_particle_ids: np.ndarray[tuple[int], np.dtype[np.int64]]|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """
        Get a list of halo IDs - one for each particle in the snapshot.
        Particles with no associated halo receive an ID according to the halo ID scheme used.

        Parameters:
            `ParticleType` particle_type:
                The target particle type.
            (optional) `bool` include_unbound:
                Include particles that are unbound from their associated halo.
                Default is `True`.
            (optional) `np.ndarray[(N,), numpy.int64]|None` snapshot_particle_ids:
                Specify particular particle IDs to use. This can be either a subset of the snapshot
                or the full set to avoid unnecessary IO operations (implementation dependant).

        Returns `np.ndarray[(N,), numpy.int64]`:
            The halo ID each particle is associated with.
        """
        raise NotImplementedError("Attempted to call an abstract method.")
    @abstractmethod
    def get_halo_indexes_by_snapshot_particle(self, particle_type: ParticleType, include_unbound: bool = True, snapshot_particle_ids: np.ndarray[tuple[int], np.dtype[np.int64]]|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """
        Get a list of halo indexes - one for each particle in the snapshot.
        Particles with no associated halo receive an index of -1.

        Parameters:
            `ParticleType` particle_type:
                The target particle type.
            (optional) `bool` include_unbound:
                Include particles that are unbound from their associated halo.
                Default is `True`.
            (optional) `np.ndarray[(N,), numpy.int64]|None` snapshot_particle_ids:
                Specify particular particle IDs to use. This can be either a subset of the snapshot
                or the full set to avoid unnecessary IO operations (implementation dependant).

        Returns `np.ndarray[(N,), numpy.int64]`:
            The halo index each particle is associated with.
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_particle_IDs(self, particle_type: ParticleType, include_unbound: bool = True) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """
        Get a list of particle IDs that are included in the catalogue.
        Set 'include_unbound' to False to retrieve only bound particles.

        Parameters:
            `ParticleType` particle_type:
                The target particle type.
            (optional) `bool` include_unbound:
                Include particles that are unbound from their associated halo.
                Default is `True`.

        Returns `np.ndarray[(?,), numpy.int64]`:
            The halo index each particle is associated with.
        """
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

    # Internal methods

    def __calculate_descendant_info(self) -> None:
        self.__direct_children, self.__total_descendants = CatalogueBase._calculate_n_children(*self._get_hierarchy_IDs())

    @staticmethod
    def _calculate_n_children(halo_ids: np.ndarray[tuple[int], np.dtype[np.int64]], parent_ids: np.ndarray[tuple[int], np.dtype[np.int64]]) -> tuple[np.ndarray[tuple[int], np.dtype[np.int64]], np.ndarray[tuple[int], np.dtype[np.int64]]]:
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

    # async versions (experimental)

    async def get_number_of_haloes_async(self, particle_type: ParticleType|None = None) -> Awaitable[int]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_number_of_haloes, particle_type)
    
    async def get_halo_IDs_async(self, particle_type: ParticleType|None = None) -> Awaitable[np.ndarray]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_IDs, particle_type)

    async def get_halo_parent_IDs_async(self, particle_type: ParticleType|None = None) -> Awaitable[np.ndarray]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_parent_IDs, particle_type)
    
    async def get_halo_top_level_parent_IDs_async(self, particle_type: ParticleType|None = None) -> Awaitable[np.ndarray]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_top_level_parent_IDs, particle_type)
    
    async def get_halo_IDs_by_snapshot_particle_async(self, particle_type: ParticleType, include_unbound: bool = True) -> Awaitable[np.ndarray]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_IDs_by_snapshot_particle, particle_type, include_unbound)

    async def get_particle_IDs_async(self, particle_type: ParticleType, include_unbound: bool = True) -> Awaitable[np.ndarray]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_particle_IDs, particle_type, include_unbound)

    async def get_halo_centres_of_mass_async(self, particle_type: ParticleType|None = None) -> Awaitable[unyt_array]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_centres_of_mass, particle_type)

    async def get_halo_centres_of_potential_async(self, particle_type: ParticleType|None = None) -> Awaitable[unyt_array]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_centres_of_potential, particle_type)

    async def get_halo_masses_async(self, halo_type: IHaloDefinition, particle_type: ParticleType|None = None) -> Awaitable[unyt_array]:
        """
        EXPERIMENTAL
        """
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_halo_masses, halo_type, particle_type)






'''
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
'''
