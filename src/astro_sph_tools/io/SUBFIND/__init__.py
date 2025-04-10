# SPDX-FileCopyrightText: 2025-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from typing import cast as typing_cast, TypeVar

import numpy as np
from unyt import unyt_array
from pyread_eagle import EagleSnapshot
import h5py as h5
from QuasarCode import Console, Settings, Stopwatch
from QuasarCode.MPI import MPI_Config, mpi_barrier, mpi_gather_array

from ...data_structures._ParticleType import ParticleType
from ...tools._ArrayReorder import ArrayReorder, ArrayReorder_2
from ...data_structures._SimulationData import T_ISimulation, SimulationDataBase
from ..data_structures._CatalogueBase import CatalogueBase, IHaloDefinition, FOFGroup, SphericalOverdensityAperture, CriticalSphericalOverdensityAperture, MeanSphericalOverdensityAperture, TopHatSphericalOverdensityAperture



T = TypeVar("T", bound=np.generic)

class CatalogueSUBFIND(CatalogueBase[T_ISimulation]):
    """
    SUBFIND catalogue data.

    Base class that covers general SUBFIND features.
    Create a child class with implementation specific features to use this class.
    """

    LimitedMode: bool = False

    def __init__(
        self,
        properties_filepaths: list[str],
        snapshot: SimulationDataBase[T_ISimulation],
        membership_filepath: str = "",
    ) -> None:
        
    #TODO: from here onwards is just a copy of the EAGLE file
    
    def _get_hierarchy_IDs(self) -> tuple[np.ndarray[tuple[int], np.dtype[np.int64]], np.ndarray[tuple[int], np.dtype[np.int64]]]:
        indexes = self.get_halo_indexes()
        return (indexes, indexes)

    def get_number_of_haloes(self, particle_type: ParticleType|None = None) -> int:
        if CatalogueSUBFIND.LimitedMode and particle_type is not None:
            raise RuntimeError("CatalogueSUBFIND object in limited mode - unable to use particle type arguments other than None.")
        return self.__n_haloes[particle_type]

    def get_halo_IDs(self, particle_type: ParticleType|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        return self.get_halo_indexes(particle_type = particle_type) + 1 # FOF group numbers are just numbers instead of indexes

    def get_halo_indexes(self, particle_type: ParticleType|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        if CatalogueSUBFIND.LimitedMode and particle_type is not None:
            raise RuntimeError("CatalogueSUBFIND object in limited mode - unable to use particle type arguments other than None.")
        return np.array(list(range(self.__n_total_FOF_groups)), dtype = np.int64)[self.__FOF_groups_containing_parttypes[particle_type]]

    def get_halo_parent_IDs(self, particle_type: ParticleType|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        raise NotImplementedError("FOF groups in EAGLE SUBFIND catalogues have no parent structure.")

    def get_halo_parent_indexes(self, particle_type: ParticleType|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        #return np.full(self.get_number_of_haloes(particle_type), -1, dtype = int) # Just using the FOF groups, so no tree structure
        raise NotImplementedError("FOF groups in EAGLE SUBFIND catalogues have no parent structure.")

    def get_halo_top_level_parent_IDs(self, particle_type: ParticleType|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        return self.get_halo_IDs(particle_type)

    def get_halo_top_level_parent_indexes(self, particle_type: ParticleType|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        return self.get_halo_indexes(particle_type)

    def get_halo_centres_of_mass(self, particle_type: ParticleType|None = None, use_proper_units: bool = False) -> unyt_array:
        raise NotImplementedError("No centre of mass data in SUBFIND catalogues.")

    def get_halo_centres_of_potential(self, particle_type: ParticleType|None = None, use_proper_units: bool = False) -> unyt_array:
        data, h_exp, a_exp, cgs = self.get_FOF_field(
            field = "GroupCentreOfPotential",
            dtype = np.float64,
            particle_type = particle_type
        )
        return self.snapshot.make_cgs_data("cm", data, h_exp = h_exp, cgs_conversion_factor = cgs, a_exp = a_exp if use_proper_units else 0).to("Mpc")

    def get_halo_masses(self, halo_type: IHaloDefinition, particle_type: ParticleType|None = None) -> unyt_array:
        if halo_type not in (
            CatalogueBase.BasicHaloDefinitions.FOF_GROUP.value,
            CatalogueBase.BasicHaloDefinitions.SO_200_CRIT.value,
            CatalogueBase.BasicHaloDefinitions.SO_500_CRIT.value,
            CatalogueBase.BasicHaloDefinitions.SO_2500_CRIT.value,
            CatalogueBase.BasicHaloDefinitions.SO_200_MEAN.value,
            CatalogueBase.BasicHaloDefinitions.SO_500_MEAN.value,
            CatalogueBase.BasicHaloDefinitions.SO_2500_MEAN.value,
            CatalogueBase.BasicHaloDefinitions.SO_200_TOP_HAT.value
        ):
            raise halo_type.make_error(CatalogueSUBFIND)

        field_name: str
        if isinstance(halo_type, FOFGroup):
            field_name = "GroupMass"
        else:
            field_specifier: str
            if isinstance(halo_type, CriticalSphericalOverdensityAperture):
                field_specifier = "Crit"
            elif isinstance(halo_type, CriticalSphericalOverdensityAperture):
                field_specifier = "Crit"
            elif isinstance(halo_type, CriticalSphericalOverdensityAperture):
                field_specifier = "Crit"
            else:
                Console.print_debug(halo_type)
                Console.print_debug(halo_type.__dir__())
                Console.print_debug(particle_type)
                raise RuntimeError("This should not be possible! Please report this error!")
            field_name = f"Group_M_{field_specifier}{typing_cast(SphericalOverdensityAperture, halo_type).overdensity_limit}"

        data, h_exp, a_exp, cgs = self.get_FOF_field(
            field = field_name,
            dtype = np.float64,
            particle_type = particle_type
        )
        return self.snapshot.make_cgs_data("g", data, h_exp = h_exp, cgs_conversion_factor = cgs).to("Msun")

    def get_halo_radii(self, halo_type: IHaloDefinition, particle_type: ParticleType|None = None, use_proper_units: bool = False) -> unyt_array:

        if halo_type not in (
            CatalogueBase.BasicHaloDefinitions.SO_200_CRIT.value,
            CatalogueBase.BasicHaloDefinitions.SO_500_CRIT.value,
            CatalogueBase.BasicHaloDefinitions.SO_2500_CRIT.value,
            CatalogueBase.BasicHaloDefinitions.SO_200_MEAN.value,
            CatalogueBase.BasicHaloDefinitions.SO_500_MEAN.value,
            CatalogueBase.BasicHaloDefinitions.SO_2500_MEAN.value,
            CatalogueBase.BasicHaloDefinitions.SO_200_TOP_HAT.value
        ):
            raise halo_type.make_error(CatalogueSUBFIND)

        field_specifier: str
        if isinstance(halo_type, CriticalSphericalOverdensityAperture):
            field_specifier = "Crit"
        elif isinstance(halo_type, CriticalSphericalOverdensityAperture):
            field_specifier = "Crit"
        elif isinstance(halo_type, CriticalSphericalOverdensityAperture):
            field_specifier = "Crit"
        else:
            raise RuntimeError("This should not be possible! Please report this error!")
        field_name = f"Group_R_{field_specifier}{typing_cast(SphericalOverdensityAperture, halo_type).overdensity_limit}"

        data, h_exp, a_exp, cgs = self.get_FOF_field(
            field = field_name,
            dtype = np.float64,
            particle_type = particle_type
        )
        return self.snapshot.make_cgs_data("cm", data, h_exp = h_exp, cgs_conversion_factor = cgs, a_exp = a_exp if use_proper_units else 0).to("Mpc")

    def get_halo_IDs_by_snapshot_particle(self, particle_type: ParticleType, snapshot_particle_ids: np.ndarray[tuple[int], np.dtype[np.int64]]|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        group_numbers =  self.snapshot.get_group_ID(particle_type = particle_type, include_nearby_unattached_particles = False)
        if snapshot_particle_ids is None:
            return group_numbers.astype(np.int64)
        else:
            result =  ArrayReorder.create(
                self.snapshot.get_IDs(particle_type = particle_type),
                snapshot_particle_ids
            )(group_numbers, default_value = SnapshotEAGLE.EAGLE_MAX_GROUP_NUMBER)
            Console.print_debug("Done reordering.")
            return result

    def get_halo_indexes_by_snapshot_particle(self, particle_type: ParticleType, snapshot_particle_ids: np.ndarray[tuple[int], np.dtype[np.int64]]|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        numbers = self.get_halo_IDs_by_snapshot_particle(particle_type = particle_type, snapshot_particle_ids = snapshot_particle_ids)
        numbers[numbers == SnapshotEAGLE.EAGLE_MAX_GROUP_NUMBER] = 0
        return numbers - 1

    def get_halo_IDs_by_all_snapshot_particles(self, particle_type: ParticleType, snapshot_particle_ids: np.ndarray[tuple[int], np.dtype[np.int64]]|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]|None:
        return mpi_gather_array(self.get_halo_IDs_by_snapshot_particle(particle_type = particle_type, snapshot_particle_ids = snapshot_particle_ids))

    def get_halo_indexes_by_all_snapshot_particles(self, particle_type: ParticleType, snapshot_particle_ids: np.ndarray[tuple[int], np.dtype[np.int64]]|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]|None:
        return mpi_gather_array(self.get_halo_indexes_by_snapshot_particle(particle_type = particle_type, snapshot_particle_ids = snapshot_particle_ids))
#    def get_halo_indexes_by_all_snapshot_particles(self, particle_type: ParticleType, snapshot_particle_ids: np.ndarray[tuple[int], np.dtype[np.int64]]|None = None) -> np.ndarray[tuple[int], np.dtype[np.int64]]|None:
#        Console.print_debug("Reconstructing snapshot order data for halo membership.")
#        group_numbers, _, _, _ = self.get_membership_field(particle_type = particle_type, field = "GroupNumber", dtype = np.int64)
#        fof_group_only_mask = group_numbers > 0 # Only include FOF particles (any -ve value is in the SO radius but not part of FOF)
#        result =  ArrayReorder.create(
#            self.get_membership_field(particle_type = particle_type, field = "ParticleIDs", dtype = np.int64)[0],
#            snapshot_particle_ids if snapshot_particle_ids is not None else self.snapshot.get_IDs(particle_type),
#            source_order_filter = fof_group_only_mask
#        )(group_numbers, default_value = -1) - 1
#        result[result == -2] = -1
#        Console.print_debug("Done reconstructing.")
#        return result

    def get_particle_IDs(self, particle_type: ParticleType, include_unbound: bool = True) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        if not include_unbound:
            raise NotImplementedError("include_unbound param not supported for EAGLE data.")
        return typing_cast(np.ndarray[tuple[int], np.dtype[np.int64]], self.get_membership_field(
            particle_type = particle_type,
            field = "ParticleIDs",
            dtype = np.int64
        )[0])

    # Raw data access methods

    #TODO: when using MPI, only read nessessary particles (HOW!?) -------------------------------------------------------------------------------
    def get_membership_field(self, particle_type: ParticleType, field: str, dtype: type[T]) -> tuple[np.ndarray[tuple[int, ...], np.dtype[T]], float, float, float]:
        Console.print_verbose_info(f"Reading catalogue membership {particle_type.name} particle dataset \"{field}\".")
        files_with_particles = np.where(self.__n_membership_particles_per_file[:, particle_type.value] > 0)[0]
        if len(files_with_particles) == 0:
            raise IOError(f"No files in snapshot's catalogue contained {particle_type.name} particles.")

#        result = self.snapshot._file_object.read_extra_dataset(particle_type.value, field, self.__membership_files[0])#TODO: circular reference as this is needed to be called before the super constructor call! this dosent work as there are a different number of particles!!!!!!!

        first_file_with_part_type_field = files_with_particles[0]
        result = np.empty(self.__n_total_membership_particles[particle_type.value], dtype = dtype)
        for i in range(self.__n_parallel_components_membership):
            if self.__n_membership_particles_per_file[i, particle_type.value] == 0:
                continue
            chunk = slice(self.__membership_file_particle_offsets[i][particle_type.value], self.__membership_file_particle_end_offsets[i][particle_type.value])
#            result[chunk] = self.__membership_files[i][field][:]
            with h5.File(self.__membership_files[i], "r") as file:
                result[chunk] = file[particle_type.common_hdf5_name][field][:]
        with h5.File(self.__membership_files[first_file_with_part_type_field], "r") as file:
            conversion_values = (
                file[particle_type.common_hdf5_name][field].attrs["h-scale-exponent"],
                file[particle_type.common_hdf5_name][field].attrs["aexp-scale-exponent"],
                file[particle_type.common_hdf5_name][field].attrs["CGSConversionFactor"]
            )
        Console.print_debug("Done reading data.")
        return (
            result,
            *conversion_values
        )

    def get_FOF_field(self, field: str, dtype: type[T], particle_type: ParticleType|None = None) -> tuple[np.ndarray[tuple[int, ...], np.dtype[T]], float, float, float]:
        if CatalogueSUBFIND.LimitedMode and particle_type is not None:
            raise RuntimeError("CatalogueSUBFIND object in limited mode - unable to use particle type arguments other than None.")
        Console.print_verbose_info(f"Reading catalogue dataset \"{field}\".")
        with h5.File(self.__halo_data_files[0], "r") as file:
            conversion_values = (
                file["FOF"][field].attrs["h-scale-exponent"],
                file["FOF"][field].attrs["aexp-scale-exponent"],
                file["FOF"][field].attrs["CGSConversionFactor"]
            )
            field_element_shape = file["FOF"][field].shape[1:]
        result = np.empty((self.__n_total_FOF_groups, *field_element_shape), dtype = dtype)
        for i in range(self.__n_parallel_components_properties):
            if self.__n_FOF_groups_per_file[i] == 0:
                continue
            chunk = slice(self.__FOF_data_offsets[i], self.__FOF_data_end_offsets[i])
#            result[chunk] = self.__halo_data_files[i]["FOF"][field][:]
            with h5.File(self.__halo_data_files[i], "r") as file:
                result[chunk] = file["FOF"][field][:]
        Console.print_debug("Done reading data.")
        return (
            result[self.__FOF_groups_containing_parttypes[particle_type]],
            *conversion_values
        )

#    def get_subhalo_field(self, field: str, dtype = float) -> np.ndarray:
#        result = np.empty(self.__n_total_subhaloes, dtype = dtype)
#        for i in self.__n_parallel_components_properties:
#            chunk = slice(self.__subhalo_data_offsets[i], self.__subhalo_data_end_offsets[i])
#            result[chunk] = self.__halo_data_files[i]["Subhalo"][field][:]
#        return result
