# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
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
from ..data_structures._CatalogueBase import CatalogueBase, IHaloDefinition, FOFGroup, SphericalOverdensityAperture, CriticalSphericalOverdensityAperture, MeanSphericalOverdensityAperture, TopHatSphericalOverdensityAperture
from ._SnapshotEAGLE import SnapshotEAGLE
from ._sim_type import SimType_EAGLE



T = TypeVar("T", bound=np.generic)

class CatalogueSUBFIND(CatalogueBase[SimType_EAGLE]):
    """
    SUBFIND catalogue data (EAGLE).
    """

    LimitedMode: bool = False

    def __init__(
        self,
        membership_filepaths: list[str],
        properties_filepaths: list[str],
        snapshot: SnapshotEAGLE,
    ) -> None:
        Console.print_debug(f"Loading SUBFIND catalogue data from: \"{membership_filepaths[0]}\" and \"{properties_filepaths[0]}\".")
        if CatalogueSUBFIND.LimitedMode:
            Console.print_verbose_warning("CatalogueSUBFIND object being loaded using limited mode - only None will be supported for particle type arguments.")

        if Settings.debug:
            stopwatch = Stopwatch.start_new("Catalogue Constructor")

            barrier_time = stopwatch.get_elapsed_time_lap()
        mpi_barrier()
        if Settings.debug:
            all_barrier_times = MPI_Config.comm.gather(barrier_time, root = MPI_Config.root)
            if MPI_Config.is_root:
                Console.print_debug("Cat init barrier delay:", ", ".join([f"{v - min(all_barrier_times):.1f}" for v in all_barrier_times]))
        
        self.__n_parallel_components_membership = len(membership_filepaths)
        self.__n_parallel_components_properties = len(properties_filepaths)

        # File handles
        self.__membership_files = membership_filepaths#[h5.File(membership_filepath, "r") for membership_filepath in membership_filepaths]
        self.__halo_data_files  = properties_filepaths#[h5.File(properties_filepath, "r") for properties_filepath in properties_filepaths]

        if Settings.debug:
            barrier_time = stopwatch.get_elapsed_time_lap()
        mpi_barrier()
        if Settings.debug:
            all_barrier_times = MPI_Config.comm.gather(barrier_time, root = MPI_Config.root)
            if MPI_Config.is_root:
                Console.print_debug("Cat init barrier delay:", ", ".join([f"{v - min(all_barrier_times):.1f}" for v in all_barrier_times]))

        Console.print_debug("    Loading membership sizes.")
        n_parts_per_file = [None] * self.__n_parallel_components_membership
        for i in range(self.__n_parallel_components_membership):
            with h5.File(self.__membership_files[i], "r") as file:
                if i == 0:
                    self.__n_total_membership_particles: np.ndarray = file["Header"].attrs["NumPart_Total"]
                n_parts_per_file[i] = file["Header"].attrs["NumPart_ThisFile"]
#        self.__n_membership_particles_per_file: np.ndarray = np.row_stack([self.__membership_files[i]["Header"].attrs["NumPart_Total"] for i in range(self.__n_parallel_components_membership)], dtype = int)
        self.__n_membership_particles_per_file: np.ndarray = np.row_stack(n_parts_per_file, dtype = int)
        self.__membership_file_particle_end_offsets: np.ndarray = np.cumsum(self.__n_membership_particles_per_file, axis = 0, dtype = int)
        self.__membership_file_particle_offsets: np.ndarray = np.row_stack([np.zeros_like(self.__n_total_membership_particles, dtype = int), self.__membership_file_particle_end_offsets[:-1, :]], dtype = int)

        if Settings.debug:
            barrier_time = stopwatch.get_elapsed_time_lap()
        mpi_barrier()
        if Settings.debug:
            all_barrier_times = MPI_Config.comm.gather(barrier_time, root = MPI_Config.root)
            if MPI_Config.is_root:
                Console.print_debug("Cat init barrier delay:", ", ".join([f"{v - min(all_barrier_times):.1f}" for v in all_barrier_times]))

        Console.print_debug("    Loading properties sizes.")
        n_groups_per_file = [None] * self.__n_parallel_components_properties
        for i in range(self.__n_parallel_components_properties):
            try:
                with h5.File(self.__halo_data_files[i], "r") as file:
                    if i == 0:
                        self.__n_total_FOF_groups: int = int(file["FOF"].attrs["TotNgroups"])
                    n_groups_per_file[i] = file["FOF"].attrs["Ngroups"]
            except Exception as e:
                Console.print_error(f"Error in file: {self.__halo_data_files[i]}")
                raise e
        if self.__n_total_FOF_groups != sum(n_groups_per_file):
            Console.print_warning("More FOF haloes in catalogue than reported. Assuming aggregate number as correct.")
            self.__n_total_FOF_groups = sum(n_groups_per_file)
#        self.__n_total_FOF_groups: int = int(self.__halo_data_files[0]["FOF"].attrs["TotNgroups"])
#        self.__n_FOF_groups_per_file: np.ndarray = np.array([self.__halo_data_files[i]["FOF"].attrs["Ngroups"] for i in range(self.__n_parallel_components_properties)], dtype = int)
        self.__n_FOF_groups_per_file: np.ndarray = np.array(n_groups_per_file, dtype = int)
        self.__FOF_data_end_offsets: np.ndarray = np.cumsum(self.__n_FOF_groups_per_file, dtype = int)
        self.__FOF_data_offsets: np.ndarray = np.array([0, *self.__FOF_data_end_offsets[:-1]], dtype = int)

#        self.__n_total_subhaloes: int = int(self.__halo_data_files[0]["Subhalo"].attrs["TotNgroups"])
#        self.__n_subhaloes_per_file: np.ndarray = np.array([self.__halo_data_files[i]["Subhalo"].attrs["Ngroups"] for i in range(self.__n_parallel_components_properties)], dtype = int)
#        self.__subhalo_data_end_offsets: np.ndarray = np.cumsum(self.__n_subhaloes_per_file, dtype = int)
#        self.__subhalo_data_offsets: np.ndarray = np.array([0, *np.cumsum(self.__subhalo_data_end_offsets, dtype = int)[:-1]], dtype = int)

        self.__FOF_groups_containing_parttypes: dict[ParticleType|None, np.ndarray[tuple[int], np.dtype[np.bool_]]] = {}
        if not CatalogueSUBFIND.LimitedMode:

            if Settings.debug:
                barrier_time = stopwatch.get_elapsed_time_lap()
            mpi_barrier()
            if Settings.debug:
                all_barrier_times = MPI_Config.comm.gather(barrier_time, root = MPI_Config.root)
                if MPI_Config.is_root:
                    Console.print_debug("Cat init barrier delay:", ", ".join([f"{v - min(all_barrier_times):.1f}" for v in all_barrier_times]))

            Console.print_debug("    Loading particle type membership.")
            for part_type in ParticleType.get_all():
                try:
                    group_numbers, _, _, _ = self.get_membership_field(
                        particle_type = part_type,
                        field = "GroupNumber",
                        dtype = np.int64
                    )
                    self.__FOF_groups_containing_parttypes[part_type] = np.unique(group_numbers[group_numbers > 0]) - 1
                except IOError:
                    self.__FOF_groups_containing_parttypes[part_type] = np.full(self.__n_total_FOF_groups, False, dtype = bool)
        # Include a 'filter' for no specific type
        self.__FOF_groups_containing_parttypes[None] = np.full(self.__n_total_FOF_groups, True, dtype = bool)

        # Pre-calculate the number of haloes for each option
        self.__n_haloes: dict[ParticleType|None, int] = { key : int(self.__FOF_groups_containing_parttypes[key].sum()) for key in self.__FOF_groups_containing_parttypes }

        if Settings.debug:
            barrier_time = stopwatch.get_elapsed_time_lap()
        mpi_barrier()
        if Settings.debug:
            all_barrier_times = MPI_Config.comm.gather(barrier_time, root = MPI_Config.root)
            if MPI_Config.is_root:
                Console.print_debug("Cat init barrier delay:", ", ".join([f"{v - min(all_barrier_times):.1f}" for v in all_barrier_times]))

        Console.print_debug("    Initialising base class.")

        super().__init__(
            membership_filepath = membership_filepaths[0],
            properties_filepath = properties_filepaths[0],
            snapshot = snapshot
        )

        if Settings.debug:
            barrier_time = stopwatch.get_elapsed_time_lap()
        mpi_barrier()
        if Settings.debug:
            all_barrier_times = MPI_Config.comm.gather(barrier_time, root = MPI_Config.root)
            if MPI_Config.is_root:
                Console.print_debug("Cat init barrier (last) delay:", ", ".join([f"{v - min(all_barrier_times):.1f}" for v in all_barrier_times]))

        if Settings.debug:
            stopwatch.stop()

        Console.print_debug("    Done creating SUBFIND catalogue reader.")

    # Overrides of base class methods for re-typing

    @property
    def snapshot(self) -> SnapshotEAGLE:
        """
        `SnapshotEAGLE` Snapshot object for snapshot data associated with this catalogue.
        """
        return typing_cast(SnapshotEAGLE, super().snapshot)
    
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













#    def get_number_of_haloes(self, particle_type: ParticleType|None = None) -> int:
#        return self.__n_haloes[particle_type]
#
#    def get_halo_IDs(self, particle_type: ParticleType|None = None) -> np.ndarray:#TODO: using indexes - update api to make this clear!
#        return np.array(list(range(self.__n_total_FOF_groups)), dtype = int)[self.__FOF_groups_containing_parttypes[particle_type]]
#
#    def get_halo_parent_IDs(self, particle_type: ParticleType|None = None) -> np.ndarray:#TODO: using indexes - update api to make this clear!
#        return np.full(self.get_number_of_haloes(particle_type), -1, dtype = int) # Just using the FOF groups, so no tree structure
#
#    def get_halo_top_level_parent_IDs(self, particle_type: ParticleType|None = None) -> np.ndarray:#TODO: using indexes - update api to make this clear!
#        return self.get_halo_IDs(particle_type) # Just using the FOF groups, so no tree structure. Therfore, own ID is top-level ID
#
#    def get_halo_IDs_by_snapshot_particle(self, particle_type: ParticleType, include_unbound: bool = True, snapshot_particle_ids: np.ndarray|None = None) -> np.ndarray:#TODO: using indexes - update api to make this clear!
#        Console.print_debug("Reconstructing snapshot order data for halo membership.")
#        if not include_unbound:
#            raise NotImplementedError("include_unbound param not supported for EAGLE data.")
#        group_numbers = self.get_membership_field("GroupNumber", particle_type, int)[0]
#        fof_group_only_mask = group_numbers > 0
#        result =  ArrayReorder.create(
#            self.get_membership_field("ParticleIDs", particle_type, int)[0],
#            snapshot_particle_ids if snapshot_particle_ids is not None else self.snapshot.get_IDs(particle_type),
#            source_order_filter = fof_group_only_mask
#        )(group_numbers, default_value = -1) - 1
#        result[result == -2] = -1
#        Console.print_debug("Done reconstructing.")
#        return result
#
#    def get_particle_IDs(self, particle_type: ParticleType, include_unbound: bool = True) -> np.ndarray:
#        if not include_unbound:
#            raise NotImplementedError("include_unbound param not supported for EAGLE data.")
#        return self.get_membership_field("ParticleIDs", particle_type, int)[0]
#
#    def get_halo_centres(self, particle_type: ParticleType|None = None, use_proper_units: bool = False) -> unyt_array:#TODO: add param for physical or co-moving (add to whole API)
#        data, h_exp, a_exp, cgs = self.get_FOF_field("GroupCentreOfPotential", particle_type, float)
#        return self.snapshot.make_cgs_data("cm", data, h_exp = h_exp, cgs_conversion_factor = cgs, a_exp = a_exp if use_proper_units else 0).to("Mpc")
#
#    def get_halo_masses(self, particle_type: ParticleType|None = None) -> unyt_array:
#        data, h_exp, a_exp, cgs = self.get_FOF_field("Group_M_Crit200", particle_type, float)
#        return self.snapshot.make_cgs_data("g", data, h_exp = h_exp, cgs_conversion_factor = cgs).to("Msun")
#
#    def get_halo_radii(self, particle_type: ParticleType|None = None, use_proper_units: bool = False) -> unyt_array:
#        """
#        Get the radii (Group_R_Crit200) of each halo.
#        """
#        data, h_exp, a_exp, cgs = self.get_FOF_field("Group_R_Crit200", particle_type, float)
#        return self.snapshot.make_cgs_data("cm", data, h_exp = h_exp, cgs_conversion_factor = cgs, a_exp = a_exp if use_proper_units else 0).to("Mpc")
#
#    def _get_hierarchy_IDs(self) -> tuple[np.ndarray, np.ndarray]:
#        indexes = self.get_halo_IDs()
#        return (indexes, indexes)









#    @staticmethod
#    def generate_filepaths(
#       *snapshot_number_strings: str,
#        directory: str,
#        membership_basename: str,
#        properties_basename: str,
#        file_extension: str = "hdf5",
#        parallel_ranks: list[int]|None = None
#    ) -> dict[
#            str,
#            tuple[
#                str|tuple[str, ...],
#                str|tuple[str, ...]
#            ]
#         ]:
#        raise NotImplementedError("Not implemented for EAGLE. Update to generalise file path creation.")#TODO:
#
#    @staticmethod
#    def scrape_filepaths(
#        directory: str,
#        ignore_basenames: list[str]|None = None
#    ) -> tuple[
#            tuple[str, ...],
#            str,
#            tuple[str, ...],
#            tuple[str, ...],
#            tuple[tuple[int, ...], ...],
#            tuple[tuple[int, ...], ...]
#         ]:
#
#        membership_pattern = re.compile(r'.*particledata_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]eagle_subfind_particles_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
#        properties_pattern = re.compile(r'.*groups_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]eagle_subfind_tab_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
#        
#        properties_info = {}
#        membership_info = {}
#
#        for root, _, files in os.walk(directory):
#            for filename in files:
#                filepath = os.path.join(root, filename)
#                membership_match = membership_pattern.match(filepath)
#                properties_match = properties_pattern.match(filepath)
#
#                if membership_match or properties_match:
#                    match = membership_match if membership_match else properties_match
#                    is_properties = bool(properties_match)
#
#                    number = match.group("number")
#                    redshift_int = match.group("redshift_int")
#                    redshift_dec = match.group("redshift_dec")
#                    parallel_index = int(match.group("parallel_index"))
#                    extension = match.group("extension")
#
#                    tag = f"{number}_z{redshift_int}p{redshift_dec}"
#                    if is_properties:
#                        basename = os.path.join(f"groups_{tag}", f"eagle_subfind_tab_{tag}")
#                        if tag not in properties_info:
#                            properties_info[tag] = [number, basename, extension, [parallel_index]]
#                        else:
#                            assert basename == properties_info[tag][1]
#                            assert extension == properties_info[tag][2]
#                            properties_info[tag][3].append(parallel_index)
#                    else:
#                        basename = os.path.join(f"particledata_{tag}", f"eagle_subfind_particles_{tag}")
#                        if tag not in membership_info:
#                            membership_info[tag] = [number, basename, extension, [parallel_index]]
#                        else:
#                            assert basename == membership_info[tag][1]
#                            assert extension == membership_info[tag][2]
#                            membership_info[tag][3].append(parallel_index)
#
#        for tag in properties_info:
#            assert tag in membership_info
#            properties_info[tag][3].sort()
#            membership_info[tag][3].sort()
#
#        tags = tuple(properties_info.keys())
#
#        return (
#            tuple([membership_info[tag][0] for tag in tags]),
#            os.path.abspath(directory),
#            tuple([membership_info[tag][1] for tag in tags]),
#            tuple([properties_info[tag][1] for tag in tags]),
#            tuple([tuple(membership_info[tag][3]) for tag in tags]),
#            tuple([tuple(properties_info[tag][3]) for tag in tags])
#        )
#    
#    @staticmethod
#    def generate_filepaths_from_partial_info(
#        directory: str,
#        membership_basename: str|None = None,
#        properties_basename: str|None = None,
#        snapshot_number_strings: list[str]|None = None,
#        file_extension: str|None = None,
#        parallel_ranks: list[int]|None = None
#    ) -> dict[
#            str,
#            tuple[
#                tuple[str, ...],
#                tuple[str, ...]
#            ]
#         ]:
#        if membership_basename is not None or properties_basename is not None or file_extension is not None or parallel_ranks is not None:
#            raise NotImplementedError("TODO: some fields not supported for EAGLE. Change API to use objects with file info specific to sim types.")#TODO:
#
#        nums, _, membership_basenames, properties_basenames, membership_parallel_indexes, properties_parallel_indexes = CatalogueSUBFIND.scrape_filepaths(directory)
#        data = { nums[i] : (membership_basenames[i], properties_basenames[i], membership_parallel_indexes[i], properties_parallel_indexes[i]) for i in range(len(nums))}
#
#        selected_files = {}
#        for num in (snapshot_number_strings if snapshot_number_strings is not None else nums):
#            if num not in nums:
#                raise FileNotFoundError("Snapshot numbers provided not all present in directory.")
#            selected_files[num] = (
#                tuple([os.path.join(directory, f"{data[num][0]}.{i}.hdf5") for i in data[num][2]]),
#                tuple([os.path.join(directory, f"{data[num][1]}.{i}.hdf5") for i in data[num][3]])
#            )
#
#        return selected_files
