# SPDX-FileCopyrightText: 2025-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed
from mpi4py import MPI
import datetime
import os

from QuasarCode import Console, Settings
from QuasarCode.MPI import MPI_Config
from QuasarCode.Tools import ScriptWrapper
from astro_sph_tools import FileTreeScraper_EAGLE, ParticleType
from astro_sph_tools.io.data_structures import SnapshotBase, CatalogueBase
from astro_sph_tools.tools import calculate_wrapped_displacement, calculate_wrapped_distance
import numpy as np
import h5py as h5
from scipy.spatial import KDTree

Console.mpi_output_root_rank_only()
Console.show_times()



def main():
    ScriptWrapper(
        command = "ast-find-nearest-haloes",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("1.0.0"),
        edit_date = datetime.date(2025, 1, 23),
        description = "Find the nearest halo to each particle.\nOnly haloes with a defined (nonzero) M_200 are considered.",
         parameters = ScriptWrapper.ParamSpec(
            ScriptWrapper.RequiredParam[float](
                name = "redshift",
                short_name = "z",
                sets_param = "target_redshift",
                conversion_function = float,
                description = "Redshift of the snapshot to use.\nIf a match is not found, use the file with the closest redshift with a higher value."
            ),
            ScriptWrapper.Flag(
                name = "EAGLE",
                sets_param = "is_EAGLE",
                description = "Running on EAGLE data.",
                conflicts = ["is_SWIFT"]
            ),
            ScriptWrapper.Flag(
                name = "SWIFT",
                sets_param = "is_SWIFT",
                description = "Running on data generated using SWIFT.",
                conflicts = ["is_EAGLE"]
            ),
            ScriptWrapper.OptionalParam[str](
                name = "data",
                short_name = "i",
                sets_param = "input_directory",
                default_value = "./",
                description = "Input simulation data directory. Defaults to the current working directory."
            ),
            ScriptWrapper.Flag(
                name = "snipshot",
                sets_param = "use_snipshots",
                description = "Use particle data from snipshots."
            ),
            ScriptWrapper.OptionalParam[list[str]](
                name = "ignore-files",
                sets_param = "skip_file_numbers",
                conversion_function = ScriptWrapper.make_list_converter(","),
                default_value = [],
                description = "Snapshot/snipshot numbers to be ignored. This can be used in the case of corrupted files.\nUse a comma seperated list."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "output-file",
                short_name = "o",
                sets_param = "output_filepath",
                description = "Name of file to write results to.\nThe file extension will be added automatically and may include a parallel index portion if using MPI.",
                default_value = "nearest-haloes"
            ),
            ScriptWrapper.Flag(
                name = "overwrite",
                sets_param = "allow_dataset_overwrite",
                description = "Allow an existing dataset at the redshift of the selected data to be overwritten.\nThis does nothing if no file yet exists or if there is no dataset already present.\nIntended as a protection against accedentally overwriting data - enable only if you are sure the right redshift is being selected!"
            ),
            ScriptWrapper.OptionalParam[list[float]](
                name = "minimum-halo-masses",
                sets_param = "minimum_log10_halo_masses",
                conversion_function = ScriptWrapper.make_list_converter(",", float),
                default_value = [],
                description = "Values of log10 halo mass (M_200) in Msun below which to exclude halos.\nUse a comma seperated list."
            )
        )
    ).run_with_async(__main)



async def __main(
            target_redshift: float,
            is_EAGLE: bool,
            is_SWIFT: bool,
            input_directory: str,
            use_snipshots: bool,
            skip_file_numbers: list[str],
            output_filepath: str,
            allow_dataset_overwrite: bool,
            minimum_log10_halo_masses: list[float]
          ) -> None:

    #LOGHALOMASSES = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]

    output_filepath = f"{output_filepath}.{MPI_Config.rank}.hdf5" if (Settings.mpi_avalible and MPI_Config.comm_size > 1) else f"{output_filepath}.hdf5"
    file_exists: bool = any(MPI_Config.comm.allgather(os.path.exists(output_filepath))) if Settings.mpi_avalible else os.path.exists(output_filepath)
    if file_exists:
        Console.print_info("Output file already exists.\nCheck for existing data will be performed after loading the snapshot.")

    # Validate sim type
    if not (is_EAGLE or is_SWIFT):
        Console.print_error("Must specify either EAGLE or SWIFT simulation type.")
        Console.print_info("Terminating...")
        return

    sim_files: FileTreeScraper_EAGLE
    if is_EAGLE:
        Console.print_info(f"Using EAGLE data from \"{input_directory}\".")
        sim_files = FileTreeScraper_EAGLE(
            input_directory,
            skip_snapshot_numbers = skip_file_numbers if not use_snipshots else None,
            skip_snipshot_numbers = skip_file_numbers if     use_snipshots else None
        )
    elif is_SWIFT:
        raise NotImplementedError("SWIFT support is not currently implemented.")
    else:
        raise RuntimeError("This should be impossible. Please report this error!")
    file_number = sim_files.snapshots.find_file_number_from_redshift(target_redshift)
    Console.print_info(f"Selected file number {file_number} for target redshift {target_redshift}.")

    cat: CatalogueBase
    snap: SnapshotBase
    if not use_snipshots:
        cat = sim_files.catalogues.get_by_number(file_number).load()
        snap = cat.snapshot
    else:
        cat = sim_files.snipshot_catalogues.get_by_number(file_number).load()
        snap = cat.snapshot
    data_redshift = snap.redshift
    Console.print_info(f"Loaded snapshot and catalogue at redshift {data_redshift}.")

    root_dataset_name = f"redshift_{data_redshift}"
    group_already_exists = False
    if file_exists:
        with h5.File(output_filepath, "r") as file:
            group_already_exists = root_dataset_name in file
        if not allow_dataset_overwrite:
            can_continue: bool = not (any(MPI_Config.comm.allgather(group_already_exists)) if Settings.mpi_avalible else group_already_exists)
            if not can_continue:
                Console.print_error("Output file already contains a dataset for this redshift.")
                Console.print_info("Terminating...")
                return






    box_width = snap.box_size[0].to("Mpc").value
    Console.print_info(f"Box size is {box_width} cMpc.")

    Console.print_info("Reading halo masses.")
    halo_masses = cat.get_halo_masses()

    Console.print_info("Creating halo mass masks.")

    # Set masks for different halo masses
    halo_mask_keys = []
    halo_masks: dict[float, np.ndarray] = {}
    # Only use haloes that have a defined M200 (and therfore have subfind haloes attached to the FOF group)
    halo_mask_keys.append(-np.inf)
    halo_masks[-np.inf] = halo_masses > 0.0
    for log_mass in minimum_log10_halo_masses:
        Console.print_info(f"    log10(M_200) > {log_mass}")
        halo_mask_keys.append(log_mass)
        halo_masks[log_mass] = halo_masses > 10**log_mass
    Console.print_info("    done.")

    del halo_masses

    Console.print_info("Reading snapshot particle positions.")
    particle_positions_this_rank: np.ndarray = snap.get_positions(ParticleType.gas).to("Mpc").value
    Console.print_info("Reading halo IDs.")
    halo_ids = cat.get_halo_IDs()
    Console.print_info("Reading halo positions.")
    halo_centres = cat.get_halo_centres().to("Mpc").value
    Console.print_info("Reading halo R_200.")
    halo_radii = cat.get_halo_radii().to("Mpc").value

    Console.print_info("Allocating memory for result data.")
    particle_nearest_halo_id:       np.ndarray = np.empty(shape = (particle_positions_this_rank.shape[0], len(halo_masks)), dtype = int)
    particle_nearest_halo_distance: np.ndarray = np.empty(shape = (particle_positions_this_rank.shape[0], len(halo_masks)), dtype = float)
    particle_nearest_halo_radius:   np.ndarray = np.empty(shape = (particle_positions_this_rank.shape[0], len(halo_masks)), dtype = float)



    Console.print_info("Searching.")
    for i, mask_key in enumerate(halo_mask_keys):
        Console.print_info(f"    Searching all haloes with log10(M) > {mask_key}.")
        if halo_masks[mask_key].sum() == 0:
            Console.print_warning("No haloes above this mass limit.")
            Console.print_warning("Setting null values.")
            particle_nearest_halo_id      [:, i] = -1
            particle_nearest_halo_distance[:, i] = np.inf
            particle_nearest_halo_radius  [:, i] = 0.0
            continue
        Console.print_info("    Creating KDTree.")
        tree = KDTree(
            halo_centres[halo_masks[mask_key]],
            boxsize = box_width
        )
        Console.print_info("    Querying.")
        distances, nearest_halo_indexes = tree.query(
            particle_positions_this_rank,
            workers = -1 # -1 means use all avalible CPU
        )
        Console.print_info("    Copying results.")
        #particle_nearest_halo_id      [:, i] = halo_ids[halo_masks[mask_key]][nearest_halo_indexes]
        np.take(halo_ids[halo_masks[mask_key]], indices = nearest_halo_indexes, axis = 0, out = particle_nearest_halo_id[:, i]) # Not sure if this is more memory efficient than fancy indexing
        particle_nearest_halo_distance[:, i] = distances
        #particle_nearest_halo_radius  [:, i] = halo_radii[halo_masks[mask_key]][nearest_halo_indexes]
        np.take(halo_radii[halo_masks[mask_key]], indices = nearest_halo_indexes, axis = 0, out = particle_nearest_halo_radius[:, i]) # Not sure if this is more memory efficient than fancy indexing
    Console.print_debug("    done.")



#    Console.print_info("Searching.")
#    for i in range(particle_positions_this_rank.shape[0]):
#        Console.print_debug(f"    Doing particle #{i + 1} / {particle_positions_this_rank.shape[0]}")
#        halo_distances_squared: np.ndarray = calculate_wrapped_distance(particle_positions_this_rank[i, :], halo_centres, box_width, do_squared_distance = True)
#        for j, mask_key in enumerate(halo_mask_keys):
#            closest_halo_index = np.where(halo_masks[mask_key])[0][halo_distances_squared[halo_masks[mask_key]].argmin()]
#            particle_nearest_halo_id      [i, j] = halo_ids[closest_halo_index]
#            particle_nearest_halo_distance[i, j] = np.sqrt(halo_distances_squared[closest_halo_index])
#            particle_nearest_halo_radius  [i, j] = halo_radii[closest_halo_index]
#    Console.print_debug("    done.")



    del particle_positions_this_rank, halo_ids, halo_centres, halo_radii

    Console.print_info("Writing results.")
    with h5.File(output_filepath, "w" if not file_exists else "a") as file:
        if group_already_exists:
            if not allow_dataset_overwrite:
                raise RuntimeError("This should be impossible. Please report this error!")
            raise NotImplementedError("TODO")
            #TODO: rename the existing group to "BACKUP__{name}" and first remove an existing backup if one exists.
        g = file.create_group(root_dataset_name)
        g.attrs["halo_masses"] = halo_mask_keys[1:] # Treat the all-fof-haloes-with-subfind case seperately
        g.create_dataset("halo_indexes", data = particle_nearest_halo_id[:, 0]).attrs["Description"] = "Catalogue index of the nearest halo."
        g.create_dataset("halo_comoving_distance", data = particle_nearest_halo_distance[:, 0]).attrs["Description"] = "Distance to the centre of the nearest halo in comoving Mpc."
        g.create_dataset("halo_comoving_radius", data = particle_nearest_halo_radius[:, 0]).attrs["Description"] = "R_200 of the nearest halo in comoving Mpc."
        g2 = g.create_group("minimum_halo_mass_limited")
        for i, log_mass in enumerate(halo_mask_keys[1:], start = 1):
            g3 = g2.create_group(f"{log_mass:.2f}")
            g3.attrs["minimum_halo_log10_M_200"] = log_mass
            g3.create_dataset("halo_indexes", data = particle_nearest_halo_id[:, i]).attrs["Description"] = "Catalogue index of the nearest halo."
            g3.create_dataset("halo_comoving_distance", data = particle_nearest_halo_distance[:, i]).attrs["Description"] = "Distance to the centre of the nearest halo in comoving Mpc."
            g3.create_dataset("halo_comoving_radius", data = particle_nearest_halo_radius[:, i]).attrs["Description"] = "R_200 of the nearest halo in comoving Mpc."
        if group_already_exists:
            pass#TODO: remove the backup group
