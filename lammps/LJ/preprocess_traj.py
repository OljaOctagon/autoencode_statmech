import numpy as np
import logging
import gc
from pathlib import Path

import freud
from lammpstools.tools import (
    pair_distance,
    get_nn_vector_one,
)
from lammpstools import (
    compute_cna_batch_from_pipelines,
    compute_cna_trajectory,
    compute_ptm_batch_from_pipeline,
    compute_ptm_rmsd_by_type_batch,
    create_cna_pipelines_from_file,
    create_restricted_ptm_rmsd_pipelines,
    create_ptm_pipeline,
    parse_ptm_rmsd_type_names,
    write_denoised_lammpstrj,
)
import argparse
from tqdm import tqdm


def pick_particles(N_pick, Natoms, Config, Box, Min_distance=5):
    """
    Randomly select N_pick unique particle indices from a configuration, ensuring a minimum pairwise distance between all selected particles.

    Parameters
    ----------
    N_pick : int
        Number of particles to pick.
    Natoms : int
        Total number of atoms in the configuration.
    Config : np.ndarray, shape (Natoms, 3)
        Cartesian coordinates for the current configuration (already scaled by box).
    Box : array-like
        Box information for periodic boundaries (passed to pair_distance).
    Min_distance : float, optional
        Minimum allowed distance between picked particles. Default is 5.

    Returns
    -------
    picked : np.ndarray, shape (N_pick,)
        Array of picked particle indices (0-based).

    Notes
    -----
    - The function samples indices without replacement, enforcing the minimum distance constraint.
    - If N_pick is too large for the given Min_distance and system size, the function may loop for a long time.
    - Indices returned are 0-based and suitable for direct NumPy indexing.
    """
    # pick first particle id uniformly at random in the range [0, Natoms-1]
    id = np.random.randint(0, Natoms)
    logging.debug(f"first id {id}")
    # initiate list of picked particle ids
    picked = [id]

    # keep sampling candidates until we have N_pick unique valid ids
    while len(picked) < N_pick:
        # propose a new candidate id at random
        id_candidate = np.random.randint(0, Natoms)
        logging.debug(f"new candidate id {id_candidate}")
        # initialize distance with a large value
        dist_ij = 1000
        # skip candidates already picked
        if id_candidate not in picked:
            # check distance to every already picked particle
            for id_check in picked:
                # compute pairwise distance accounting for box (periodic boundaries)
                dist_ij = pair_distance(id_check, id_candidate, Config, Box)
                # if too close to any picked particle, reject candidate immediately
                if dist_ij < Min_distance:
                    logging.debug(
                        f"dist between {id_candidate} and {id_check}: {dist_ij} : too small"
                    )
                    break
            # if the loop did not break (candidate was far enough from all),
            # and if id_check equals the last element of picked; append candidate
            if id_check == picked[-1]:
                picked.append(id_candidate)
                logging.debug(f"picked so far: {picked}")

    picked = np.array(picked)
    return picked


def count_lammpstrj_frames(filename):
    """Count frames without loading coordinates into memory."""
    n_frames = 0
    with open(filename, "r") as traj_file:
        for line in traj_file:
            if line.startswith("ITEM: TIMESTEP"):
                n_frames += 1
    return n_frames


def iter_lammpstrj_frames(filename):
    """Yield fractional coordinates and box lengths from a LAMMPS atom dump."""
    with open(filename, "r") as traj_file:
        while True:
            line = traj_file.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                raise ValueError(f"Expected 'ITEM: TIMESTEP', got: {line.strip()}")

            traj_file.readline()  # timestep

            line = traj_file.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Expected 'ITEM: NUMBER OF ATOMS', got: {line.strip()}")
            natoms = int(traj_file.readline())

            line = traj_file.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Expected 'ITEM: BOX BOUNDS', got: {line.strip()}")

            box = np.empty(3, dtype=np.float32)
            for dim in range(3):
                lo, hi = np.array(traj_file.readline().split()[:2], dtype=np.float32)
                box[dim] = hi - lo

            atom_header = traj_file.readline().split()
            try:
                x_col = atom_header.index("xs") - 2
                y_col = atom_header.index("ys") - 2
                z_col = atom_header.index("zs") - 2
            except ValueError:
                x_col, y_col, z_col = 2, 3, 4

            config = np.empty((natoms, 3), dtype=np.float32)
            for atom_idx in range(natoms):
                fields = traj_file.readline().split()
                config[atom_idx, 0] = float(fields[x_col])
                config[atom_idx, 1] = float(fields[y_col])
                config[atom_idx, 2] = float(fields[z_col])

            yield config, box


def iter_lammpstrj_batches(filename, batch_size):
    """Yield trajectory batches as (start_frame, Config, Box, Natoms)."""
    configs = []
    boxes = []
    batch_start = 0
    natoms = None

    for frame_index, (config, box) in enumerate(iter_lammpstrj_frames(filename)):
        if not configs:
            batch_start = frame_index
            natoms = config.shape[0]

        configs.append(config)
        boxes.append(box)

        if len(configs) == batch_size:
            yield (
                batch_start,
                np.stack(configs, axis=0),
                np.stack(boxes, axis=0),
                natoms,
            )
            configs = []
            boxes = []

    if configs:
        yield (
            batch_start,
            np.stack(configs, axis=0),
            np.stack(boxes, axis=0),
            natoms,
        )


class DescriptorBatchComputer:
    """
    Reuse expensive descriptor setup while keeping the main loop batch-oriented.
    """

    def __init__(
        self,
        traj_path,
        nn,
        descriptor_nn,
        cna_fixed_cutoff,
        ptm_rmsd_type_names,
        denoised_traj_path=None,
    ):
        self.nn = nn
        self.descriptor_nn = descriptor_nn
        self.cna_fixed_cutoff = cna_fixed_cutoff
        self.ql_order = freud.order.Steinhardt(
            l=list(range(1, 21)),
            average=True,
            wl=False,
        )
        self.ql_no_average_order = freud.order.Steinhardt(
            l=list(range(1, 21)),
            average=False,
            wl=False,
        )
        self.w4w6_order = freud.order.Steinhardt(
            l=[4, 6],
            average=True,
            wl=True,
        )
        self.w4w6_no_average_order = freud.order.Steinhardt(
            l=[4, 6],
            average=False,
            wl=True,
        )
        self.ptm_pipeline, self.ptm_columns = create_ptm_pipeline(filename=str(traj_path))
        self.ptm_rmsd_pipelines, self.ptm_rmsd_type_names = (
            create_restricted_ptm_rmsd_pipelines(
                filename=str(traj_path),
                type_names=ptm_rmsd_type_names,
                rmsd_cutoff=0.0,
            )
        )
        self.cna_columns = ["cna_fixed", "cna_adaptive", "cna_interval"]
        self.n_frames = self.ptm_pipeline.num_frames
        self.denoised_enabled = denoised_traj_path is not None

        if self.denoised_enabled:
            self.ptm_denoised_pipeline, self.ptm_denoised_columns = create_ptm_pipeline(
                filename=str(denoised_traj_path)
            )
            self.ptm_denoised_rmsd_pipelines, self.ptm_denoised_rmsd_type_names = (
                create_restricted_ptm_rmsd_pipelines(
                    filename=str(denoised_traj_path),
                    type_names=ptm_rmsd_type_names,
                    rmsd_cutoff=0.0,
                )
            )
            self.cna_denoised_pipelines, self.cna_denoised_columns = (
                create_cna_pipelines_from_file(
                    filename=str(denoised_traj_path),
                    fixed_cutoff=self.cna_fixed_cutoff,
                    include_fixed=True,
                    include_adaptive=True,
                    include_interval=True,
                )
            )
            if self.ptm_denoised_pipeline.num_frames != self.n_frames:
                raise ValueError(
                    "Denoised trajectory frame count does not match raw trajectory "
                    f"({self.ptm_denoised_pipeline.num_frames} != {self.n_frames})."
                )
        else:
            self.ptm_denoised_columns = self.ptm_columns
            self.ptm_denoised_rmsd_type_names = self.ptm_rmsd_type_names
            self.cna_denoised_columns = self.cna_columns

    def compute(self, batch_start, Config, Box):
        batch_len = len(Config)
        natoms = Config.shape[1]

        ql = np.empty((batch_len, natoms, 20), dtype=np.float32)
        ql_no_average = np.empty((batch_len, natoms, 20), dtype=np.float32)
        w4w6 = np.empty((batch_len, natoms, 2), dtype=np.float32)
        w4w6_no_average = np.empty((batch_len, natoms, 2), dtype=np.float32)

        for local_t in range(batch_len):
            positions = np.asarray(Config[local_t], dtype=np.float32)
            box_lengths = np.asarray(Box[local_t], dtype=np.float32)
            positions = positions * box_lengths
            freud_box = freud.box.Box(*box_lengths)

            self.ql_order.compute(
                system=(freud_box, positions),
                neighbors={"num_neighbors": self.descriptor_nn},
            )
            ql[local_t] = self.ql_order.particle_order

            self.ql_no_average_order.compute(
                system=(freud_box, positions),
                neighbors={"num_neighbors": self.descriptor_nn},
            )
            ql_no_average[local_t] = self.ql_no_average_order.particle_order

            self.w4w6_order.compute(
                system=(freud_box, positions),
                neighbors={"num_neighbors": self.descriptor_nn},
            )
            w4w6[local_t] = self.w4w6_order.particle_order

            self.w4w6_no_average_order.compute(
                system=(freud_box, positions),
                neighbors={"num_neighbors": self.descriptor_nn},
            )
            w4w6_no_average[local_t] = self.w4w6_no_average_order.particle_order

        ptm = compute_ptm_batch_from_pipeline(
            self.ptm_pipeline,
            t_start=batch_start,
            t_max=batch_len,
        )

        ptm_rmsd_by_type = compute_ptm_rmsd_by_type_batch(
            self.ptm_rmsd_pipelines,
            t_start=batch_start,
            t_max=batch_len,
        )

        cna, cna_columns = compute_cna_trajectory(
            Config,
            Box,
            Natoms=natoms,
            t_max=None,
            fractional=True,
            fixed_cutoff=self.cna_fixed_cutoff,
            include_fixed=True,
            include_adaptive=True,
            include_interval=True,
            return_columns=True,
        )
        self.cna_columns = cna_columns

        ptm_denoised = None
        ptm_denoised_rmsd_by_type = None
        cna_denoised = None
        if self.denoised_enabled:
            ptm_denoised = compute_ptm_batch_from_pipeline(
                self.ptm_denoised_pipeline,
                t_start=batch_start,
                t_max=batch_len,
            )
            ptm_denoised_rmsd_by_type = compute_ptm_rmsd_by_type_batch(
                self.ptm_denoised_rmsd_pipelines,
                t_start=batch_start,
                t_max=batch_len,
            )
            cna_denoised = compute_cna_batch_from_pipelines(
                self.cna_denoised_pipelines,
                t_start=batch_start,
                t_max=batch_len,
            )

        return (
            ql,
            ql_no_average,
            w4w6,
            w4w6_no_average,
            ptm,
            ptm_rmsd_by_type,
            cna,
            ptm_denoised,
            ptm_denoised_rmsd_by_type,
            cna_denoised,
        )


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Read number of particles to pick from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-npick")
    parser.add_argument("-nn", default=12)
    parser.add_argument("-descriptor_nn", default=None)
    parser.add_argument("-picked_ids_from", default=None)
    parser.add_argument("-seed", default=None)
    parser.add_argument("-f", default="traj.lammpstrj")
    parser.add_argument("-o", default="particle_data.npz")
    parser.add_argument("-batch", default=5)
    parser.add_argument("-cna_cutoff", default=1.5)
    parser.add_argument("-ptm_rmsd_types", default="fcc,hcp,bcc,ico")
    parser.add_argument("-denoise", action="store_true")
    parser.add_argument("-denoise_structure", default="FCC")
    parser.add_argument("-denoise_steps", default=8)
    parser.add_argument("-denoise_device", default="cpu")
    parser.add_argument("-denoise_scale", default=None)
    parser.add_argument("-denoise_model_path", default=None)
    parser.add_argument("-denoise_tmp", default=None)
    parser.add_argument("-denoised_traj", default=None)
    parser.add_argument("-keep_denoised_traj", action="store_true")

    args = parser.parse_args()
    # number of particles to pick per frame
    N_pick = int(args.npick)
    nn = int(args.nn)
    descriptor_nn = nn if args.descriptor_nn is None else int(args.descriptor_nn)
    if args.seed is not None:
        np.random.seed(int(args.seed))
    # nb this is 5 diameters distance (i.e LJ units)
    Min_distance = 5
    batch_size = int(args.batch)
    cna_fixed_cutoff = float(args.cna_cutoff)
    ptm_rmsd_type_names = parse_ptm_rmsd_type_names(args.ptm_rmsd_types)

    traj_path = Path(args.f)
    denoised_traj_path = None
    if args.denoised_traj is not None:
        denoised_traj_path = Path(args.denoised_traj)
        logging.info("Using existing denoised trajectory %s.", denoised_traj_path)
    elif args.denoise:
        if args.denoise_tmp is None:
            denoised_traj_path = traj_path.with_name(
                f"{traj_path.stem}_denoised_{args.denoise_structure.lower()}.lammpstrj"
            )
        else:
            denoised_traj_path = Path(args.denoise_tmp)
        logging.info(
            "Writing denoised trajectory to %s using structure=%s, steps=%s, device=%s.",
            denoised_traj_path,
            args.denoise_structure,
            args.denoise_steps,
            args.denoise_device,
        )
        write_denoised_lammpstrj(
            input_filename=traj_path,
            output_filename=denoised_traj_path,
            structure=args.denoise_structure,
            steps=int(args.denoise_steps),
            device=args.denoise_device,
            scale=None if args.denoise_scale is None else float(args.denoise_scale),
            model_path=args.denoise_model_path,
        )

    descriptors = DescriptorBatchComputer(
        traj_path,
        nn,
        descriptor_nn,
        cna_fixed_cutoff,
        ptm_rmsd_type_names,
        denoised_traj_path=denoised_traj_path,
    )
    Tmax = descriptors.n_frames
    if Tmax == 0:
        raise ValueError(f"No frames found in {traj_path}")
    logging.info(f"Found {Tmax} frames in {traj_path}.")

    n_samples = Tmax * N_pick
    picked_ids_source = None
    picked_frame_indices_source = None
    if args.picked_ids_from is not None:
        picked_ids_path = Path(args.picked_ids_from)
        logging.info("Reusing picked particle ids from %s.", picked_ids_path)
        with np.load(picked_ids_path, allow_pickle=True) as picked_data:
            picked_source_n_pick = int(np.asarray(picked_data["n_pick"]).item())
            if picked_source_n_pick != N_pick:
                raise ValueError(
                    f"picked_ids_from has n_pick={picked_source_n_pick}, "
                    f"but this run uses npick={N_pick}."
                )
            picked_ids_source = picked_data["picked_ids"].astype(np.int32)
            picked_frame_indices_source = picked_data["frame_indices"].astype(np.int32)
        if picked_ids_source.shape[0] != n_samples:
            raise ValueError(
                "picked_ids_from sample count does not match this trajectory and npick "
                f"({picked_ids_source.shape[0]} != {n_samples})."
            )

    all_dist_picked = np.empty((n_samples, nn), dtype=np.float32)
    all_vec_dist_picked = np.empty((n_samples, nn, 3), dtype=np.float32)
    all_w4w6_picked = np.empty((n_samples, 2), dtype=np.float32)
    all_w4w6_no_average_picked = np.empty((n_samples, 2), dtype=np.float32)
    all_ql_picked = np.empty((n_samples, 20), dtype=np.float32)
    all_ql_no_average_picked = np.empty((n_samples, 20), dtype=np.float32)
    all_ptm_picked = np.empty((n_samples, len(descriptors.ptm_columns)), dtype=np.float32)
    all_picked_ids = np.empty(n_samples, dtype=np.int32)
    all_frame_indices = np.empty(n_samples, dtype=np.int32)
    all_ptm_rmsd_by_type_picked = np.empty(
        (n_samples, len(descriptors.ptm_rmsd_type_names)),
        dtype=np.float32,
    )
    all_cna_picked = np.empty((n_samples, len(descriptors.cna_columns)), dtype=np.int8)
    if descriptors.denoised_enabled:
        all_ptm_denoised_picked = np.empty(
            (n_samples, len(descriptors.ptm_denoised_columns)),
            dtype=np.float32,
        )
        all_ptm_denoised_rmsd_by_type_picked = np.empty(
            (n_samples, len(descriptors.ptm_denoised_rmsd_type_names)),
            dtype=np.float32,
        )
        all_cna_denoised_picked = np.empty(
            (n_samples, len(descriptors.cna_denoised_columns)),
            dtype=np.int8,
        )

    total_batches = (Tmax + batch_size - 1) // batch_size
    batch_iter = iter_lammpstrj_batches(traj_path, batch_size)
    for batch_start, Config, Box, Natoms in tqdm(
        batch_iter,
        total=total_batches,
        desc="Batches",
    ):
        batch_len = len(Config)
        logging.info(
            f"Processing batch starting at frame {batch_start + 1} "
            f"with {batch_len} frames ..."
        )

        (
            ql,
            ql_no_average,
            w4w6,
            w4w6_no_average,
            ptm,
            ptm_rmsd_by_type,
            cna,
            ptm_denoised,
            ptm_denoised_rmsd_by_type,
            cna_denoised,
        ) = descriptors.compute(
            batch_start,
            Config,
            Box,
        )

        for local_istep in range(batch_len):
            istep = batch_start + local_istep
            logging.info(f"  Processing frame {istep + 1}/{Tmax} ...")
            Config_i = Config[local_istep] * Box[local_istep]
            Box_i = Box[local_istep]

            # obtain picked particle ids for this frame
            start = istep * N_pick
            stop = start + N_pick
            if picked_ids_source is None:
                logging.info("    Picking particles ...")
                picked_ids = pick_particles(
                    N_pick,
                    Natoms,
                    Config_i,
                    Box_i,
                    Min_distance=Min_distance,
                )
            else:
                logging.info("    Reusing picked particles ...")
                picked_ids = picked_ids_source[start:stop]
                expected_frames = picked_frame_indices_source[start:stop]
                if not np.all(expected_frames == istep):
                    raise ValueError(
                        "picked_ids_from frame_indices are not grouped as expected "
                        f"for frame {istep}."
                    )
            # obtain nearest neighbor distances and vectors for picked particles only
            logging.info(
                "    Computing nearest neighbor distances and vectors for picked particles ..."
            )
            dist_picked = np.empty((N_pick, nn), dtype=np.float32)
            vec_dist_picked = np.empty((N_pick, nn, 3), dtype=np.float32)
            for local_idx, pid in enumerate(picked_ids):
                vec = get_nn_vector_one(pid, Config_i, Box_i, nn)
                vec_dist_picked[local_idx] = vec
                dist_picked[local_idx] = np.sqrt(np.sum(vec * vec, axis=1))

            all_dist_picked[start:stop] = dist_picked
            all_vec_dist_picked[start:stop] = vec_dist_picked
            all_w4w6_picked[start:stop] = w4w6[local_istep, picked_ids, :]
            all_w4w6_no_average_picked[start:stop] = w4w6_no_average[
                local_istep,
                picked_ids,
                :,
            ]
            all_ql_picked[start:stop] = ql[local_istep, picked_ids, :]
            all_ql_no_average_picked[start:stop] = ql_no_average[local_istep, picked_ids, :]
            all_ptm_picked[start:stop] = ptm[local_istep, picked_ids, :]
            all_picked_ids[start:stop] = picked_ids
            all_frame_indices[start:stop] = istep
            all_ptm_rmsd_by_type_picked[start:stop] = ptm_rmsd_by_type[
                local_istep,
                picked_ids,
                :,
            ]
            all_cna_picked[start:stop] = cna[local_istep, picked_ids, :]
            if descriptors.denoised_enabled:
                all_ptm_denoised_picked[start:stop] = ptm_denoised[
                    local_istep,
                    picked_ids,
                    :,
                ]
                all_ptm_denoised_rmsd_by_type_picked[start:stop] = (
                    ptm_denoised_rmsd_by_type[
                        local_istep,
                        picked_ids,
                        :,
                    ]
                )
                all_cna_denoised_picked[start:stop] = cna_denoised[
                    local_istep,
                    picked_ids,
                    :,
                ]

            del Config_i, dist_picked, vec_dist_picked

        del Config, Box, ql, ql_no_average, w4w6, w4w6_no_average, ptm, ptm_rmsd_by_type, cna
        if descriptors.denoised_enabled:
            del ptm_denoised, ptm_denoised_rmsd_by_type, cna_denoised
        gc.collect()

    # Save to disk as compressed npz
    logging.info(f"Saving picked particle data to {args.o} ...")
    save_data = {
        "dist": all_dist_picked,
        "vec_dist": all_vec_dist_picked,
        "w4w6": all_w4w6_picked,
        "w4w6_no_average": all_w4w6_no_average_picked,
        "ql": all_ql_picked,
        "ql_no_average": all_ql_no_average_picked,
        "ptm": all_ptm_picked,
        "picked_ids": all_picked_ids,
        "frame_indices": all_frame_indices,
        "n_pick": np.array(N_pick),
        "nn": np.array(nn),
        "descriptor_nn": np.array(descriptor_nn),
        "ptm_rmsd_by_type": all_ptm_rmsd_by_type_picked,
        "ptm_rmsd_type_names": np.array(descriptors.ptm_rmsd_type_names),
        "cna": all_cna_picked,
        "cna_columns": np.array(descriptors.cna_columns),
    }
    if descriptors.denoised_enabled:
        save_data.update(
            {
                "ptm_denoised": all_ptm_denoised_picked,
                "ptm_denoised_rmsd_by_type": all_ptm_denoised_rmsd_by_type_picked,
                "ptm_denoised_rmsd_type_names": np.array(
                    descriptors.ptm_denoised_rmsd_type_names
                ),
                "cna_denoised": all_cna_denoised_picked,
                "cna_denoised_columns": np.array(descriptors.cna_denoised_columns),
                "denoise_structure": np.array(args.denoise_structure),
                "denoise_steps": np.array(int(args.denoise_steps)),
            }
        )
    np.savez_compressed(args.o, **save_data)
    if args.denoise and args.denoised_traj is None and not args.keep_denoised_traj:
        denoised_traj_path.unlink(missing_ok=True)
    logging.info("Done.")
