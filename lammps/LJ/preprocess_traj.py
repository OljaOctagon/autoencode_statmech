import numpy as np
import logging
import gc
from pathlib import Path

from lammpstools.tools import (
    pair_distance,
    _compute_w4_w6_trajectory,
    _compute_ql_trajectory,
    get_nn_vector_one,
)
from lammpstools import compute_ptm_from_lammpstrj, get_ptm_columns
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


def compute_descriptor_batch(traj_path, batch_start, Config, Box, Natoms, nn):
    """
    Compute all full-frame descriptors for one batch.

    Keeping this outside the main block preserves the original shape: the main
    script orchestrates batching, while descriptor details stay in helper code.
    """
    logging.info("  Computing ql for batch ...")
    ql = _compute_ql_trajectory(Config, Box, Natoms, t_max=None, num_neighbors=nn)

    logging.info("  Computing w4/w6 for batch ...")
    w4w6 = _compute_w4_w6_trajectory(Config, Box, Natoms, t_max=None, num_neighbors=nn)

    logging.info("  Computing PTM for batch ...")
    ptm = compute_ptm_from_lammpstrj(
        filename=str(traj_path),
        t_start=batch_start,
        t_max=len(Config),
    )

    return ql, w4w6, ptm


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
    parser.add_argument("-f", default="traj.lammpstrj")
    parser.add_argument("-batch", default=5)

    args = parser.parse_args()
    # number of particles to pick per frame
    N_pick = int(args.npick)
    nn = int(args.nn)
    # nb this is 5 diameters distance (i.e LJ units)
    Min_distance = 5
    batch_size = int(args.batch)

    traj_path = Path(args.f)
    Tmax = count_lammpstrj_frames(traj_path)
    if Tmax == 0:
        raise ValueError(f"No frames found in {traj_path}")
    logging.info(f"Found {Tmax} frames in {traj_path}.")

    ptm_columns = get_ptm_columns()

    n_samples = Tmax * N_pick
    all_dist_picked = np.empty((n_samples, nn), dtype=np.float32)
    all_vec_dist_picked = np.empty((n_samples, nn, 3), dtype=np.float32)
    all_w4w6_picked = np.empty((n_samples, 2), dtype=np.float32)
    all_ql_picked = np.empty((n_samples, 20), dtype=np.float32)
    all_ptm_picked = np.empty((n_samples, len(ptm_columns)), dtype=np.float32)

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

        ql, w4w6, ptm = compute_descriptor_batch(
            traj_path,
            batch_start,
            Config,
            Box,
            Natoms,
            nn,
        )

        for local_istep in range(batch_len):
            istep = batch_start + local_istep
            logging.info(f"  Processing frame {istep + 1}/{Tmax} ...")
            Config_i = Config[local_istep] * Box[local_istep]
            Box_i = Box[local_istep]

            # obtain picked particle ids for this frame
            logging.info("    Picking particles ...")
            picked_ids = pick_particles(
                N_pick,
                Natoms,
                Config_i,
                Box_i,
                Min_distance=Min_distance,
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

            start = istep * N_pick
            stop = start + N_pick

            all_dist_picked[start:stop] = dist_picked
            all_vec_dist_picked[start:stop] = vec_dist_picked
            all_w4w6_picked[start:stop] = w4w6[local_istep, picked_ids, :]
            all_ql_picked[start:stop] = ql[local_istep, picked_ids, :]
            all_ptm_picked[start:stop] = ptm[local_istep, picked_ids, :]

            del Config_i, dist_picked, vec_dist_picked

        del Config, Box, ql, w4w6, ptm
        gc.collect()

    # Save to disk as compressed npz
    logging.info("Saving picked particle data to particle_data.npz ...")
    np.savez_compressed(
        "particle_data.npz",
        dist=all_dist_picked,
        vec_dist=all_vec_dist_picked,
        w4w6=all_w4w6_picked,
        ql=all_ql_picked,
        ptm=all_ptm_picked,
    )
    logging.info("Done.")
