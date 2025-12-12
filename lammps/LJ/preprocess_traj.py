import keras
import numpy as np
import pandas as pd
from lammpstools.tools import (
    distances,
    neighbours,
    read_traj,
    pair_distance,
    nextN_neighbours_per_id,
    _compute_w4_w6_trajectory,
    _get_nn_vector,
    _get_nn_dist,
)
from scipy.spatial.distance import squareform
import argparse


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
    print("first id", id)
    # initiate list of picked particle ids
    picked = [id]

    # keep sampling candidates until we have N_pick unique valid ids
    while len(picked) < N_pick:
        # propose a new candidate id at random
        id_candidate = np.random.randint(0, Natoms)
        print("new candidate id ", id_candidate)
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
                    print(
                        "dist between {} and {}: {} : too small".format(
                            id_candidate, id_check, dist_ij
                        )
                    )
                    break
            # if the loop did not break (candidate was far enough from all),
            # and if id_check equals the last element of picked; append candidate
            if id_check == picked[-1]:
                picked.append(id_candidate)
                print("picked so far: ", picked)

    picked = np.array(picked)
    return picked


if __name__ == "__main__":
    # Read number of particles to pick from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-npick")
    parser.add_argument("-nn", default=12)

    args = parser.parse_args()
    N_pick = int(args.npick)
    nn = int(args.nn)
    # nb this is 5 diameters distance (i.e LJ units)
    Min_distance = 5

    # Read trajectory
    Natoms, Config, Box = read_traj("trajectory.lammpstrj")
    Tmax = len(Config)

    # Precompute w4 and w6 for all particles in all frames
    w4w6 = _compute_w4_w6_trajectory(
        Config,
        Box,
        Natoms,
        t_max=None,
        num_neighbors=12,
    )

    # Prepare lists to collect picked data for all frames
    all_dist_picked = []
    all_vec_dist_picked = []
    all_w4w6_picked = []

    for istep in range(Tmax):
        Config_i = Config[istep] * Box[istep]
        Box_i = Box[istep]

        # obtain picked particle ids for this frame
        picked_ids = pick_particles(
            N_pick,
            Natoms,
            Config_i,
            Box_i,
            Min_distance=Min_distance,
        )
        # obtain nearest neighbor distances and vectors for all particles of this frame
        dist = _get_nn_dist(Config_i, Box_i, nn)
        vec_dist = _get_nn_vector(Config_i, Box_i, nn)

        # gather scalar and vector distances and w4w6 for picked particles
        dist_picked = dist[picked_ids, :]
        vec_dist_picked = vec_dist[picked_ids, :]
        w4w6_picked = w4w6[istep, picked_ids, :]

        all_dist_picked.append(dist_picked)
        all_vec_dist_picked.append(vec_dist_picked)
        all_w4w6_picked.append(w4w6_picked)

    # Convert lists to arrays (shape: [Tmax, N_pick, ...])
    all_dist_picked = np.array(all_dist_picked)
    all_vec_dist_picked = np.array(all_vec_dist_picked)
    all_w4w6_picked = np.array(all_w4w6_picked)

    # Save to disk as compressed npz
    np.savez_compressed(
        "particle_data.npz",
        dist=all_dist_picked,
        vec_dist=all_vec_dist_picked,
        w4w6=all_w4w6_picked,
    )
