import keras
import numpy as np
import pandas as pd
import logging
from lammpstools.tools import (
    distances,
    neighbours,
    read_traj,
    pair_distance,
    nextN_neighbours_per_id,
    _compute_w4_w6_trajectory,
    _compute_ql_trajectory,
    get_nn_vector_one,
    get_nn_dist_one,
)
from scipy.spatial.distance import squareform
import argparse
from tqdm import tqdm


def write_pdb_beta(
    filename,
    pos,
    beta,
    box=None,
    ids=None,
    types=None,
    model_index=None,
    wrap_into_box=False,
):
    """
    Write a PDB where tempFactor (B-factor) = beta per atom.
    If model_index is not None, writes MODEL/ENDMDL wrappers (multi-model PDB).

    pos: (N,3)
    beta: (N,)
    box: (3,) lengths or
    ids: (N,) optional, used for serial (else 1..N)
    types: (N,) optional, used for element heuristic
    wrap_into_box: if True and box bounds provided, wraps positions into [lo,hi)
    """
    pos = np.asarray(pos, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must be (N,3), got {pos.shape}")
    if beta.shape[0] != pos.shape[0]:
        raise ValueError("beta must have same length as pos")

    N = pos.shape[0]
    if ids is None:
        ids = np.arange(1, N + 1, dtype=int)
    else:
        ids = np.asarray(ids, dtype=int)
    if types is None:
        types = np.ones(N, dtype=int)
    else:
        types = np.asarray(types, dtype=int)

    # Element guess from type (customize as you like)
    type_to_elem = {1: "C", 2: "O", 3: "N", 4: "S"}

    with open(filename, "a" if model_index is not None else "w") as f:
        if model_index is not None:
            f.write(f"MODEL     {int(model_index):4d}\n")

        if box is not None:
            Lx, Ly, Lz = box

        for i in range(N):
            elem = type_to_elem.get(int(types[i]), "C")
            serial = int(ids[i]) % 100000  # PDB serial is 5 digits
            x, y, z = pos[i]
            bf = float(beta[i])

            # tempFactor (B-factor) is columns 61-66 in classic PDB formatting
            f.write(
                "ATOM  {serial:5d} {name:<4s} {res:>3s} {chain:1s}{resseq:4d}    "
                "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bf:6.7f}          {elem:>2s}\n".format(
                    serial=serial,
                    name=elem,
                    res="LJ",
                    chain="A",
                    resseq=1,
                    x=x,
                    y=y,
                    z=z,
                    occ=1.00,
                    bf=bf,
                    elem=elem,
                )
            )

        if model_index is not None:
            f.write("ENDMDL\n")
        else:
            f.write("END\n")


def write_multimodel_pdb_beta(
    filename,
    frames_pos,
    frames_beta,
    box=None,
    ids=None,
    types=None,
    wrap_into_box=False,
):
    """
    Write multi-frame PDB with MODEL/ENDMDL; beta per frame.
    frames_pos: iterable of (N,3)
    frames_beta: iterable of (N,)
    """
    # start fresh
    open(filename, "w").close()
    for k, (pos, beta) in enumerate(zip(frames_pos, frames_beta), start=1):
        write_pdb_beta(
            filename,
            pos,
            beta,
            box=box,
            ids=ids,
            types=types,
            model_index=k,
            wrap_into_box=wrap_into_box,
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
    parser.add_argument("-nn", default=12)
    parser.add_argument("-f", default="traj.lammpstrj")
    parser.add_argument("-type", default="snapshot")

    args = parser.parse_args()
    nn = int(args.nn)

    # Read trajectory
    logging.info(f"Reading trajectory from {args.f} ...")
    Natoms, Config, Box = read_traj(args.f)
    Tmax = len(Config)
    logging.info(f"Trajectory loaded: {Tmax} frames, {Natoms} atoms per frame.")
    logging.info("First frame coordinates:")
    logging.info(Config[0] if Tmax > 0 else "No frames loaded.")

    # Precompute ql for all particles in all frames
    logging.info("Computing ql for all frames ...")
    ql = _compute_ql_trajectory(Config, Box, Natoms, t_max=None, num_neighbors=nn)

    w4w6 = _compute_w4_w6_trajectory(Config, Box, Natoms, t_max=None, num_neighbors=nn)
    logging.info("ql computation complete.")

    if args.type == "snapshot":
        # Write last frame only
        Lx = Box[-1]
        write_pdb_beta("q6_frame.pdb", Config[-1], ql[0, :, 5], box=[Lx, Lx, Lx])
        write_pdb_beta("q4_frame.pdb", Config[-1], ql[0, :, 3], box=[Lx, Lx, Lx])
        write_pdb_beta("w4_frame.pdb", Config[-1], w4w6[0, :, 0], box=[Lx, Lx, Lx])
        write_pdb_beta("w6_frame.pdb", Config[-1], w4w6[0, :, 1], box=[Lx, Lx, Lx])

    logging.info("PDB writing complete.")

np.savez_compressed(
    "particle_data_seq.npz",
    w4w6=w4w6,
    ql=ql,
)

q6 = ql[:, :, 5]  # shape (T, N)
with open("q6.dat", "w") as f:
    for t in range(q6.shape[0]):
        # 6 decimals is plenty; VMD will read as floats
        f.write(" ".join(f"{val:.6f}" for val in q6[t]) + "\n")

q4 = ql[:, :, 3]  # shape (T, N)
with open("q4.dat", "w") as f:
    for t in range(q4.shape[0]):
        # 6 decimals is plenty; VMD will read as floats
        f.write(" ".join(f"{val:.6f}" for val in q4[t]) + "\n")

w4 = w4w6[:, :, 0]  # shape (T, N)
with open("w4.dat", "w") as f:
    for t in range(w4.shape[0]):
        # 6 decimals is plenty; VMD will read as floats
        f.write(" ".join(f"{val:.6f}" for val in w4[t]) + "\n")

w6 = w4w6[:, :, 1]  # shape (T, N)
with open("w6.dat", "w") as f:
    for t in range(w6.shape[0]):
        # 6 decimals is plenty; VMD will read as floats
        f.write(" ".join(f"{val:.6f}" for val in w6[t]) + "\n")

logging.info("Done.")
