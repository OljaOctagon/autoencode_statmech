
import freud
from typing import  Optional
import numpy as np 


def _compute_w4_w6_trajectory(
    
    Config,
    Box,
    Natoms: int,
    t_max: Optional[int] = None,
    num_neighbors: int = 12,
) -> np.ndarray:
    """
    Compute averaged w_4 and w_6 (Lechner–Dellago) for all frames up to t_max.

    Parameters
    ----------
    Config : array-like, shape (T, Natoms, 3)
        Fractional coordinates per frame.
    Box : array-like, shape (T, 3)
        Box lengths [Lx, Ly, Lz] per frame.
    Natoms : int
        Number of atoms.
    t_max : int or None
        Number of time steps to use (from 0 to t_max-1).
        If None, use all available frames.
    num_neighbors : int
        Number of neighbors for Steinhardt neighbor search.

    Returns
    -------
    w_traj_all : np.ndarray, shape (T_used, Natoms, 2)
        Full per-particle array [w_4, w_6] for each frame.
    """
    T_total = len(Config)
    if t_max is None or t_max > T_total:
        t_max = T_total

    Natoms = int(Natoms)
    l_list = [4, 6]

    # Pre-allocate: (T_used, Natoms, 2)
    w_traj_all = np.empty((t_max, Natoms, len(l_list)), dtype=np.float32)

    # Reuse the same Steinhardt instance for all frames
    steinhardt = freud.order.Steinhardt(
        l=l_list,
        average=True,   # Lechner–Dellago averaged OP
        wl=True,        # compute w_l (not q_l)
    )

    for t in range(t_max):
        # positions: (Natoms, 3), in Cartesian coordinates
        positions = np.asarray(Config[t], dtype=np.float32)
        box_lengths = np.asarray(Box[t], dtype=np.float32)  # [Lx, Ly, Lz]
        positions = positions * box_lengths  # assuming Config is fractional

        # Build box for this frame
        Lx, Ly, Lz = box_lengths
        box = freud.box.Box(Lx, Ly, Lz)

        # Compute averaged w_l for this frame
        steinhardt.compute(
            system=(box, positions),
            neighbors={"num_neighbors": num_neighbors},
        )
        
        w_traj_all[t, :, :] = steinhardt.particle_order # shape (Natoms, 2): [w_4, w_6]

    return w_traj_all #shape (t_max,Natoms,2)