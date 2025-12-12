import numpy as np
import gzip
from pathlib import Path
import numba
from scipy.spatial.distance import squareform
import freud
from typing import Optional
import numpy as np


def read_traj(t):
    Nskip = 9

    Config = []
    Box = []
    frame_nr_old = -1
    mfile = Path(t)
    Natoms = 0
    if mfile.is_file():
        try:
            with open(t, "r") as traj_file:
                for i, line in enumerate(traj_file):
                    modulo = i % (Nskip + Natoms)
                    frame_nr = i // (Nskip + Natoms)
                    if frame_nr != frame_nr_old:
                        Config.append([])
                        Box.append([])

                    if modulo == 3:
                        Natoms = np.array(line.split()).astype(float)[0]

                    if modulo == 5:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        Lx = Lend - Lstart
                        Box[-1].extend([Lx])

                    if modulo == 6:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        Ly = Lend - Lstart
                        Box[-1].extend([Ly])

                    if modulo == 7:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        Lz = Lend - Lstart
                        Box[-1].extend([Lz])

                    if modulo >= Nskip:
                        whole_line = np.array(line.split()).astype(float)
                        x = whole_line[2]
                        y = whole_line[3]
                        z = whole_line[4]

                        Config[-1].append(np.array([x, y, z]))

                    frame_nr_old = frame_nr

        except (EOFError, IndexError, ValueError) as er:
            print("Caught error in {}:".format(t), er)

        if Config:
            if len(Config[-1]) != Natoms:
                del Config[-1]
                del Box[-1]

    Config = np.array(Config)
    return Natoms, Config, Box


def read_mag2patch(t):
    Nskip = 9

    Config = []
    Box = []
    frame_nr_old = -1
    mfile = Path(t)
    Natoms = 0
    if mfile.is_file():
        try:
            with open(t, "r") as traj_file:
                for i, line in enumerate(traj_file):
                    modulo = i % (Nskip + Natoms)
                    frame_nr = i // (Nskip + Natoms)
                    if frame_nr != frame_nr_old:
                        Config.append([])
                        Box.append([])

                    if modulo == 3:
                        Natoms = np.array(line.split()).astype(float)[0]

                    if modulo == 5:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        Lx = Lend - Lstart
                        Box[-1].extend([Lx])

                    if modulo == 6:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        Ly = Lend - Lstart
                        Box[-1].extend([Ly])

                    if modulo == 7:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        Lz = Lend - Lstart
                        Box[-1].extend([Lz])

                    if modulo >= Nskip:
                        if whole_line[1] == 1:
                            whole_line = np.array(line.split()).astype(float)
                            x = whole_line[2]
                            y = whole_line[3]
                            z = whole_line[4]

                            Config[-1].append(np.array([x, y, z]))

                    frame_nr_old = frame_nr

        except (EOFError, IndexError, ValueError) as er:
            print("Caught error in {}:".format(t), er)

        if Config:
            if len(Config[-1]) != Natoms:
                del Config[-1]
                del Box[-1]

    Config = np.array(Config)
    return Natoms, Config, Box


def read_bop(t, Natoms):
    Nskip = 9
    BOP = []
    frame_nr_old = -1
    mfile = Path(t)
    if mfile.is_file():
        with open(t, "r") as traj_file:
            try:
                for i, line in enumerate(traj_file):
                    modulo = i % (Nskip + Natoms)
                    frame_nr = i // (Nskip + Natoms)
                    if frame_nr != frame_nr_old:
                        BOP.append([])

                    if modulo >= Nskip:
                        whole_line = np.array(line.split()).astype(float)
                        BOP[-1].append(np.array(whole_line[1:]))

                    frame_nr_old = frame_nr
            except EOFError as er:
                print(er)

        if BOP:
            if len(BOP[-1]) != Natoms:
                del BOP[-1]

    BOP = np.array(BOP)
    return BOP


@numba.njit(fastmath=True, parallel=False)
def distances(frame_i, Box):
    lx_box = Box[0]
    ly_box = Box[1]
    lz_box = Box[2]

    dist_norm = []
    for i, ipos in enumerate(frame_i):
        for j, jpos in enumerate(frame_i):
            if j > i:
                dist = ipos - jpos

                dx = dist[0]
                dy = dist[1]
                dz = dist[2]

                sign_dx = np.sign(dx)
                sign_dy = np.sign(dy)
                sign_dz = np.sign(dz)

                # pbc only for x and y
                dx = sign_dx * (min(np.fabs(dx), lx_box - np.fabs(dx)))
                dy = sign_dy * (min(np.fabs(dy), ly_box - np.fabs(dy)))
                dz = sign_dz * (min(np.fabs(dz), lz_box - np.fabs(dz)))

                dist_ij = np.sqrt(dx * dx + dy * dy + dz * dz)
                dist_norm.append(dist_ij)

    return dist_norm


def pair_distance(id1, id2, frame_i, Box):
    lx_box = Box[0]
    ly_box = Box[1]
    lz_box = Box[2]

    ipos = frame_i[id1]
    jpos = frame_i[id2]

    dist = ipos - jpos

    dx = dist[0]
    dy = dist[1]
    dz = dist[2]

    sign_dx = np.sign(dx)
    sign_dy = np.sign(dy)
    sign_dz = np.sign(dz)

    # pbc
    dx = sign_dx * (min(np.fabs(dx), lx_box - np.fabs(dx)))
    dy = sign_dy * (min(np.fabs(dy), ly_box - np.fabs(dy)))
    dz = sign_dz * (min(np.fabs(dz), lz_box - np.fabs(dz)))

    dist_ij = np.sqrt(dx * dx + dy * dy + dz * dz)

    return dist_ij


@numba.njit(fastmath=True, parallel=False)
def one_particle_distance(id1, frame_i, Box):
    lx_box = Box[0]
    ly_box = Box[1]
    lz_box = Box[2]

    dist_norm = []
    for i, ipos in enumerate(frame_i):
        if i != id1:
            dist = ipos - frame_i[id1]

            dx = dist[0]
            dy = dist[1]
            dz = dist[2]

            sign_dx = np.sign(dx)
            sign_dy = np.sign(dy)
            sign_dz = np.sign(dz)

            # pbc
            dx = sign_dx * (min(np.fabs(dx), lx_box - np.fabs(dx)))
            dy = sign_dy * (min(np.fabs(dy), ly_box - np.fabs(dy)))
            dz = sign_dz * (min(np.fabs(dz), lz_box - np.fabs(dz)))

            dist_ij = np.sqrt(dx * dx + dy * dy + dz * dz)
            dist_norm.append(dist_ij)

    return dist_norm


def nextN_neighbours_per_id(id1, nn, frame_i, Box):
    dist = one_particle_distance(id1, frame_i, Box)
    NextN = np.sort(dist)[:nn]
    return NextN


@numba.njit(
    fastmath=True, parallel=False
)  # I would not recommend it this way, because using lists doesnt work with numba, I could change it to have different shapes, but a buffer approach is probably faster anyway
def vector_squareform_distances(frame_i, Box):
    lx_box = Box[0]
    ly_box = Box[1]
    lz_box = Box[2]

    dist_norm = []
    vdist = []
    for i, ipos in enumerate(frame_i):
        for j, jpos in enumerate(frame_i):
            dist = ipos - jpos

            dx = dist[0]
            dy = dist[1]
            dz = dist[2]

            sign_dx = np.sign(dx)
            sign_dy = np.sign(dy)
            sign_dy = np.sign(dz)

            # pbc only for x and y
            dx = sign_dx * (min(np.fabs(dx), lx_box - np.fabs(dx)))
            dy = sign_dy * (min(np.fabs(dy), ly_box - np.fabs(dy)))
            dz = sign_dy * (min(np.fabs(dz), lz_box - np.fabs(dz)))

            dist_ij = np.sqrt(dx * dx + dy * dy + dz * dz)
            dist_norm.append(dist_ij)
            vdist.append([dx, dy, dz])

    return dist_norm, vdist


def neighbours(sq_dist, cutoff):
    b = np.where((sq_dist < cutoff) & (sq_dist > 0.01))
    neighbour_list = [[b[0][i], b[1][i]] for i in range(len(b[0]))]
    return neighbour_list


def nextN_neighbours(Natoms, sq_dist, nn):
    NextN = np.zeros((int(Natoms), int(nn)))
    for i in range(int(Natoms)):
        NextN[i] = np.sort(sq_dist[i])[1 : (nn + 1)]

    return NextN


# When creating an array of fixed size, the numba acceleration can take full effect. (Instead of lists)
# Furthermore, not having to save all distances from every particle to every other particles saves a lot of time -> buffers


@numba.njit(fastmath=True, parallel=False)
def _get_nn_vector(frame_i, box, nn):
    """
    Numba-accelerated kernel that is intended to be numerically compatible
    with your original `vector_squareform_distances` + `nextN_neighbours_vector`.

    Parameters
    ----------
    frame_i : (Natoms, 3)
        Cartesian positions (already Config[ti] * Box[ti]).
    box : (3,)
        Box lengths [Lx, Ly, Lz].
    nn : int
        Number of nearest neighbours.

    Returns
    -------
    nextN_flat : (Natoms, nn*3)
        For each atom i, displacement vectors to its nn nearest neighbors,
        flattened [dx1, dy1, dz1, dx2, dy2, dz2, ...].
    """
    Natoms = frame_i.shape[0]
    lx_box = box[0]
    ly_box = box[1]
    lz_box = box[2]

    nextN_flat = np.empty((Natoms, nn * 3))

    # temporary buffers (per atom)
    dist_row = np.empty(Natoms)
    vec_row = np.empty((Natoms, 3))

    big = 1.0e30

    for i in range(Natoms):
        xi = frame_i[i, 0]
        yi = frame_i[i, 1]
        zi = frame_i[i, 2]

        # compute distances from atom i to all j (including self)
        for j in range(Natoms):
            dx = xi - frame_i[j, 0]
            dy = yi - frame_i[j, 1]
            dz = zi - frame_i[j, 2]

            # ---- PBC like in vector_squareform_distances ----
            # dx
            if dx > 0.0:
                sign_dx = 1.0
            elif dx < 0.0:
                sign_dx = -1.0
            else:
                sign_dx = 0.0
            adx = abs(dx)
            adx_m = lx_box - adx
            if adx_m < adx:
                dx = sign_dx * adx_m
            else:
                dx = sign_dx * adx

            # dy
            if dy > 0.0:
                sign_dy = 1.0
            elif dy < 0.0:
                sign_dy = -1.0
            else:
                sign_dy = 0.0
            ady = abs(dy)
            ady_m = ly_box - ady
            if ady_m < ady:
                dy = sign_dy * ady_m
            else:
                dy = sign_dy * ady

            # dz
            if dz > 0.0:
                sign_dz = 1.0
            elif dz < 0.0:
                sign_dz = -1.0
            else:
                sign_dz = 0.0
            adz = abs(dz)
            adz_m = lz_box - adz
            if adz_m < adz:
                dz = sign_dz * adz_m
            else:
                dz = sign_dz * adz

            dist_ij = np.sqrt(dx * dx + dy * dy + dz * dz)
            dist_row[j] = dist_ij
            vec_row[j, 0] = dx
            vec_row[j, 1] = dy
            vec_row[j, 2] = dz

        # don't pick self as neighbour
        dist_row[i] = big

        # sort neighbors by distance
        idx_sorted = np.argsort(dist_row)

        # take nn nearest and flatten
        for k in range(nn):
            j_idx = idx_sorted[k]
            base = 3 * k
            nextN_flat[i, base] = vec_row[j_idx, 0]
            nextN_flat[i, base + 1] = vec_row[j_idx, 1]
            nextN_flat[i, base + 2] = vec_row[j_idx, 2]

    return nextN_flat


@numba.njit(fastmath=True, parallel=False)
def _get_nn_dist(frame_i, box, nn):
    """
    Numba-accelerated kernel that is intended to be numerically compatible
    with your original `vector_squareform_distances` + `nextN_neighbours_vector`.

    Parameters
    ----------
    frame_i : (Natoms, 3)
        Cartesian positions (already Config[ti] * Box[ti]).
    box : (3,)
        Box lengths [Lx, Ly, Lz].
    nn : int
        Number of nearest neighbours.

    Returns
    -------
    nextN_dist : (Natoms, nn)
        For each atom i, distance to its nn nearest neighbors,
        flattened [d_1,d_2,d_3, ...].
    """
    Natoms = frame_i.shape[0]
    lx_box = box[0]
    ly_box = box[1]
    lz_box = box[2]

    nextN_dist = np.empty((Natoms, nn))

    # temporary buffers (per atom)
    dist_row = np.empty(Natoms)

    big = 1.0e30

    for i in range(Natoms):
        xi = frame_i[i, 0]
        yi = frame_i[i, 1]
        zi = frame_i[i, 2]

        # compute distances from atom i to all j (including self)
        for j in range(Natoms):
            dx = xi - frame_i[j, 0]
            dy = yi - frame_i[j, 1]
            dz = zi - frame_i[j, 2]

            # ---- PBC like in vector_squareform_distances ----
            # dx
            if dx > 0.0:
                sign_dx = 1.0
            elif dx < 0.0:
                sign_dx = -1.0
            else:
                sign_dx = 0.0
            adx = abs(dx)
            adx_m = lx_box - adx
            if adx_m < adx:
                dx = sign_dx * adx_m
            else:
                dx = sign_dx * adx

            # dy
            if dy > 0.0:
                sign_dy = 1.0
            elif dy < 0.0:
                sign_dy = -1.0
            else:
                sign_dy = 0.0
            ady = abs(dy)
            ady_m = ly_box - ady
            if ady_m < ady:
                dy = sign_dy * ady_m
            else:
                dy = sign_dy * ady

            # dz
            if dz > 0.0:
                sign_dz = 1.0
            elif dz < 0.0:
                sign_dz = -1.0
            else:
                sign_dz = 0.0
            adz = abs(dz)
            adz_m = lz_box - adz
            if adz_m < adz:
                dz = sign_dz * adz_m
            else:
                dz = sign_dz * adz

            dist_ij = np.sqrt(dx * dx + dy * dy + dz * dz)
            dist_row[j] = dist_ij

        # don't pick self as neighbour
        dist_row[i] = big

        # sort neighbors by distance
        idx_sorted = np.argsort(dist_row)

        # take nn nearest and flatten
        for k in range(nn):
            j_idx = idx_sorted[k]
            nextN_dist[i, k] = dist_row[j_idx]
    return nextN_dist


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
        average=True,  # Lechner–Dellago averaged OP
        wl=True,  # compute w_l (not q_l)
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

        w_traj_all[t, :, :] = steinhardt.particle_order  # shape (Natoms, 2): [w_4, w_6]

    return w_traj_all  # shape (t_max,Natoms,2)


@numba.njit(fastmath=True, parallel=False)
def get_nn_dist_one(id1, frame_i, box, nn):
    """
    Compute distances from particle id1 to all others and return the nn nearest neighbor distances.

    Parameters
    ----------
    id1 : int
        Index of the reference particle.
    frame_i : (Natoms, 3)
        Cartesian positions.
    box : (3,)
        Box lengths [Lx, Ly, Lz].
    nn : int
        Number of nearest neighbors.

    Returns
    -------
    nn_dist : (nn,)
        Distances to the nn nearest neighbors.
    """
    Natoms = frame_i.shape[0]
    lx_box, ly_box, lz_box = box
    dist_row = np.empty(Natoms)
    for j in range(Natoms):
        if j == id1:
            dist_row[j] = 1e30  # exclude self
            continue
        dx = frame_i[id1, 0] - frame_i[j, 0]
        dy = frame_i[id1, 1] - frame_i[j, 1]
        dz = frame_i[id1, 2] - frame_i[j, 2]
        # PBC
        dx = np.sign(dx) * min(np.abs(dx), lx_box - np.abs(dx))
        dy = np.sign(dy) * min(np.abs(dy), ly_box - np.abs(dy))
        dz = np.sign(dz) * min(np.abs(dz), lz_box - np.abs(dz))
        dist_row[j] = np.sqrt(dx * dx + dy * dy + dz * dz)
    idx_sorted = np.argsort(dist_row)
    return dist_row[idx_sorted[:nn]]


@numba.njit(fastmath=True, parallel=False)
def get_nn_vector_one(id1, frame_i, box, nn):
    """
    Compute displacement vectors from particle id1 to its nn nearest neighbors.

    Parameters
    ----------
    id1 : int
        Index of the reference particle.
    frame_i : (Natoms, 3)
        Cartesian positions.
    box : (3,)
        Box lengths [Lx, Ly, Lz].
    nn : int
        Number of nearest neighbors.

    Returns
    -------
    nn_vecs : (nn, 3)
        Displacement vectors to the nn nearest neighbors.
    """
    Natoms = frame_i.shape[0]
    lx_box, ly_box, lz_box = box
    dist_row = np.empty(Natoms)
    vec_row = np.empty((Natoms, 3))
    for j in range(Natoms):
        if j == id1:
            dist_row[j] = 1e30
            vec_row[j, :] = 0.0
            continue
        dx = frame_i[id1, 0] - frame_i[j, 0]
        dy = frame_i[id1, 1] - frame_i[j, 1]
        dz = frame_i[id1, 2] - frame_i[j, 2]
        # PBC
        dx = np.sign(dx) * min(np.abs(dx), lx_box - np.abs(dx))
        dy = np.sign(dy) * min(np.abs(dy), ly_box - np.abs(dy))
        dz = np.sign(dz) * min(np.abs(dz), lz_box - np.abs(dz))
        dist_row[j] = np.sqrt(dx * dx + dy * dy + dz * dz)
        vec_row[j, 0] = dx
        vec_row[j, 1] = dy
        vec_row[j, 2] = dz
    idx_sorted = np.argsort(dist_row)
    return vec_row[idx_sorted[:nn], :]
