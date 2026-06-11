from typing import Optional
import numpy as np

from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier


def compute_ptm_from_lammpstrj(
    filename: str,
    t_max: Optional[int] = None,
    rmsd_cutoff: float = 0.1,
    include_deformation_gradient: bool = False,
    return_columns: bool = False,
):
    """
    Compute PTM descriptors directly from a LAMMPS trajectory file.

    Default output columns:
        structure_type, rmsd, interatomic_distance, qx, qy, qz, qw
    """
    pipeline = import_file(filename, multiple_frames=True)

    ptm = PolyhedralTemplateMatchingModifier(
        output_rmsd=True,
        output_interatomic_distance=True,
        output_orientation=True,
        output_deformation_gradient=include_deformation_gradient,
        rmsd_cutoff=rmsd_cutoff,
    )
    pipeline.modifiers.append(ptm)

    T_total = pipeline.num_frames
    if t_max is None or t_max > T_total:
        t_max = T_total

    first = pipeline.compute(0)
    Natoms = first.particles.count

    columns = [
        "structure_type",
        "rmsd",
        "interatomic_distance",
        "qx",
        "qy",
        "qz",
        "qw",
    ]

    if include_deformation_gradient:
        columns += [
            "Fxx", "Fxy", "Fxz",
            "Fyx", "Fyy", "Fyz",
            "Fzx", "Fzy", "Fzz",
        ]

    ptm_traj_all = np.empty((t_max, Natoms, len(columns)), dtype=np.float32)

    for t in range(t_max):
        data = pipeline.compute(t)
        particles = data.particles

        structure_type = np.asarray(particles["Structure Type"], dtype=np.float32)
        rmsd = np.asarray(particles["RMSD"], dtype=np.float32)
        interatomic_distance = np.asarray(
            particles["Interatomic Distance"],
            dtype=np.float32,
        )
        orientation = np.asarray(particles["Orientation"], dtype=np.float32)

        out_cols = [
            structure_type[:, None],
            rmsd[:, None],
            interatomic_distance[:, None],
            orientation,
        ]

        if include_deformation_gradient:
            F = np.asarray(
                particles["Elastic Deformation Gradient"],
                dtype=np.float32,
            ).reshape(Natoms, 9)
            out_cols.append(F)

        ptm_traj_all[t, :, :] = np.concatenate(out_cols, axis=1)

    if return_columns:
        return ptm_traj_all, columns

    return ptm_traj_all