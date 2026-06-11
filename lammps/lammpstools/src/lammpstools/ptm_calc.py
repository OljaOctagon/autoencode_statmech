from typing import Optional
import numpy as np

from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier


def compute_ptm_from_lammpstrj(
    filename: str,
    t_start: int = 0,
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
    if t_start < 0 or t_start >= T_total:
        raise ValueError(f"t_start={t_start} is outside trajectory with {T_total} frames")

    T_available = T_total - t_start
    if t_max is None or t_max > T_available:
        t_max = T_available

    first = pipeline.compute(0)
    Natoms = first.particles.count

    columns = get_ptm_columns(include_deformation_gradient)

    ptm_traj_all = np.empty((t_max, Natoms, len(columns)), dtype=np.float32)

    for local_t, global_t in enumerate(range(t_start, t_start + t_max)):
        data = pipeline.compute(global_t)
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

        ptm_traj_all[local_t, :, :] = np.concatenate(out_cols, axis=1)

    if return_columns:
        return ptm_traj_all, columns

    return ptm_traj_all


def get_ptm_columns(include_deformation_gradient: bool = False):
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

    return columns


def create_ptm_pipeline(
    filename: str,
    rmsd_cutoff: float = 0.1,
    include_deformation_gradient: bool = False,
):
    pipeline = import_file(filename, multiple_frames=True)

    ptm = PolyhedralTemplateMatchingModifier(
        output_rmsd=True,
        output_interatomic_distance=True,
        output_orientation=True,
        output_deformation_gradient=include_deformation_gradient,
        rmsd_cutoff=rmsd_cutoff,
    )
    pipeline.modifiers.append(ptm)

    columns = get_ptm_columns(include_deformation_gradient)

    return pipeline, columns


def compute_ptm_frame(
    pipeline,
    frame_index: int,
    include_deformation_gradient: bool = False,
):
    data = pipeline.compute(frame_index)
    particles = data.particles
    natoms = particles.count

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
        ).reshape(natoms, 9)
        out_cols.append(F)

    return np.concatenate(out_cols, axis=1)
