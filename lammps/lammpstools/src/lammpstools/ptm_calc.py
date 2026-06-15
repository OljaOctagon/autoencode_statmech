from typing import Optional
import numpy as np

from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier


PTM_RMSD_TYPE_IDS = {
    "fcc": PolyhedralTemplateMatchingModifier.Type.FCC,
    "hcp": PolyhedralTemplateMatchingModifier.Type.HCP,
    "bcc": PolyhedralTemplateMatchingModifier.Type.BCC,
    "ico": PolyhedralTemplateMatchingModifier.Type.ICO,
    "sc": PolyhedralTemplateMatchingModifier.Type.SC,
    "cubic_diamond": PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND,
    "hex_diamond": PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND,
    "graphene": PolyhedralTemplateMatchingModifier.Type.GRAPHENE,
}


def parse_ptm_rmsd_type_names(type_names):
    if type_names is None:
        return []

    if isinstance(type_names, str):
        type_names = [name.strip() for name in type_names.split(",")]

    parsed = []
    for name in type_names:
        name = str(name).strip().lower()
        if not name:
            continue
        if name == "all":
            return list(PTM_RMSD_TYPE_IDS)
        if name not in PTM_RMSD_TYPE_IDS:
            known = ", ".join(list(PTM_RMSD_TYPE_IDS) + ["all"])
            raise ValueError(f"Unknown PTM RMSD type '{name}'. Known values: {known}")
        parsed.append(name)

    return parsed


def _make_ptm_modifier(
    rmsd_cutoff: float = 0.1,
    include_deformation_gradient: bool = False,
    output_interatomic_distance: bool = True,
    output_orientation: bool = True,
):
    return PolyhedralTemplateMatchingModifier(
        output_rmsd=True,
        output_interatomic_distance=output_interatomic_distance,
        output_orientation=output_orientation,
        output_deformation_gradient=include_deformation_gradient,
        rmsd_cutoff=rmsd_cutoff,
    )


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

    ptm = _make_ptm_modifier(
        rmsd_cutoff=rmsd_cutoff,
        include_deformation_gradient=include_deformation_gradient,
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


def compute_ptm_batch_from_pipeline(
    pipeline,
    t_start: int,
    t_max: int,
    include_deformation_gradient: bool = False,
):
    if t_start < 0 or t_start >= pipeline.num_frames:
        raise ValueError(
            f"t_start={t_start} is outside trajectory with {pipeline.num_frames} frames"
        )

    t_max = min(t_max, pipeline.num_frames - t_start)
    columns = get_ptm_columns(include_deformation_gradient)

    first = pipeline.compute(t_start)
    natoms = first.particles.count
    ptm_traj_all = np.empty((t_max, natoms, len(columns)), dtype=np.float32)

    for local_t, global_t in enumerate(range(t_start, t_start + t_max)):
        if local_t == 0:
            data = first
        else:
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
            ).reshape(natoms, 9)
            out_cols.append(F)

        ptm_traj_all[local_t, :, :] = np.concatenate(out_cols, axis=1)

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

    ptm = _make_ptm_modifier(
        rmsd_cutoff=rmsd_cutoff,
        include_deformation_gradient=include_deformation_gradient,
    )
    pipeline.modifiers.append(ptm)

    columns = get_ptm_columns(include_deformation_gradient)

    return pipeline, columns


def create_restricted_ptm_rmsd_pipelines(
    filename: str,
    type_names=("fcc", "hcp", "bcc", "ico"),
    rmsd_cutoff: float = 0.0,
):
    """
    Create one PTM pipeline per requested structure type.

    OVITO's PTM modifier only outputs the winning RMSD. To approximate a
    per-class RMSD vector, each pipeline enables exactly one structure type
    and disables all other PTM candidates. Setting rmsd_cutoff=0 disables the
    cutoff, so high-RMSD matches remain available instead of becoming "Other".
    """
    type_names = parse_ptm_rmsd_type_names(type_names)
    pipelines = []

    for type_name in type_names:
        pipeline = import_file(filename, multiple_frames=True)
        ptm = _make_ptm_modifier(
            rmsd_cutoff=rmsd_cutoff,
            output_interatomic_distance=False,
            output_orientation=False,
        )

        enabled_type = PTM_RMSD_TYPE_IDS[type_name]
        for structure_type in PTM_RMSD_TYPE_IDS.values():
            ptm.structures[structure_type].enabled = False
        ptm.structures[enabled_type].enabled = True

        pipeline.modifiers.append(ptm)
        pipelines.append(pipeline)

    return pipelines, type_names


def compute_ptm_rmsd_by_type_batch(
    pipelines,
    t_start: int,
    t_max: int,
):
    if not pipelines:
        return None

    first_pipeline = pipelines[0]
    if t_start < 0 or t_start >= first_pipeline.num_frames:
        raise ValueError(
            f"t_start={t_start} is outside trajectory with {first_pipeline.num_frames} frames"
        )

    t_max = min(t_max, first_pipeline.num_frames - t_start)
    first = first_pipeline.compute(t_start)
    natoms = first.particles.count
    rmsd_by_type = np.empty((t_max, natoms, len(pipelines)), dtype=np.float32)

    for type_idx, pipeline in enumerate(pipelines):
        for local_t, global_t in enumerate(range(t_start, t_start + t_max)):
            if type_idx == 0 and local_t == 0:
                data = first
            else:
                data = pipeline.compute(global_t)

            particles = data.particles
            structure_type = np.asarray(particles["Structure Type"])
            rmsd = np.asarray(particles["RMSD"], dtype=np.float32)

            # If no topology-compatible template match was found, OVITO reports
            # "Other". Keep these cases distinct from high-RMSD valid matches.
            rmsd = rmsd.copy()
            rmsd[structure_type == 0] = np.nan
            rmsd_by_type[local_t, :, type_idx] = rmsd

    return rmsd_by_type


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
