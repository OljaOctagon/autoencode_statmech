from typing import Optional, Tuple
import numpy as np

from ovito.data import DataCollection, SimulationCell
from ovito.io import import_file
from ovito.pipeline import Pipeline, StaticSource
from ovito.modifiers import CommonNeighborAnalysisModifier


def _make_ovito_data_from_frame(
    positions: np.ndarray,
    box_lengths: np.ndarray,
    fractional: bool = True,
    pbc: Tuple[bool, bool, bool] = (True, True, True),
) -> DataCollection:
    """
    Build an OVITO DataCollection for one frame.
    Assumes an orthorhombic box [Lx, Ly, Lz].
    """
    positions = np.asarray(positions, dtype=np.float64)
    box_lengths = np.asarray(box_lengths, dtype=np.float64)

    if fractional:
        positions = positions * box_lengths

    data = DataCollection()

    particles = data.create_particles()
    particles.create_property("Position", data=positions)

    cell = SimulationCell(pbc=pbc)
    Lx, Ly, Lz = box_lengths
    cell[...] = [
        [Lx, 0.0, 0.0, 0.0],
        [0.0, Ly, 0.0, 0.0],
        [0.0, 0.0, Lz, 0.0],
    ]
    data.objects.append(cell)

    return data


def _compute_cna_trajectory(
    Config,
    Box,
    Natoms: Optional[int] = None,
    t_max: Optional[int] = None,
    fractional: bool = True,
    fixed_cutoff: Optional[float] = None,
    include_fixed: bool = True,
    include_adaptive: bool = True,
    include_interval: bool = True,
    return_columns: bool = False,
):
    """
    Compute CNA structure classifications for all frames.

    Parameters
    ----------
    Config : array-like, shape (T, Natoms, 3)
        Particle coordinates per frame.
        Fractional coordinates by default.
    Box : array-like, shape (T, 3)
        Orthorhombic box lengths [Lx, Ly, Lz] per frame.
    Natoms : int or None
        Number of atoms. If None, inferred from Config.shape[1].
    t_max : int or None
        Number of frames to use.
    fractional : bool
        If True, Config is interpreted as fractional coordinates.
    fixed_cutoff : float or None
        Cutoff for conventional fixed-cutoff CNA.
        Required if include_fixed=True.
    include_fixed : bool
        Compute conventional CNA.
    include_adaptive : bool
        Compute adaptive CNA.
    include_interval : bool
        Compute interval CNA.
    return_columns : bool
        If True, return (array, column_names).

    Returns
    -------
    cna_traj_all : np.ndarray
        Shape: (T_used, Natoms, num_modes)

        Structure type encoding:
            0 = Other
            1 = FCC
            2 = HCP
            3 = BCC
            4 = ICO
    """
    Config = np.asarray(Config)
    Box = np.asarray(Box)

    T_total = len(Config)
    if t_max is None or t_max > T_total:
        t_max = T_total

    if Natoms is None:
        Natoms = Config.shape[1]
    Natoms = int(Natoms)

    modes, columns = _make_cna_modes(
        fixed_cutoff=fixed_cutoff,
        include_fixed=include_fixed,
        include_adaptive=include_adaptive,
        include_interval=include_interval,
    )

    cna_traj_all = np.empty((t_max, Natoms, len(modes)), dtype=np.int8)

    for t in range(t_max):
        frame_data = _make_ovito_data_from_frame(
            positions=Config[t],
            box_lengths=Box[t],
            fractional=fractional,
        )

        for mode_idx, (_, modifier) in enumerate(modes):
            pipeline = Pipeline(source=StaticSource(data=frame_data))
            pipeline.modifiers.append(modifier)

            result = pipeline.compute()
            structure_type = np.asarray(
                result.particles["Structure Type"],
                dtype=np.int8,
            )

            cna_traj_all[t, :, mode_idx] = structure_type

    if return_columns:
        return cna_traj_all, columns

    return cna_traj_all


def _make_cna_modes(
    fixed_cutoff: Optional[float] = None,
    include_fixed: bool = True,
    include_adaptive: bool = True,
    include_interval: bool = True,
):
    modes = []
    columns = []

    if include_fixed:
        if fixed_cutoff is None:
            raise ValueError(
                "fixed_cutoff must be provided when include_fixed=True."
            )
        modes.append(
            (
                "cna_fixed",
                CommonNeighborAnalysisModifier(
                    mode=CommonNeighborAnalysisModifier.Mode.FixedCutoff,
                    cutoff=float(fixed_cutoff),
                ),
            )
        )
        columns.append("cna_fixed")

    if include_adaptive:
        modes.append(
            (
                "cna_adaptive",
                CommonNeighborAnalysisModifier(
                    mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff,
                ),
            )
        )
        columns.append("cna_adaptive")

    if include_interval:
        modes.append(
            (
                "cna_interval",
                CommonNeighborAnalysisModifier(
                    mode=CommonNeighborAnalysisModifier.Mode.IntervalCutoff,
                ),
            )
        )
        columns.append("cna_interval")

    if not modes:
        raise ValueError("At least one CNA mode must be enabled.")

    return modes, columns


def create_cna_pipelines_from_file(
    filename: str,
    fixed_cutoff: Optional[float] = None,
    include_fixed: bool = True,
    include_adaptive: bool = True,
    include_interval: bool = True,
):
    modes, columns = _make_cna_modes(
        fixed_cutoff=fixed_cutoff,
        include_fixed=include_fixed,
        include_adaptive=include_adaptive,
        include_interval=include_interval,
    )
    pipelines = []

    for _, modifier in modes:
        pipeline = import_file(filename, multiple_frames=True)
        pipeline.modifiers.append(modifier)
        pipelines.append(pipeline)

    return pipelines, columns


def compute_cna_batch_from_pipelines(
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
    cna = np.empty((t_max, natoms, len(pipelines)), dtype=np.int8)

    for mode_idx, pipeline in enumerate(pipelines):
        for local_t, global_t in enumerate(range(t_start, t_start + t_max)):
            if mode_idx == 0 and local_t == 0:
                data = first
            else:
                data = pipeline.compute(global_t)

            cna[local_t, :, mode_idx] = np.asarray(
                data.particles["Structure Type"],
                dtype=np.int8,
            )

    return cna


def compute_cna_trajectory(*args, **kwargs):
    return _compute_cna_trajectory(*args, **kwargs)
