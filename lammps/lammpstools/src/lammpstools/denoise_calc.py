from pathlib import Path

from ovito.io import export_file, import_file


def write_denoised_lammpstrj(
    input_filename,
    output_filename,
    structure="FCC",
    steps=8,
    device="cpu",
    scale=None,
    model_path=None,
    start_frame=0,
    end_frame=None,
):
    """
    Apply the OVITO score-based denoiser once and write a denoised trajectory.

    This function intentionally keeps denoising separate from descriptor
    computation, so PTM/CNA can run on a cached denoised dump instead of
    re-running the denoiser for every descriptor pipeline.
    """
    try:
        from scoreBasedDenoising import ScoreBasedDenoising
    except ImportError as exc:
        raise ImportError(
            "ScoreBasedDenoising is not installed. Install "
            "https://github.com/ovito-org/ScoreBasedDenoising in the active "
            "environment before using -denoise."
        ) from exc

    input_filename = str(input_filename)
    output_filename = str(output_filename)

    pipeline = import_file(input_filename, multiple_frames=True)
    denoiser = ScoreBasedDenoising(
        structure=str(structure),
        steps=int(steps),
        device=str(device),
    )
    if scale is not None:
        denoiser.scale = float(scale)
    if model_path is not None:
        denoiser.model_path = str(model_path)

    pipeline.modifiers.append(denoiser)

    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "multiple_frames": True,
        "start_frame": int(start_frame),
        "columns": [
            "Particle Identifier",
            "Particle Type",
            "Position.X",
            "Position.Y",
            "Position.Z",
        ],
    }
    if end_frame is not None:
        export_kwargs["end_frame"] = int(end_frame)

    export_file(pipeline, output_filename, "lammps/dump", **export_kwargs)

    return Path(output_filename)
