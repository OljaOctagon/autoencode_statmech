import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lammpstools import (
    compute_cna_batch_from_pipelines,
    compute_ptm_batch_from_pipeline,
    compute_ptm_rmsd_by_type_batch,
    create_cna_pipelines_from_file,
    create_ptm_pipeline,
    create_restricted_ptm_rmsd_pipelines,
    parse_ptm_rmsd_type_names,
)


def load_npz_dict(npz_path):
    with np.load(npz_path, allow_pickle=True) as arr:
        return {key: arr[key] for key in arr.files}


def _coerce_type_names(values):
    if values is None:
        return None

    names = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def append_denoised_features(
    npz_path,
    denoised_traj_path,
    batch_size,
    cna_cutoff=1.5,
    ptm_rmsd_type_names=None,
    overwrite=False,
    denoise_structure=None,
    denoise_steps=None,
):
    data = load_npz_dict(npz_path)

    required_fields = ("picked_ids", "frame_indices")
    missing = [field for field in required_fields if field not in data]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Cannot append denoised descriptors because this npz file is missing "
            f"{missing_text}. Regenerate the base npz once with the current "
            "preprocess_traj.py and then append-only enrichment will work."
        )

    denoised_fields = (
        "ptm_denoised",
        "ptm_denoised_rmsd_by_type",
        "cna_denoised",
        "cna_denoised_columns",
        "ptm_denoised_rmsd_type_names",
    )
    existing = [field for field in denoised_fields if field in data]
    if existing and not overwrite:
        existing_text = ", ".join(existing)
        raise ValueError(
            f"{npz_path} already contains denoised fields ({existing_text}). "
            "Pass -overwrite to replace them."
        )

    picked_ids = np.asarray(data["picked_ids"], dtype=np.int64)
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
    total_frames = int(frame_indices.max()) + 1

    if ptm_rmsd_type_names is None:
        ptm_rmsd_type_names = _coerce_type_names(data.get("ptm_rmsd_type_names"))
    if ptm_rmsd_type_names is None:
        ptm_rmsd_type_names = parse_ptm_rmsd_type_names("fcc,hcp,bcc,ico")

    ptm_pipeline, ptm_columns = create_ptm_pipeline(filename=str(denoised_traj_path))
    ptm_rmsd_pipelines, ptm_type_names = create_restricted_ptm_rmsd_pipelines(
        filename=str(denoised_traj_path),
        type_names=ptm_rmsd_type_names,
        rmsd_cutoff=0.0,
    )
    cna_pipelines, cna_columns = create_cna_pipelines_from_file(
        filename=str(denoised_traj_path),
        fixed_cutoff=float(cna_cutoff),
        include_fixed=True,
        include_adaptive=True,
        include_interval=True,
    )

    if ptm_pipeline.num_frames != total_frames:
        raise ValueError(
            "Denoised trajectory frame count does not match the existing npz "
            f"sample metadata ({ptm_pipeline.num_frames} != {total_frames})."
        )

    ptm_out = np.empty((picked_ids.shape[0], len(ptm_columns)), dtype=np.float32)
    ptm_rmsd_out = np.empty((picked_ids.shape[0], len(ptm_type_names)), dtype=np.float32)
    cna_out = np.empty((picked_ids.shape[0], len(cna_columns)), dtype=np.int8)

    total_batches = (total_frames + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, total_frames, batch_size), total=total_batches, desc="Appending denoised features"):
        t_max = min(batch_size, total_frames - batch_start)
        ptm_batch = compute_ptm_batch_from_pipeline(
            ptm_pipeline,
            t_start=batch_start,
            t_max=t_max,
        )
        ptm_rmsd_batch = compute_ptm_rmsd_by_type_batch(
            ptm_rmsd_pipelines,
            t_start=batch_start,
            t_max=t_max,
        )
        cna_batch = compute_cna_batch_from_pipelines(
            cna_pipelines,
            t_start=batch_start,
            t_max=t_max,
        )

        for local_t in range(t_max):
            frame_idx = batch_start + local_t
            sample_mask = frame_indices == frame_idx
            if not np.any(sample_mask):
                continue

            frame_picked_ids = picked_ids[sample_mask]
            ptm_out[sample_mask] = ptm_batch[local_t, frame_picked_ids, :]
            ptm_rmsd_out[sample_mask] = ptm_rmsd_batch[local_t, frame_picked_ids, :]
            cna_out[sample_mask] = cna_batch[local_t, frame_picked_ids, :]

    data["ptm_denoised"] = ptm_out
    data["ptm_denoised_rmsd_by_type"] = ptm_rmsd_out
    data["ptm_denoised_rmsd_type_names"] = np.array(ptm_type_names)
    data["cna_denoised"] = cna_out
    data["cna_denoised_columns"] = np.array(cna_columns)
    if denoise_structure is not None:
        data["denoise_structure"] = np.array(denoise_structure)
    if denoise_steps is not None:
        data["denoise_steps"] = np.array(int(denoise_steps))

    tmp_path = npz_path.with_name(f"{npz_path.stem}.tmp.npz")
    np.savez_compressed(tmp_path, **data)
    tmp_path.replace(npz_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-npz", default="particle_data.npz")
    parser.add_argument("-denoised_traj", required=True)
    parser.add_argument("-batch", default=10)
    parser.add_argument("-cna_cutoff", default=1.5)
    parser.add_argument("-ptm_rmsd_types", default=None)
    parser.add_argument("-denoise_structure", default=None)
    parser.add_argument("-denoise_steps", default=None)
    parser.add_argument("-overwrite", action="store_true")
    args = parser.parse_args()

    ptm_rmsd_type_names = None
    if args.ptm_rmsd_types is not None:
        ptm_rmsd_type_names = parse_ptm_rmsd_type_names(args.ptm_rmsd_types)

    append_denoised_features(
        npz_path=Path(args.npz),
        denoised_traj_path=Path(args.denoised_traj),
        batch_size=int(args.batch),
        cna_cutoff=float(args.cna_cutoff),
        ptm_rmsd_type_names=ptm_rmsd_type_names,
        overwrite=args.overwrite,
        denoise_structure=args.denoise_structure,
        denoise_steps=args.denoise_steps,
    )
