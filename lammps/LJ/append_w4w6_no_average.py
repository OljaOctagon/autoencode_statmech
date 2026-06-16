import argparse
import logging
from pathlib import Path

import freud
import numpy as np
from tqdm import tqdm

from preprocess_traj import iter_lammpstrj_batches


QL_VALUES = list(range(1, 21))


def compute_ql_no_average_batch(Config, Box, nn):
    batch_len, natoms, _ = Config.shape
    order = freud.order.Steinhardt(
        l=QL_VALUES,
        average=False,
        wl=False,
    )
    ql_no_average = np.empty((batch_len, natoms, len(QL_VALUES)), dtype=np.float32)

    for local_t in range(batch_len):
        box_lengths = np.asarray(Box[local_t], dtype=np.float32)
        positions = np.asarray(Config[local_t], dtype=np.float32) * box_lengths
        freud_box = freud.box.Box(*box_lengths)
        order.compute(
            system=(freud_box, positions),
            neighbors={"num_neighbors": nn},
        )
        ql_no_average[local_t] = order.particle_order

    return ql_no_average


def compute_w4w6_no_average_batch(Config, Box, nn):
    batch_len, natoms, _ = Config.shape
    order = freud.order.Steinhardt(
        l=[4, 6],
        average=False,
        wl=True,
    )
    w4w6_no_average = np.empty((batch_len, natoms, 2), dtype=np.float32)

    for local_t in range(batch_len):
        box_lengths = np.asarray(Box[local_t], dtype=np.float32)
        positions = np.asarray(Config[local_t], dtype=np.float32) * box_lengths
        freud_box = freud.box.Box(*box_lengths)
        order.compute(
            system=(freud_box, positions),
            neighbors={"num_neighbors": nn},
        )
        w4w6_no_average[local_t] = order.particle_order

    return w4w6_no_average


def load_npz_dict(npz_path):
    with np.load(npz_path, allow_pickle=True) as arr:
        return {key: arr[key] for key in arr.files}


def append_order_features(
    npz_path,
    traj_path,
    batch_size,
    overwrite=False,
    include_w4w6_no_average=True,
    include_ql_no_average=True,
):
    data = load_npz_dict(npz_path)
    if not include_w4w6_no_average and not include_ql_no_average:
        raise ValueError("Nothing to append. Enable at least one feature.")
    if "w4w6_no_average" in data and include_w4w6_no_average and not overwrite:
        raise ValueError(f"{npz_path} already contains 'w4w6_no_average'. Pass -overwrite to replace it.")
    if "ql_no_average" in data and include_ql_no_average and not overwrite:
        raise ValueError(f"{npz_path} already contains 'ql_no_average'. Pass -overwrite to replace it.")
    if "picked_ids" not in data or "frame_indices" not in data:
        raise ValueError(
            "Cannot append exact per-particle descriptors because this npz file "
            "does not contain 'picked_ids' and 'frame_indices'. Regenerate the "
            "npz once with the updated preprocess_traj.py, then future append-only "
            "descriptor additions will be possible."
        )
    if "nn" not in data:
        raise ValueError("Cannot infer nearest-neighbor count because 'nn' is missing.")

    picked_ids = np.asarray(data["picked_ids"], dtype=np.int64)
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
    nn = int(np.asarray(data["nn"]))

    w4w6_out = None
    ql_out = None
    if include_w4w6_no_average:
        w4w6_out = np.empty((picked_ids.shape[0], 2), dtype=np.float32)
    if include_ql_no_average:
        ql_out = np.empty((picked_ids.shape[0], len(QL_VALUES)), dtype=np.float32)
    total_frames = int(frame_indices.max()) + 1
    total_batches = (total_frames + batch_size - 1) // batch_size

    for batch_start, Config, Box, _ in tqdm(
        iter_lammpstrj_batches(traj_path, batch_size),
        total=total_batches,
        desc="Appending order features",
    ):
        batch_len = len(Config)
        w4w6_no_average = None
        ql_no_average = None
        if include_w4w6_no_average:
            w4w6_no_average = compute_w4w6_no_average_batch(Config, Box, nn)
        if include_ql_no_average:
            ql_no_average = compute_ql_no_average_batch(Config, Box, nn)

        for local_t in range(batch_len):
            frame_idx = batch_start + local_t
            sample_mask = frame_indices == frame_idx
            if not np.any(sample_mask):
                continue
            if include_w4w6_no_average:
                w4w6_out[sample_mask] = w4w6_no_average[
                    local_t,
                    picked_ids[sample_mask],
                    :,
                ]
            if include_ql_no_average:
                ql_out[sample_mask] = ql_no_average[
                    local_t,
                    picked_ids[sample_mask],
                    :,
                ]

    if include_w4w6_no_average:
        data["w4w6_no_average"] = w4w6_out
    if include_ql_no_average:
        data["ql_no_average"] = ql_out
    tmp_path = npz_path.with_name(f"{npz_path.stem}.tmp.npz")
    np.savez_compressed(tmp_path, **data)
    tmp_path.replace(npz_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-npz", default="particle_data.npz")
    parser.add_argument("-f", default="traj.lammpstrj")
    parser.add_argument("-batch", default=10)
    parser.add_argument("-overwrite", action="store_true")
    parser.add_argument(
        "-only",
        choices=["w4w6_no_average", "ql_no_average", "both"],
        default="both",
    )
    args = parser.parse_args()

    append_order_features(
        npz_path=Path(args.npz),
        traj_path=Path(args.f),
        batch_size=int(args.batch),
        overwrite=args.overwrite,
        include_w4w6_no_average=args.only in {"w4w6_no_average", "both"},
        include_ql_no_average=args.only in {"ql_no_average", "both"},
    )
