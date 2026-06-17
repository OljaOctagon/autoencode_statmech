import argparse
import logging
import os
from pathlib import Path

from lammpstools import write_denoised_lammpstrj


def count_complete_lammpstrj_frames(filename):
    n_frames = 0
    last_complete_offset = 0
    with open(filename, "rb") as traj_file:
        while traj_file.readline().startswith(b"ITEM: TIMESTEP"):
            if not traj_file.readline():
                break
            if not traj_file.readline().startswith(b"ITEM: NUMBER OF ATOMS"):
                break

            natoms_line = traj_file.readline()
            if not natoms_line:
                break
            natoms = int(natoms_line)

            if not traj_file.readline().startswith(b"ITEM: BOX BOUNDS"):
                break
            if not all(traj_file.readline() for _ in range(3)):
                break
            if not traj_file.readline().startswith(b"ITEM: ATOMS"):
                break
            if not all(traj_file.readline() for _ in range(natoms)):
                break

            n_frames += 1
            last_complete_offset = traj_file.tell()

    return n_frames, last_complete_offset


def append_file(src, dst):
    with open(dst, "a+b") as dst_file:
        if dst_file.tell() > 0:
            dst_file.seek(-1, os.SEEK_END)
            if dst_file.read(1) != b"\n":
                dst_file.write(b"\n")
        with open(src, "rb") as src_file:
            dst_file.write(src_file.read())


def append_complete_frames(src, dst, expected_new_frames):
    src_frames, src_offset = count_complete_lammpstrj_frames(src)
    src_size = src.stat().st_size
    if src_frames != expected_new_frames or src_offset != src_size:
        raise RuntimeError(
            f"Temporary output {src} has {src_frames}/{expected_new_frames} "
            "complete frames."
        )

    original_size = dst.stat().st_size if dst.exists() else 0
    try:
        append_file(src, dst)
        with open(dst, "a+b") as dst_file:
            dst_file.flush()
            os.fsync(dst_file.fileno())
    except Exception:
        with open(dst, "r+b") as dst_file:
            dst_file.truncate(original_size)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default="traj.lammpstrj")
    parser.add_argument("-o", default=None)
    parser.add_argument("-structure", default="FCC")
    parser.add_argument("-steps", default=8)
    parser.add_argument("-device", default="cpu")
    parser.add_argument("-scale", default=None)
    parser.add_argument("-model_path", default=None)
    parser.add_argument(
        "-resume_chunk_size",
        default=1,
        type=int,
        help="Number of frames to denoise before appending progress to the output.",
    )
    args = parser.parse_args()

    input_path = Path(args.f)
    output_path = (
        Path(args.o)
        if args.o is not None
        else input_path.with_name(f"{input_path.stem}_denoised_{args.structure.lower()}.lammpstrj")
    )

    logging.info(
        "Writing denoised trajectory %s from %s using structure=%s, steps=%s, device=%s.",
        output_path,
        input_path,
        args.structure,
        args.steps,
        args.device,
    )
    input_frames, _ = count_complete_lammpstrj_frames(input_path)
    resume_frame = 0
    if output_path.exists():
        resume_frame, output_offset = count_complete_lammpstrj_frames(output_path)
        output_size = output_path.stat().st_size
        if output_offset < output_size:
            with open(output_path, "r+b") as output_file:
                output_file.truncate(output_offset)

    resume_chunk_size = max(1, int(args.resume_chunk_size))

    if resume_frame >= input_frames:
        logging.info(
            "Denoised trajectory already has %s/%s frames; nothing to do.",
            resume_frame,
            input_frames,
        )
    else:
        if resume_frame > 0:
            logging.info("Resuming from frame %s/%s.", resume_frame, input_frames)

        next_frame = resume_frame
        while next_frame < input_frames:
            chunk_start = next_frame
            chunk_end = min(input_frames - 1, chunk_start + resume_chunk_size - 1)
            chunk_frames = chunk_end - chunk_start + 1
            tmp_output_path = output_path.with_name(
                f".{output_path.name}.frames.{chunk_start}-{chunk_end}.{os.getpid()}.tmp"
            )
            logging.info(
                "Denoising frames %s-%s/%s.",
                chunk_start,
                chunk_end,
                input_frames - 1,
            )
            write_denoised_lammpstrj(
                input_filename=input_path,
                output_filename=tmp_output_path,
                structure=args.structure,
                steps=int(args.steps),
                device=args.device,
                scale=None if args.scale is None else float(args.scale),
                model_path=args.model_path,
                start_frame=chunk_start,
                end_frame=chunk_end,
            )
            append_complete_frames(tmp_output_path, output_path, chunk_frames)
            tmp_output_path.unlink(missing_ok=True)
            next_frame = chunk_end + 1
            logging.info("Denoised trajectory now has %s/%s frames.", next_frame, input_frames)
    logging.info("Done.")
