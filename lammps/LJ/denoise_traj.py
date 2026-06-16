import argparse
import logging
from pathlib import Path

from lammpstools import write_denoised_lammpstrj


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
    write_denoised_lammpstrj(
        input_filename=input_path,
        output_filename=output_path,
        structure=args.structure,
        steps=int(args.steps),
        device=args.device,
        scale=None if args.scale is None else float(args.scale),
        model_path=args.model_path,
    )
    logging.info("Done.")
