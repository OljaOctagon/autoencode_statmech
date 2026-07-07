#!/usr/bin/env python3
"""Fetch or generate extra particle datasets that stress q_l/q_ls baselines.

The target directory is intentionally ignored by git. This file is the only
versioned artifact in ``other_data``; downloaded/generated payloads stay local.

Recommended use from this directory:

    python download_q_ls_hard_data.py list
    python download_q_ls_hard_data.py generate-noisy-crystals
    python download_q_ls_hard_data.py submit-martirossyan-globus \
        --destination-endpoint <YOUR_GLOBUS_CONNECT_PERSONAL_ENDPOINT_ID> \
        --destination-path /~/autoencode_statmech/other_data/martirossyan_complex_crystals/

The Martirossyan dataset is distributed through Globus by the Materials Data
Facility, not as a plain HTTP archive. The script submits a Globus transfer when
given a destination endpoint. For the local workstation, that usually means a
Globus Connect Personal endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    source_type: str
    url: str
    doi: str | None
    license: str | None
    local_dir: str
    rationale: str
    access_note: str


MARTIROSSYAN_SOURCE_ENDPOINT = "82f1b5c6-6e9b-11e5-ba47-22000b92c6ec"
MARTIROSSYAN_SOURCE_PATH = "/mdf_open/e52f77de-6756-4ca9-8fdb-f4791b395c1f/1.0/"
MARTIROSSYAN_GLOBUS_URL = (
    "https://app.globus.org/file-manager"
    "?origin_id=82f1b5c6-6e9b-11e5-ba47-22000b92c6ec"
    "&origin_path=/mdf_open/e52f77de-6756-4ca9-8fdb-f4791b395c1f/1.0/"
)
UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


SOURCES = {
    "martirossyan_complex_crystals": Source(
        key="martirossyan_complex_crystals",
        title=(
            "Martirossyan et al. 2024, Local structural features elucidate "
            "crystallization of complex structures"
        ),
        source_type="globus_dataset",
        url="https://acdc.alcf.anl.gov/mdf/detail/e52f77de-6756-4ca9-8fdb-f4791b395c1f-1.0/",
        doi="10.18126/wy01-4e11",
        license="CC BY 4.0",
        local_dir="martirossyan_complex_crystals",
        rationale=(
            "One-component particles with isotropic multiwell pair potentials; "
            "10 complex crystal structures plus pathway variants. Multiple "
            "local motifs make scalar q_l/q_ls descriptors a useful stress test."
        ),
        access_note=(
            "Public MDF/ACDC record. Data are exposed via Globus at "
            f"{MARTIROSSYAN_GLOBUS_URL}"
        ),
    ),
    "noisy_simple_crystals": Source(
        key="noisy_simple_crystals",
        title=(
            "Haeberle-style synthetic noisy crystal structures "
            "(fcc, hcp, bcc, sc, diamond)"
        ),
        source_type="generated",
        url="https://arxiv.org/abs/1906.08111",
        doi="10.48550/arXiv.1906.08111",
        license=None,
        local_dir="noisy_simple_crystals",
        rationale=(
            "Controlled 3D particle structures with thermal-like displacement "
            "noise. Fcc/hcp/bcc confusion under neighbor-definition changes is "
            "a known q_l/q_ls failure mode."
        ),
        access_note=(
            "No stable raw-data archive was found; this script generates a "
            "reproducible local benchmark from the published setup idea."
        ),
    ),
    "gispen_lj_nucleation_manual": Source(
        key="gispen_lj_nucleation_manual",
        title=(
            "Gispen et al. 2024, bcc coating of Lennard-Jones crystal nuclei "
            "and local-structure detector dependence"
        ),
        source_type="manual_candidate",
        url="https://arxiv.org/abs/2412.03276",
        doi="10.48550/arXiv.2412.03276",
        license=None,
        local_dir="gispen_lj_nucleation",
        rationale=(
            "Canonical isotropic Lennard-Jones nucleation trajectories where "
            "fcc/hcp/bcc/interface labels depend strongly on the local detector."
        ),
        access_note=(
            "The paper states that trajectories are in the supplementary "
            "material, but the arXiv source bundle contains only manuscript "
            "source and figures. Add this source manually if a stable archive "
            "or journal supplement endpoint becomes available."
        ),
    ),
}


def write_manifest(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Additional 3D particle crystalline or near-crystalline data for "
            "testing q_l/q_ls and related Steinhardt-style baselines."
        ),
        "sources": {key: asdict(source) for key, source in SOURCES.items()},
        "notes": [
            "Only this downloader is meant to be tracked in git.",
            "Downloaded archives, extracted trajectories, and generated npz files stay ignored.",
        ],
    }
    manifest_path = output_dir / "q_ls_hard_data_sources.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def list_sources() -> None:
    for source in SOURCES.values():
        print(f"{source.key}")
        print(f"  type: {source.source_type}")
        print(f"  title: {source.title}")
        print(f"  url: {source.url}")
        if source.doi:
            print(f"  doi: {source.doi}")
        if source.license:
            print(f"  license: {source.license}")
        print(f"  local_dir: {source.local_dir}")
        print(f"  rationale: {source.rationale}")
        print(f"  access: {source.access_note}")
        print()


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def submit_martirossyan_globus(args: argparse.Namespace) -> None:
    if not args.dry_run and shutil.which("globus") is None:
        raise SystemExit(
            "The 'globus' CLI is required for this dataset. Install it with "
            "'python -m pip install globus-cli', then run 'globus login'."
        )

    if not args.destination_endpoint or not args.destination_path:
        raise SystemExit(
            "Martirossyan data are on Globus. Provide both "
            "--destination-endpoint and --destination-path for your Globus "
            "Connect Personal endpoint, or open this URL manually:\n"
            f"{MARTIROSSYAN_GLOBUS_URL}"
        )
    if (
        args.destination_endpoint == "YOUR_ENDPOINT_ID"
        or not UUID_PATTERN.match(args.destination_endpoint)
    ):
        raise SystemExit(
            "--destination-endpoint must be your real Globus endpoint UUID, "
            "not the placeholder YOUR_ENDPOINT_ID. Run 'globus endpoint search "
            "\"Globus Connect Personal\"' or inspect your endpoint in the "
            "Globus web app, then rerun with that UUID."
        )

    destination_path = args.destination_path
    if not destination_path.endswith("/"):
        destination_path += "/"

    command = [
        "globus",
        "transfer",
        f"{MARTIROSSYAN_SOURCE_ENDPOINT}:{MARTIROSSYAN_SOURCE_PATH}",
        f"{args.destination_endpoint}:{destination_path}",
        "--recursive",
        "--label",
        args.label,
    ]
    if args.sync_level:
        command.extend(["--sync-level", args.sync_level])
    run_command(command, dry_run=args.dry_run)


def primitive_offsets(kind: str) -> list[tuple[float, float, float]]:
    if kind == "sc":
        return [(0.0, 0.0, 0.0)]
    if kind == "bcc":
        return [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
    if kind == "fcc":
        return [
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ]
    if kind == "diamond":
        fcc = primitive_offsets("fcc")
        return fcc + [(x + 0.25, y + 0.25, z + 0.25) for x, y, z in fcc]
    raise ValueError(f"unknown cubic lattice: {kind}")


def cubic_lattice(kind: str, cells: int, lattice_constant: float) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    offsets = primitive_offsets(kind)
    points: list[tuple[float, float, float]] = []
    for i in range(cells):
        for j in range(cells):
            for k in range(cells):
                for ox, oy, oz in offsets:
                    points.append((i + ox, j + oy, k + oz))
    positions = np.asarray(points, dtype=np.float64) * lattice_constant
    box = np.asarray([cells, cells, cells], dtype=np.float64) * lattice_constant
    return positions, box


def hcp_lattice(cells: int, lattice_constant: float) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    # Conventional orthorhombic HCP cell with four atoms.
    a = lattice_constant
    c = math.sqrt(8.0 / 3.0) * a
    cell = np.asarray(
        [
            [a, 0.0, 0.0],
            [0.0, math.sqrt(3.0) * a, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=np.float64,
    )
    fractional = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 1.0 / 6.0, 0.5],
            [0.0, 2.0 / 3.0, 0.5],
        ],
        dtype=np.float64,
    )

    points: list["np.ndarray"] = []
    for i in range(cells):
        for j in range(cells):
            for k in range(cells):
                shift = np.asarray([i, j, k], dtype=np.float64)
                for frac in fractional:
                    points.append((shift + frac) @ cell)
    box = np.asarray([cells * a, cells * math.sqrt(3.0) * a, cells * c], dtype=np.float64)
    return np.asarray(points, dtype=np.float64), box


def wrap_positions(positions: "np.ndarray", box: "np.ndarray") -> "np.ndarray":
    import numpy as np

    return np.mod(positions, box)


def make_noisy_snapshots(
    *,
    structures: Iterable[str],
    cells: int,
    samples_per_structure: int,
    noise_levels: Iterable[float],
    lattice_constant: float,
    seed: int,
) -> dict[str, object]:
    import numpy as np

    rng = np.random.default_rng(seed)
    arrays: dict[str, object] = {
        "structures": np.asarray(tuple(structures)),
        "noise_levels": np.asarray(tuple(noise_levels), dtype=np.float64),
    }

    for structure in arrays["structures"]:
        if structure == "hcp":
            base_positions, box = hcp_lattice(cells, lattice_constant)
        else:
            base_positions, box = cubic_lattice(structure, cells, lattice_constant)

        positions: list[np.ndarray] = []
        boxes: list[np.ndarray] = []
        per_snapshot_noise: list[float] = []
        for noise in noise_levels:
            sigma = noise * lattice_constant
            for _ in range(samples_per_structure):
                displaced = base_positions + rng.normal(0.0, sigma, size=base_positions.shape)
                positions.append(wrap_positions(displaced, box))
                boxes.append(box.copy())
                per_snapshot_noise.append(float(noise))

        arrays[f"positions_{structure}"] = np.asarray(positions, dtype=np.float64)
        arrays[f"boxes_{structure}"] = np.asarray(boxes, dtype=np.float64)
        arrays[f"noise_{structure}"] = np.asarray(per_snapshot_noise, dtype=np.float64)

    return arrays


def generate_noisy_crystals(args: argparse.Namespace) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "numpy is required for synthetic data generation. Install numpy in "
            "the active environment and rerun this command."
        ) from exc

    output_dir = args.output_dir.resolve()
    dataset_dir = output_dir / SOURCES["noisy_simple_crystals"].local_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    structures = tuple(args.structures.split(","))
    noise_levels = tuple(float(item) for item in args.noise_levels.split(","))
    valid = {"fcc", "hcp", "bcc", "sc", "diamond"}
    unknown = sorted(set(structures) - valid)
    if unknown:
        raise SystemExit(f"unknown structures: {', '.join(unknown)}")

    snapshots = make_noisy_snapshots(
        structures=structures,
        cells=args.cells,
        samples_per_structure=args.samples_per_structure,
        noise_levels=noise_levels,
        lattice_constant=args.lattice_constant,
        seed=args.seed,
    )

    data_path = dataset_dir / args.filename
    np.savez_compressed(data_path, **snapshots)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": asdict(SOURCES["noisy_simple_crystals"]),
        "parameters": {
            "structures": structures,
            "cells": args.cells,
            "samples_per_structure": args.samples_per_structure,
            "noise_levels": noise_levels,
            "lattice_constant": args.lattice_constant,
            "seed": args.seed,
        },
        "arrays": {
            "structures": "shape (structures,), labels included in this file",
            "noise_levels": "shape (noise_levels,), requested Gaussian displacement sigmas/a",
            "positions_<structure>": (
                "shape (snapshots, particles_for_structure, 3), periodic "
                "wrapped Cartesian coordinates"
            ),
            "boxes_<structure>": "shape (snapshots, 3), orthorhombic periodic box lengths",
            "noise_<structure>": "shape (snapshots,), Gaussian displacement sigma/a",
        },
        "why_it_is_here": SOURCES["noisy_simple_crystals"].rationale,
    }
    metadata_path = dataset_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest_path = write_manifest(output_dir)

    print(f"Wrote {data_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or generate q_l/q_ls-hard 3D particle datasets.",
    )
    parser.set_defaults(func=lambda _args: list_sources())
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List selected sources and access notes.")
    list_parser.set_defaults(func=lambda _args: list_sources())

    manifest_parser = subparsers.add_parser(
        "write-manifest",
        help="Write source metadata into the ignored other_data directory.",
    )
    manifest_parser.add_argument("--output-dir", type=Path, default=THIS_DIR)
    manifest_parser.set_defaults(
        func=lambda args: print(f"Wrote {write_manifest(args.output_dir.resolve())}")
    )

    globus_parser = subparsers.add_parser(
        "submit-martirossyan-globus",
        help="Submit the Martirossyan MDF dataset transfer through Globus CLI.",
    )
    globus_parser.add_argument("--destination-endpoint", help="Destination Globus endpoint UUID.")
    globus_parser.add_argument(
        "--destination-path",
        help="Destination path on that Globus endpoint.",
    )
    globus_parser.add_argument(
        "--sync-level",
        choices=("exists", "size", "mtime", "checksum"),
        default="checksum",
        help="Globus sync level for reruns.",
    )
    globus_parser.add_argument(
        "--label",
        default="autoencode_statmech Martirossyan q_ls-hard data",
        help="Globus task label.",
    )
    globus_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Globus command without submitting it.",
    )
    globus_parser.set_defaults(func=submit_martirossyan_globus)

    noisy_parser = subparsers.add_parser(
        "generate-noisy-crystals",
        help="Generate a local Haeberle-style noisy-crystal NPZ benchmark.",
    )
    noisy_parser.add_argument("--output-dir", type=Path, default=THIS_DIR)
    noisy_parser.add_argument(
        "--structures",
        default="fcc,hcp,bcc,sc,diamond",
        help="Comma-separated subset of fcc,hcp,bcc,sc,diamond.",
    )
    noisy_parser.add_argument("--cells", type=int, default=6)
    noisy_parser.add_argument("--samples-per-structure", type=int, default=8)
    noisy_parser.add_argument(
        "--noise-levels",
        default="0.00,0.03,0.06,0.09,0.12",
        help="Comma-separated Gaussian displacement sigmas in units of lattice constant.",
    )
    noisy_parser.add_argument("--lattice-constant", type=float, default=1.0)
    noisy_parser.add_argument("--seed", type=int, default=20260706)
    noisy_parser.add_argument("--filename", default="noisy_crystals.npz")
    noisy_parser.set_defaults(func=generate_noisy_crystals)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
