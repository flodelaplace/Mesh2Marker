"""Headless CLI for Mesh2Marker (pure numpy + stdlib, no bpy).

Subcommands:
  generate  one subject -> one repositioned .osim
  batch     many subjects (a betas CSV, or a dir of subject .npz) -> one .osim each

Run via ``python -m mesh2marker ...`` or the ``mesh2marker`` console script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .generate import generate_subject_osim, sample_from_betas
from .io import read as read_correspondence
from .mhr import MhrSample, load_mhr_npz
from .morph import load_shape_basis


def _correspondence_map(path: str) -> dict[str, list[int]]:
    corr = read_correspondence(path)
    return {m.name: list(m.mhr_vertices) for m in corr.markers}


def _parse_betas(betas: str | None, betas_file: str | None) -> np.ndarray:
    if betas_file:
        if betas_file.endswith(".npy"):
            return np.asarray(np.load(betas_file), dtype=float).reshape(-1)
        loaded = json.loads(Path(betas_file).read_text())
        return np.asarray(loaded, dtype=float).reshape(-1)
    if betas:
        return np.array([float(x) for x in betas.split(",") if x.strip() != ""])
    return np.zeros(0)  # all-zero -> mean shape (morph zero-pads)


def _sample_for_generate(args) -> MhrSample:
    if args.npz:
        return load_mhr_npz(args.npz)
    if not args.basis:
        raise SystemExit("generate needs either --npz or --basis [+ --betas]")
    basis = load_shape_basis(args.basis)
    return sample_from_betas(basis, _parse_betas(args.betas, args.betas_file))


def cmd_generate(args) -> int:
    correspondence = _correspondence_map(args.map)
    sample = _sample_for_generate(args)
    locations = generate_subject_osim(args.osim, args.output, correspondence, sample)
    print(f"Wrote {len(locations)} marker locations -> {args.output}")
    return 0


def cmd_batch(args) -> int:
    correspondence = _correspondence_map(args.map)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    if args.npz_dir:
        for npz_path in sorted(Path(args.npz_dir).glob("*.npz")):
            sample = load_mhr_npz(str(npz_path))
            dst = out_dir / f"{npz_path.stem}.osim"
            generate_subject_osim(args.osim, str(dst), correspondence, sample)
            written.append(str(dst))
    else:
        if not (args.basis and args.betas_csv):
            raise SystemExit("batch needs --npz-dir, or both --basis and --betas-csv")
        basis = load_shape_basis(args.basis)
        with open(args.betas_csv, newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0].strip():
                    continue
                try:
                    betas = np.array([float(x) for x in row[1:]])
                except ValueError:
                    continue  # header or non-numeric row
                sample = sample_from_betas(basis, betas)
                dst = out_dir / f"{row[0].strip()}.osim"
                generate_subject_osim(args.osim, str(dst), correspondence, sample)
                written.append(str(dst))

    print(f"Batch: {len(written)} subjects written to {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mesh2marker", description="Generate subject-specific .osim files."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="one subject -> one repositioned .osim")
    gen.add_argument("--osim", required=True, help="source .osim model")
    gen.add_argument("--map", required=True, help="correspondence JSON file")
    gen.add_argument("--output", required=True, help="destination .osim")
    gen.add_argument("--npz", help="subject MHR mesh .npz")
    gen.add_argument("--basis", help="shape basis .npz (with --betas)")
    gen.add_argument("--betas", help="comma-separated shape coefficients")
    gen.add_argument("--betas-file", help="betas as .npy or .json")
    gen.set_defaults(func=cmd_generate)

    batch = sub.add_parser("batch", help="many subjects -> one .osim each")
    batch.add_argument("--osim", required=True, help="source .osim model")
    batch.add_argument("--map", required=True, help="correspondence JSON file")
    batch.add_argument("--output-dir", required=True, help="output directory")
    batch.add_argument("--basis", help="shape basis .npz (with --betas-csv)")
    batch.add_argument(
        "--betas-csv", help="CSV: id,b0,b1,... one subject per row"
    )
    batch.add_argument("--npz-dir", help="directory of subject .npz files")
    batch.set_defaults(func=cmd_batch)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
