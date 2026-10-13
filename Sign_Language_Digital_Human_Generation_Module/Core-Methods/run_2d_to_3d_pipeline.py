"""Run five preprocessing stages from OpenPose JSON and export 3D frames."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path


BODY_INDICES = tuple(range(8))
HAND_INDICES = tuple(range(21))
JOINT_COUNT = len(BODY_INDICES) + 2 * len(HAND_INDICES)
FRAME_PATTERN = re.compile(r"_(\d+)_keypoints\.json$", re.IGNORECASE)


def frame_index(path: Path) -> int | None:
    """Extract an OpenPose frame index when it is present in the filename."""
    match = FRAME_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def select_keypoints(person: dict, field: str, indices: tuple[int, ...]) -> list[float]:
    """Select x, y, and confidence values for fixed OpenPose joint indices."""
    values = person.get(field) or []
    selected: list[float] = []
    for index in indices:
        offset = 3 * index
        selected.extend(values[offset:offset + 3] if len(values) >= offset + 3 else (0.0, 0.0, 0.0))
    return selected


def collect_sequences(json_dir: Path, np):
    """Group JSON frames by parent folder and retain their temporal order."""
    files = sorted(json_dir.rglob("*.json"))
    if not files:
        raise ValueError(f"No JSON files were found in {json_dir}")

    groups: dict[str, list[Path]] = {}
    for path in files:
        relative_parent = path.parent.relative_to(json_dir)
        if relative_parent.parts:
            key = str(relative_parent).replace("\\", "/")
        else:
            match = FRAME_PATTERN.search(path.name)
            key = path.name[:match.start()] if match else json_dir.name
        groups.setdefault(key, []).append(path)

    sequences = {}
    for key, paths in groups.items():
        if all(frame_index(path) is not None for path in paths):
            paths.sort(key=lambda path: (frame_index(path), path.name))
        else:
            paths.sort(key=lambda path: path.name)

        rows = []
        for path in paths:
            with path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            people = data.get("people") or []
            if not people:
                continue
            person = people[0]
            points = (
                select_keypoints(person, "pose_keypoints_2d", BODY_INDICES)
                + select_keypoints(person, "hand_left_keypoints_2d", HAND_INDICES)
                + select_keypoints(person, "hand_right_keypoints_2d", HAND_INDICES)
            )
            if points[0] != 0.0:
                rows.append(points)
        if rows:
            sequences[key] = np.asarray(rows, dtype="float32")

    if not sequences:
        raise ValueError("All JSON groups were empty or had an invalid root joint.")
    return sequences


def load_filter_module(pipeline_dir: Path, np):
    """Load the supplied stage-two filter and inject its original runtime globals."""
    required = [
        "pipeline_demo_02_filter.py", "skeletalModel.py", "pose2D.py",
        "pose2Dto3D.py", "pose3D.py",
    ]
    missing = [name for name in required if not (pipeline_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pipeline modules in {pipeline_dir}: {', '.join(missing)}")

    sys.path.insert(0, str(pipeline_dir))
    try:
        module = importlib.import_module("pipeline_demo_02_filter")
    except ModuleNotFoundError as exc:
        raise ImportError(f"Pipeline dependency is missing: {exc.name}") from exc
    module.dtype = "float32"
    module.randomNubersGenerator = np.random.RandomState(1234)
    return module


def load_deltas_module(pipeline_dir: Path):
    """Load the original stage-three delta connection map and calculation."""
    name = "pipeline_demo_03_deltas.py"
    if not (pipeline_dir / name).is_file():
        raise FileNotFoundError(f"Missing pipeline module: {pipeline_dir / name}")
    return importlib.import_module("pipeline_demo_03_deltas")


def write_h5(path: Path, groups: dict, h5py, attributes: dict | None = None) -> None:
    """Write grouped sequences to an HDF5 file."""
    with h5py.File(path, "w") as handle:
        if attributes:
            for name, value in attributes.items():
                handle.attrs[name] = value
        for key, data in groups.items():
            handle.create_dataset(key, data=data, dtype=data.dtype)


def normalize_deltas(groups: dict, np):
    """Divide deltas by the global mean of per-feature standard deviations."""
    values = np.concatenate(list(groups.values()), axis=0).astype("float64")
    sigma = float(np.std(values, axis=0).mean())
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Normalization sigma is zero or non-finite; provide varied valid frames.")
    return {key: data.astype("float64") / sigma for key, data in groups.items()}, sigma


def compute_mu(groups: dict, np):
    """Compute the stage-five mean pose over all normalized delta frames."""
    values = np.concatenate(list(groups.values()), axis=0)
    return values.mean(axis=0, keepdims=True)


def export_3d_frames(groups: dict, frame_dir: Path, np) -> int:
    """Write estimated 3D coordinates before the lossy delta transformation."""
    count = 0
    for key, data in groups.items():
        if data.ndim != 2 or data.shape[1] != JOINT_COUNT * 3:
            raise ValueError(f"Unexpected 3D shape for {key}: {data.shape}")
        destination = frame_dir / key
        destination.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(data):
            coordinates = np.asarray(frame).reshape(JOINT_COUNT, 3)
            content = " ".join(f"{value:.5f}" for value in coordinates.reshape(-1))
            (destination / f"frame_{index:04d}.txt").write_text(content + "\n", encoding="utf-8")
            count += 1
    return count


def run_pipeline(json_dir: Path, pipeline_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Run JSON packing, 3D filtering, deltas, normalization, and mean pose."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise ImportError("Install numpy and h5py before running the pipeline.") from exc

    if not json_dir.is_dir():
        raise FileNotFoundError(f"OpenPose JSON directory does not exist: {json_dir}")
    if not pipeline_dir.is_dir():
        raise FileNotFoundError(f"Pipeline directory does not exist: {pipeline_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    filter_module = load_filter_module(pipeline_dir, np)
    deltas_module = load_deltas_module(pipeline_dir)
    filenames = [
        "keypoints-01-raw.h5", "keypoints-02-filter.h5",
        "keypoints-03-deltas.h5", "keypoints-04-deltas-norm.h5",
    ]
    paths = [output_dir / name for name in filenames]

    raw = collect_sequences(json_dir, np)
    write_h5(paths[0], raw, h5py, {"joint_count": JOINT_COUNT, "stage": "raw_2d"})
    print(f"Stage 1/5: {paths[0]}")

    filtered = {}
    for key, data in raw.items():
        result = np.asarray(filter_module.use_filter(data), dtype="float32")
        if result.ndim != 2 or result.shape[1] != JOINT_COUNT * 3 or result.shape[0] == 0:
            raise ValueError(f"Unexpected 3D filter output for {key}: {result.shape}")
        filtered[key] = result
    write_h5(paths[1], filtered, h5py, {"joint_count": JOINT_COUNT, "stage": "estimated_3d"})
    print(f"Stage 2/5: {paths[1]}")

    deltas = {key: np.asarray(deltas_module.computeDeltas(data), dtype="float32")
              for key, data in filtered.items()}
    if any(data.ndim != 2 or data.shape[1] != 98 for data in deltas.values()):
        raise ValueError("Unexpected delta feature width; expected 49 planar links = 98 values.")
    write_h5(paths[2], deltas, h5py, {"stage": "planar_deltas"})
    print(f"Stage 3/5: {paths[2]}")

    normalized, sigma = normalize_deltas(deltas, np)
    write_h5(paths[3], normalized, h5py, {"stage": "normalized_planar_deltas", "sigma": sigma})
    print(f"Stage 4/5: {paths[3]} (sigma={sigma:.6g})")

    mu = compute_mu(normalized, np)
    mu_path = output_dir / "mu.txt"
    mu_path.write_text("\t".join(f"{value:e}" for value in mu[0]) + "\n", encoding="utf-8")
    print(f"Stage 5/5: {mu_path}")

    frame_dir = output_dir / "frame_data"
    count = export_3d_frames(filtered, frame_dir, np)
    print(f"Exported {count} estimated 3D frames to {frame_dir}")
    return {"raw": paths[0], "filtered_3d": paths[1], "deltas": paths[2],
            "normalized": paths[3], "mu": mu_path, "frames": frame_dir}


def main() -> int:
    """Provide a single command-line entry point for all five stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_dir", help="Directory containing OpenPose 2D JSON frames.")
    parser.add_argument(
        "--pipeline-dir",
        default=str(Path(__file__).resolve().parent.parent / "Text‑to‑Sign-Language" / "pose_pipeline"),
        help="Directory containing the original pipeline and pose estimator modules.",
    )
    parser.add_argument("--output-dir", default="pipeline_output", help="Directory for all generated results.")
    args = parser.parse_args()
    try:
        run_pipeline(Path(args.json_dir).expanduser().resolve(),
                     Path(args.pipeline_dir).expanduser().resolve(),
                     Path(args.output_dir).expanduser().resolve())
        return 0
    except (FileNotFoundError, ValueError, ImportError, OSError) as error:
        print(f"Pipeline error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
