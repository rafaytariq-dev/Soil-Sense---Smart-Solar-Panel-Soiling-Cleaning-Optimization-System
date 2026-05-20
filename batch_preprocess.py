"""
batch_preprocess.py
-------------------
Batch-process all .jpg images in data/raw/PanelImages/ through the DIP
pipeline (white_balance -> apply_clahe -> apply_bilateral -> resize 224x224),
classify each image by its L value extracted from the filename, save it to
the appropriate class subfolder, and write a CSV label manifest.

Filename convention expected:  ..._L_<float>_I_<float>...jpg
  L < 0.20  → Clean
  L < 0.50  → Low_Dust
  L >= 0.50 → Heavy_Dust

Output layout:
  data/processed/
    Clean/
    Low_Dust/
    Heavy_Dust/
    dataset_labels.csv

Usage:
    python batch_preprocess.py

Requirements:
    data/raw/PanelImages/ must exist and contain .jpg images.
    Output directory data/processed/ is created automatically.
"""

import csv
import os
import re
import sys
import concurrent.futures
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

# ── Ensure project root is on sys.path so Vision imports work in
#    worker processes (which are spawned fresh on Windows).
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_DIR = ROOT / "Data" / "raw" / "PanelImages"
OUTPUT_DIR = ROOT / "Data" / "processed"

# Regex: matches _L_<number>_I_<number> anywhere in the filename
_LI_PATTERN = re.compile(r"_L_(\d+(?:\.\d+)?)_I_(\d+(?:\.\d+)?)", re.IGNORECASE)

# Class definitions: (label_int, label_name, upper_bound_exclusive)
# Evaluated in order; last entry has no upper bound (catch-all).
_CLASSES = [
    (0, "Clean",      lambda l: l < 0.20),
    (1, "Low_Dust",   lambda l: l < 0.50),
    (2, "Heavy_Dust", lambda _: True),
]


def _classify(l_value: float) -> tuple[int, str]:
    """Return (label_int, label_name) for a given L value."""
    for label_int, label_name, predicate in _CLASSES:
        if predicate(l_value):
            return label_int, label_name
    return 2, "Heavy_Dust"  # unreachable but satisfies type checker


def _parse_li(filename: str) -> tuple[float | None, float | None]:
    """Extract L and I float values from a filename. Returns (None, None) on failure."""
    m = _LI_PATTERN.search(filename)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


# ── Worker function ────────────────────────────────────────────────────────────
# Must be defined at module level so ProcessPoolExecutor can pickle it.
def _process_one(task: tuple[str, str]) -> tuple[str, str, str, float | None, float | None]:
    """Run the DIP pipeline on one image, classify it, and save the output.

    Args:
        task: (input_path, output_base_dir) as strings.

    Returns:
        (input_path, output_path, label_name, l_value, i_value)

    Raises:
        RuntimeError: If the pipeline fails or the file cannot be written.
    """
    from Vision.preprocessing.pipeline_runner import run_pipeline

    in_path, output_base = task
    filename = Path(in_path).name

    l_value, i_value = _parse_li(filename)

    if l_value is None:
        _, label_name = 2, "Heavy_Dust"
    else:
        _, label_name = _classify(l_value)

    out_dir = Path(output_base) / label_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / filename)

    display_img, _, _, _ = run_pipeline(in_path, skip_visualiser=True)

    ok = cv2.imwrite(out_path, display_img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {out_path}")

    return in_path, out_path, label_name, l_value, i_value


# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_tasks(source: Path, output: Path) -> list[tuple[str, str]]:
    """Return (input_path, output_base_dir) pairs for every .jpg under source."""
    return [
        (str(img_path), str(output))
        for img_path in sorted(source.rglob("*.jpg"))
    ]


def _write_csv(records: list[tuple[str, str, str, float | None, float | None]], output: Path) -> None:
    """Write dataset_labels.csv to output directory."""
    csv_path = output / "dataset_labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label_int", "label_name", "L_value", "I_value"])
        for _, out_path, label_name, l_val, i_val in records:
            label_int = next(li for li, ln, _ in _CLASSES if ln == label_name)
            writer.writerow([
                out_path,
                label_int,
                label_name,
                f"{l_val:.4f}" if l_val is not None else "",
                f"{i_val:.4f}" if i_val is not None else "",
            ])
    print(f"\nCSV manifest written → {csv_path}")


def _print_distribution(records: list[tuple]) -> None:
    """Print class distribution and flag potential imbalance."""
    counts: Counter = Counter(r[2] for r in records)
    total = sum(counts.values())
    print("\n── Class distribution ──────────────────────────────")
    for _, label_name, _ in _CLASSES:
        n = counts.get(label_name, 0)
        pct = 100.0 * n / total if total else 0.0
        bar = "█" * int(pct / 2)
        print(f"  {label_name:<12}  {n:>5} images  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':<12}  {total:>5} images")

    # Imbalance warning: flag if any class deviates > 20 pp from equal share
    if total > 0:
        equal_share = 100.0 / len(_CLASSES)
        imbalanced = [
            ln for _, ln, _ in _CLASSES
            if abs(100.0 * counts.get(ln, 0) / total - equal_share) > 20
        ]
        if imbalanced:
            print(f"\n  ⚠  Potential class imbalance detected in: {', '.join(imbalanced)}")
    print("────────────────────────────────────────────────────")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not SOURCE_DIR.exists():
        print(
            "Source directory not found. Please add the PanelImages folder "
            "manually to data/raw/ and try again."
        )
        sys.exit(0)

    tasks = _build_tasks(SOURCE_DIR, OUTPUT_DIR)

    if not tasks:
        print(f"No .jpg images found under {SOURCE_DIR}. Nothing to do.")
        sys.exit(0)

    # Create class subfolders upfront
    for _, label_name, _ in _CLASSES:
        (OUTPUT_DIR / label_name).mkdir(parents=True, exist_ok=True)

    n_workers = os.cpu_count() or 1
    print(f"Found {len(tasks):,} images  |  workers: {n_workers}  |  output: {OUTPUT_DIR}")

    failed: list[tuple[str, str]] = []
    records: list[tuple[str, str, str, float | None, float | None]] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one, t): t[0] for t in tasks}

        with tqdm(total=len(tasks), unit="img", desc="Preprocessing", dynamic_ncols=True) as bar:
            for future in concurrent.futures.as_completed(futures):
                src_path = futures[future]
                try:
                    result = future.result()
                    records.append(result)
                except Exception as exc:
                    failed.append((src_path, str(exc)))
                finally:
                    bar.update(1)

    n_ok = len(records)
    print(f"\nDone: {n_ok:,} saved, {len(failed):,} failed.")

    if records:
        _write_csv(records, OUTPUT_DIR)
        _print_distribution(records)

    if failed:
        print("\nFailed images:")
        for path, err in failed:
            print(f"  {path}\n    -> {err}")
        sys.exit(1)
