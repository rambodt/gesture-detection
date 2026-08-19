"""
ee446_data.py
Loader + windowing helpers for Edge Impulse exported data (EE 446 final project).

Handles both formats produced by:
    Edge Impulse Studio -> Data acquisition -> Export data
    -> "Original files (original file names)"

Data collected through the EI daemon is stored as CBOR, so most exports will
be .cbor. JSON is also supported. CBOR needs:  pip install cbor2

Each exported sample decodes to:
    {
      "protected": {...},
      "signature": "...",
      "payload": {
        "device_name": "...",
        "interval_ms": 10,
        "sensors": [{"name": "accX", "units": "m/s2"}, ...],
        "values": [[ax, ay, az, gx, gy, gz], ...]
      }
    }

Filenames look like:  Left1.703dofv5.cbor   ->  label = "Left1"

IMPORTANT: export with "original file names", not "uploader compatible" -
the uploader-compatible option renames files and the label is lost.

Usage:
    from ee446_data import load_export, make_windows, group_train_test_split

    recs = load_export("ei-export")
    X, y, groups, classes = make_windows(recs, TARGET_LABELS)
"""

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np

# ----------------------------------------------------------------------
# Configuration - these must match the Edge Impulse impulse settings
# ----------------------------------------------------------------------

FREQ_HZ = 100
WINDOW_MS = 1500
STRIDE_MS = 100

WINDOW_SAMPLES = int(WINDOW_MS * FREQ_HZ / 1000)   # 150
STRIDE_SAMPLES = int(STRIDE_MS * FREQ_HZ / 1000)   # 10

# Canonical axis order. The exported files carry their own sensor order,
# so we reorder each sample to match this rather than trusting the file.
AXES = ["accX", "accY", "accZ", "gyrX", "gyrY", "gyrZ"]

# Same scaling as the Raw Data block in Edge Impulse (Scale axes = 0.05).
# Accelerometer is in m/s^2 with peaks near 18, so this brings the input
# into roughly the +/-1 range that trains well.
SCALE = 0.05

# The seven abandoned gesture labels -> source task for transfer learning.
SOURCE_LABELS = ["Circle", "Square", "Triangle", "Down", "Left", "Right", "Up"]

# The five current labels -> target task.
TARGET_LABELS = ["Down1", "Idle", "Left1", "Right1", "Up1"]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _label_from_filename(path):
    """'Left1.703dofv5.cbor' -> 'Left1'  |  'Idle.705qsncm.cbor' -> 'Idle'"""
    return Path(path).name.split(".")[0]


def _sample_id_from_filename(path):
    """
    'Left1.703dofv5.cbor' -> 'Left1.703dofv5'
    Used as the group key so windows from one recording never straddle a split.
    """
    name = Path(path).name
    return re.sub(r"\.(json|cbor)$", "", name, flags=re.IGNORECASE)


def _read_one(path):
    """Parse a single exported sample (.cbor or .json). Returns None if unusable."""
    path = Path(path)

    try:
        if path.suffix.lower() == ".cbor":
            import cbor2
            with open(path, "rb") as f:
                blob = cbor2.load(f)
        else:
            with open(path, "r") as f:
                blob = json.load(f)
    except ImportError:
        raise ImportError(
            "Reading .cbor exports requires cbor2. Run:  pip install cbor2"
        )
    except Exception as e:
        print(f"  skip (unreadable): {path.name}  [{e}]")
        return None

    payload = blob.get("payload", {})
    values = payload.get("values")
    sensors = payload.get("sensors")

    if not values or not sensors:
        print(f"  skip (no values/sensors): {path.name}")
        return None

    names = [s.get("name") for s in sensors]

    # Reorder columns into canonical AXES order.
    try:
        order = [names.index(a) for a in AXES]
    except ValueError:
        print(f"  skip (axes {names} != expected {AXES}): {path.name}")
        return None

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < len(AXES):
        print(f"  skip (bad shape {arr.shape}): {path.name}")
        return None

    arr = arr[:, order]

    return {
        "label": _label_from_filename(path),
        "sample_id": _sample_id_from_filename(path),
        "device": payload.get("device_name", "unknown"),   # <-- add this
        "values": arr,
        "interval_ms": payload.get("interval_ms", 1000 / FREQ_HZ),
        "path": str(path),
    }


def load_export(root, verbose=True):
    """
    Walk an unzipped Edge Impulse export directory and load every sample.

    Works whether or not the export has training/ and testing/ subfolders -
    we re-split ourselves anyway, group-aware.

    Returns: list of record dicts.
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"No such directory: {root.resolve()}")

    files = sorted(list(root.rglob("*.cbor")) + list(root.rglob("*.json")))

    # Ignore Edge Impulse's own metadata files if present.
    files = [f for f in files if f.name not in {"info.labels", "metadata.json"}]

    if verbose:
        n_cbor = sum(1 for f in files if f.suffix.lower() == ".cbor")
        n_json = len(files) - n_cbor
        print(f"Found {len(files)} sample files under {root}  "
              f"({n_cbor} .cbor, {n_json} .json)")

    records = []
    for f in files:
        rec = _read_one(f)
        if rec is not None:
            records.append(rec)

    if verbose:
        counts = Counter(r["label"] for r in records)
        print(f"\nLoaded {len(records)} samples across {len(counts)} labels:")
        for lab in sorted(counts):
            print(f"  {lab:12s} {counts[lab]:4d}")

    return records


# ----------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------

def make_windows(records, labels, window=WINDOW_SAMPLES, stride=STRIDE_SAMPLES,
                 scale=SCALE, zero_pad=True, verbose=True):
    """
    Slice records into fixed windows, matching the Edge Impulse impulse config.

    Only records whose label is in `labels` are used, and labels are remapped
    to 0..K-1 in the order given.

    Returns:
        X       (n_windows, window, 6) float32
        y       (n_windows,) int
        groups  (n_windows,) str    - sample_id, for leakage-free splitting
        classes list[str]           - index -> label name
    """
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    X, y, groups = [], [], []
    skipped_short = 0

    for rec in records:
        if rec["label"] not in label_to_idx:
            continue

        vals = rec["values"]
        n = len(vals)

        if n < window:
            if not zero_pad:
                skipped_short += 1
                continue
            # Edge Impulse zero-pads short samples; mirror that.
            pad = np.zeros((window - n, vals.shape[1]), dtype=np.float32)
            vals = np.vstack([vals, pad])
            n = len(vals)

        for start in range(0, n - window + 1, stride):
            X.append(vals[start:start + window])
            y.append(label_to_idx[rec["label"]])
            groups.append(rec["sample_id"])

    if not X:
        raise ValueError(
            f"No windows produced. None of the loaded records matched labels={labels}. "
            "Check the label names printed by load_export()."
        )

    X = np.asarray(X, dtype=np.float32) * scale
    y = np.asarray(y, dtype=np.int32)
    groups = np.asarray(groups)

    if verbose:
        print(f"Windows: {X.shape}  (window={window}, stride={stride}, scale={scale})")
        counts = Counter(y.tolist())
        for i, lab in enumerate(labels):
            print(f"  {lab:12s} {counts.get(i, 0):5d}")
        if skipped_short:
            print(f"  ({skipped_short} samples skipped as too short)")

    return X, y, groups, list(labels)


# ----------------------------------------------------------------------
# Group-aware splitting
# ----------------------------------------------------------------------

def group_train_test_split(X, y, groups, test_size=0.2, seed=42, verbose=True):
    """
    Split so that all windows from one recording land on the same side.

    This is what prevents the window-overlap leakage that inflated the
    Edge Impulse validation accuracy to 100% while held-out test was 97.5%.
    """
    rng = np.random.default_rng(seed)

    uniq = np.unique(groups)
    rng.shuffle(uniq)

    n_test = max(1, int(round(len(uniq) * test_size)))
    test_groups = set(uniq[:n_test].tolist())

    test_mask = np.array([g in test_groups for g in groups])
    train_mask = ~test_mask

    if verbose:
        print(f"Recordings: {len(uniq)}  ->  train {len(uniq) - n_test} / test {n_test}")
        print(f"Windows:    train {train_mask.sum()} / test {test_mask.sum()}")

    return (X[train_mask], y[train_mask],
            X[test_mask], y[test_mask])


def summarize(y, classes, name="set"):
    """Print per-class window counts."""
    counts = Counter(np.asarray(y).tolist())
    total = sum(counts.values())
    print(f"{name}: {total} windows")
    for i, lab in enumerate(classes):
        c = counts.get(i, 0)
        pct = 100.0 * c / total if total else 0.0
        print(f"  {lab:12s} {c:5d}  ({pct:4.1f}%)")




def split_by_device(records, device_substr):
    """Return (matching, non-matching) record lists."""
    sub = device_substr.lower()
    hit, miss = [], []
    for r in records:
        (hit if sub in r.get("device", "").lower() else miss).append(r)
    return hit, miss