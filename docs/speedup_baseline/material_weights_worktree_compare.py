"""Compare material-weight calculation before/after the approved fast path."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


BASELINE_COMMIT = "73aa899afac2779af3ae6c911aeaa6c54549df86"
N_HISTORIES = 200_000
SEED = 20260728
N_CHUNKS = 8

CHILD_CODE = r"""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

root, operation, output, case, batch_size = sys.argv[1:]
sys.path.insert(0, root)
import chatcarlo.kernel as kernel
import chatcarlo.tally_njit as tally_njit
from chatcarlo.tally import VoxelGrid

print("kernel_file=" + str(Path(kernel.__file__).resolve()), flush=True)

def make_case(case_name):
    grid = VoxelGrid(np.array([-5.01, -50.01, -50.01]), (6, 51, 51), 2.0)
    if case_name == "water":
        names = ["water", "air"]
        boxes = [{"center": (0.0, 0.0, 0.0),
                  "size_cm": (10.0, 100.0, 100.0), "material": "water"}]
    else:
        names = ["water", "lead", "air"]
        boxes = [
            {"center": (0.0, 0.0, 0.0),
             "size_cm": (10.0, 100.0, 100.0), "material": "water"},
            {"center": (0.0, 0.0, 0.0),
             "size_cm": (0.02, 100.0, 100.0), "material": "lead"},
        ]
    tables = kernel.bake_scene_materials(names)
    geom = kernel.bake_box_scene(
        boxes, background="air", tables=tables, bbox_margin_cm=0.01)
    return tables, geom, grid

def execute(case_name, histories, selected_batch_size, capture_counts=False):
    tables, geom, grid = make_case(case_name)
    counts = np.zeros(len(tables.material_names), dtype=np.int64)
    original_transport = kernel._run_batch_scalar_tally
    if capture_counts:
        def transport_spy(*args, **kwargs):
            result = original_transport(*args, **kwargs)
            segment_counts = result[-2]
            segment_materials = args[-2]
            for chunk, count in enumerate(segment_counts):
                counts[:] += np.bincount(
                    segment_materials[chunk, :int(count)],
                    minlength=len(counts))[:len(counts)]
            return result
        kernel._run_batch_scalar_tally = transport_spy
    try:
        kernel.run_dose_grid(
            tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0),
            histories, 20260728, grid, batch_size=selected_batch_size,
            n_chunks=8, fluorescence_enabled=True,
            max_segments_per_history=16, use_njit_dda=True)
    finally:
        kernel._run_batch_scalar_tally = original_transport
    return grid, tables.material_names, counts

if operation == "correctness":
    grid, names, counts = execute(
        case, 200000, int(batch_size), capture_counts=(case == "multi"))
    np.savez(output, kerma_keV=grid.kerma_keV,
             h10_track_pSv_cm3=grid.h10_track_pSv_cm3,
             material_names=np.asarray(names), segment_counts=counts)
else:
    execute("water", 2000, 2000)
    tables, geom, grid = make_case("water")
    dda_seconds = [0.0]
    original_dda = tally_njit.accumulate_track_length_multi_njit
    def timed_dda(*args, **kwargs):
        start = time.perf_counter()
        result = original_dda(*args, **kwargs)
        dda_seconds[0] += time.perf_counter() - start
        return result
    tally_njit.accumulate_track_length_multi_njit = timed_dda
    start = time.perf_counter()
    kernel.run_dose_grid(
        tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0),
        200000, 20260728, grid, batch_size=200000, n_chunks=8,
        fluorescence_enabled=True, max_segments_per_history=16,
        use_njit_dda=True)
    e2e = time.perf_counter() - start
    Path(output).write_text(json.dumps({
        "e2e": e2e, "dda": dda_seconds[0],
        "residual": e2e - dda_seconds[0],
        "checksum": float(grid.kerma_keV.sum() + grid.h10_track_pSv_cm3.sum()),
    }))
"""


def _run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def _child(root: Path, operation: str, output: Path, case="water",
           batch_size=200_000):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    result = _run(
        [sys.executable, "-c", CHILD_CODE, str(root), operation, str(output),
         case, str(batch_size)],
        cwd=root, env=environment, capture_output=True)
    print(result.stdout, end="")
    expected = (root / "chatcarlo" / "kernel.py").resolve()
    marker = f"kernel_file={expected}"
    if marker not in result.stdout:
        raise AssertionError(
            f"child imported an unexpected kernel; expected marker {marker}")
    return result


def _stats(values):
    return min(values), float(np.median(values)), max(values)


def _compare_arrays(label, old_file, new_file):
    with np.load(old_file) as old, np.load(new_file) as new:
        for name in ("kerma_keV", "h10_track_pSv_cm3"):
            a, b = old[name], new[name]
            same = (a.shape == b.shape and a.dtype == b.dtype
                    and np.array_equal(a, b))
            print(
                f"{label}/{name}: {'PASS' if same else 'FAIL'} "
                f"shape={a.shape} dtype={a.dtype} "
                f"max_abs={float(np.max(np.abs(a - b))):.17g}")
            if not same:
                raise AssertionError(f"{label}/{name} differs")
        if label == "multi_material":
            for arm, data in (("old", old), ("new", new)):
                names = data["material_names"].tolist()
                counts = data["segment_counts"]
                print(f"{label}/{arm} segment_counts="
                      + ", ".join(f"{n}:{int(c)}" for n, c in zip(names, counts)))
                if np.any(counts <= 0):
                    raise AssertionError(f"{arm}: not every material has a segment")


def correctness(old_root, new_root, scratch):
    cases = (
        ("one_batch_200000", "water", 200_000),
        ("remainder_batches_60000", "water", 60_000),
        ("multi_material", "multi", 200_000),
    )
    for label, case, batch_size in cases:
        old_file = scratch / f"{label}_old.npz"
        new_file = scratch / f"{label}_new.npz"
        _child(old_root, "correctness", old_file, case, batch_size)
        _child(new_root, "correctness", new_file, case, batch_size)
        _compare_arrays(label, old_file, new_file)
    print("correctness overall: PASS")


def timing(old_root, new_root, scratch):
    rows = []
    for repetition, order in enumerate((("old", "new"), ("new", "old")) * 4, 1):
        row = {}
        for arm in order:
            output = scratch / f"timing_{repetition}_{arm}.json"
            _child(old_root if arm == "old" else new_root, "timing", output)
            row[arm] = json.loads(output.read_text())
        if row["old"]["checksum"] != row["new"]["checksum"]:
            raise AssertionError("timing-arm grid checksum mismatch")
        row["e2e_ratio"] = row["old"]["e2e"] / row["new"]["e2e"]
        row["residual_ratio"] = (
            row["old"]["residual"] / row["new"]["residual"])
        rows.append(row)
        print(
            f"rep={repetition} order={'/'.join(order)} "
            f"old_e2e={row['old']['e2e']:.9f} "
            f"old_dda={row['old']['dda']:.9f} "
            f"old_residual={row['old']['residual']:.9f} "
            f"new_e2e={row['new']['e2e']:.9f} "
            f"new_dda={row['new']['dda']:.9f} "
            f"new_residual={row['new']['residual']:.9f} "
            f"e2e_ratio={row['e2e_ratio']:.6f} "
            f"residual_ratio={row['residual_ratio']:.6f}")
    for arm in ("old", "new"):
        for metric in ("e2e", "dda", "residual"):
            lo, median, hi = _stats([row[arm][metric] for row in rows])
            print(f"{arm}_{metric}: min={lo:.9f} median={median:.9f} max={hi:.9f}")
    print("per-repetition residual old/new ratios: "
          + ", ".join(f"{row['residual_ratio']:.6f}" for row in rows))
    print("per-repetition E2E old/new ratios: "
          + ", ".join(f"{row['e2e_ratio']:.6f}" for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("correctness", "timing"), required=True)
    args = parser.parse_args()
    new_root = Path(__file__).resolve().parents[2]
    print(f"baseline_commit={BASELINE_COMMIT}")
    print(f"new_root={new_root}")
    with tempfile.TemporaryDirectory(prefix="material-weights-") as temp:
        scratch = Path(temp)
        worktree_admin = scratch / "worktree-admin"
        old_root = scratch / "old-worktree"
        _run(["git", "clone", "--no-checkout", "--no-local", str(new_root),
              str(worktree_admin)])
        _run(["git", "worktree", "add", "--detach", str(old_root), BASELINE_COMMIT],
             cwd=worktree_admin)
        try:
            print(f"old_root={old_root}")
            if args.mode == "correctness":
                correctness(old_root, new_root, scratch)
            else:
                timing(old_root, new_root, scratch)
        finally:
            _run(["git", "worktree", "remove", "--force", str(old_root)],
                 cwd=worktree_admin)
            print("old worktree removed")


if __name__ == "__main__":
    main()
