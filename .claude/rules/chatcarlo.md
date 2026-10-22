---
paths:
  - "chatcarlo/**"
  - "tests/**"
  - "examples/**"
  - "scripts/fetch_*.py"
  - "docs/egs5_crosscheck/**"
  - "docs/speedup_baseline/**"
  - "docs/gpu_speedup/**"
  - "docs/plan_*.md"
---

# ChatCarlo package — architecture and design rules

Moved from CLAUDE.md's "Architecture" section on 2026-10-19 (headings kept). The data-provenance table, the R
interpretation guide and "Known sharp edges" stay in CLAUDE.md because they apply to every run.

## Transport

**Transport is not voxelized.** Geometry stays as analytic primitives (box/cylinder/sphere in
[geometry.py](../../chatcarlo/geometry.py)); each photon steps to the next material boundary by an analytic ray/primitive
intersection distance ([transport.py](../../chatcarlo/transport.py)). This is analytic surface tracking, not Woodcock
delta-tracking: each segment has one homogeneous material, so μ is constant along a step and no virtual-collision
rejection is needed. Overlapping objects resolve by list order (later wins); open space is `background` (default air).

**Module layout around the kernel**: `transport.py` is only the transport loop + `run_transport`. Spectrum generation
(SpekPy/Kramers, heel off-axis spectra) is in [spectrum.py](../../chatcarlo/spectrum.py); source/field sampling and the mAs
photon-count calibration in [source.py](../../chatcarlo/source.py); interaction angle/energy sampling in
[physics.py](../../chatcarlo/physics.py); trajectory recording for `trace` in [trajectory.py](../../chatcarlo/trajectory.py);
dose-map conversion and the non-physical-max warnings in [diagnostics.py](../../chatcarlo/diagnostics.py); the terminal
planar detector tally (primary/scatter discrimination) in [detector.py](../../chatcarlo/detector.py) — plan:
[docs/ai/plans/2026-08-03-scatter-phase0-detector-tally.md](../../docs/ai/plans/2026-08-03-scatter-phase0-detector-tally.md).
`detector.py` stays here as a ChatCarlo feature even though the scatter-correction research moved to `~/Projects/mcscatter`
(see CLAUDE.md); it serves that project as an *independent* cross-check of MC-GPU's image tally. Do not delete it.

**Experimental parallel implementation**: [kernel.py](../../chatcarlo/kernel.py) is a from-scratch Numba per-history
scalar kernel (Phase B of [docs/plan_chatcarlo_speedup_post_egs5.md](../../docs/plan_chatcarlo_speedup_post_egs5.md);
box geometry only). Its dose grid uses the Numba scalar DDA in [tally_njit.py](../../chatcarlo/tally_njit.py) by default,
with the audited NumPy `tally.accumulate_track_length_multi` kept as the `use_njit_dda=False` reference/fallback.
`transport.py` remains the production path and the permanent reference that `kernel.py` is **statistically**
cross-checked against (MT19937 vs PCG64, so not bit-identity; see the plan doc's "Phase Bの検証戦略"). Before extending
or duplicating transport logic, check whether it already exists in `kernel.py`. `run --engine kernel` is an opt-in
Phase 1 adapter (box-only, monochromatic, parallel field, one effective worker; source origins via
`sample_source_photons`; same batch R/SEM tracking as numpy).

**GPU**: a measured ceiling for an existing diagnostic-energy GPU code (unmodified DIDSR/MCGPUv1.3_PCD on a Colab T4) is in
[docs/gpu_speedup/route1_mcgpu/RESULTS.md](../../docs/gpu_speedup/route1_mcgpu/RESULTS.md) (headline basis: vs EGS5's
single-threaded `-O` build, user decision 2026-08-13; EGS5 throughput in
[docs/egs5_crosscheck/speed_comparison/RESULTS.md](../../docs/egs5_crosscheck/speed_comparison/RESULTS.md)). It is an
upper-bound reference from another code, not a promise for this one. Route 2 (CUDA port of `kernel.py`) was scoped, not
implemented: [docs/gpu_speedup/route2_scoping/RESULTS.md](../../docs/gpu_speedup/route2_scoping/RESULTS.md) — every scene
under `examples/` is blocked for several reasons at once (cylinder/sphere, polychromatic `kvp`, divergent `rect`,
`mas`/`ctdi_vol_mGy`/`heel_effect`/`rotation`), and a heightfield primitive would be needed for the scatter line.

**GPU engine G1 (2026-10-21, implemented)**: `chatcarlo/gpu/` — one C++ history function (ported from `kernel.py`'s
diagnostic physics) built as a CPU twin (clang, ctypes) and a CUDA kernel (CuPy NVRTC), Philox per history, host-side
source sampling. G4 CLI: `run --engine gpu --gpu-device cpu|colab` (Colab requires `preview` → `approve RUN_DIR` → `run`; detector images: NPZ/JSON/PNG; receipts carry the observed native-call arguments, and CUDA source directions are checked against an independent replay on the same VM because float rounding differs between Mac and VM — G4 results: docs/validation/gpu/g4/RESULTS.md). API: `run_transport(..., engine="gpu", gpu_device="cpu"|"cuda", source_seed=...)` (seed required;
box-only, `kvp`/`spectrum`, `rect`/`cone`/`parallel`, plane detector; no voxel tally, `mas`, heel, rotation,
cylinder/sphere — `gpu_engine_compatible` refuses with a reason). Colab runs go through `chatcarlo.gpu.colab_driver`
(hash-verified bundles and receipts). Gates: CPU twin vs `transport.py` statistically (51 quantities |z|≤3), T4 GPU vs
CPU twin within rounding: [docs/validation/gpu/g1/RESULTS.md](../../docs/validation/gpu/g1/RESULTS.md). NVRTC has no
host `<cmath>` macros (`INFINITY`): CPU-only gates cannot catch GPU compile errors. G2–G5: parent plan
`docs/ai/plans/2026-10-21-chatcarlo-gpu-engine.md`.

**3D dose overlay (2026-10-22)**: `preview --dose X.npz [--quantity h10|dose]` (`chatcarlo/dose3d.py`) embeds a thinned subset of an existing `.npz` into the preview HTML; colours use the turbo map (low blue, high red; changed from inferno on user request 2026-10-22; initial range = 1st–99th percentile of reliable voxels, above H drawn in the top colour), voxels failing the R/batch rule are grey, near-source air is hidden, and it refuses `--engine gpu --gpu-device colab`. Verification: docs/validation/dose3d/RESULTS.md.

**Colab CLI runs (all GPU stages, not only MV)**: the operating rules in [mv.md](mv.md) "Which code to use" apply here too — poll the VM with `colab exec` **every 10 minutes** (user decision 2026-10-21), collect finished results at every poll, always `colab stop`. A new launcher must reuse `chatcarlo.gpu.colab_driver` (bundle split, `_bootstrap`, `verify`) and bound every `colab` call with a timeout; run an independent watchdog on the launcher's heartbeat/log so a stuck launcher is noticed (G2, 2026-10-22: a hung `colab exec` froze polling unnoticed for 2 h 20 min while the VM had already failed).

## Physics

Photoelectric / Compton / Rayleigh. **Bound Compton**: free-electron Klein-Nishina via Kahn rejection, then an extra
S(Z,q)/Z rejection from `xraylib.SF_Compt`; compounds by mass-fraction-weighted element pick. **Rayleigh**: atomic form
factor F(Z,q) via `xraylib.FF_Rayl`, element picked the same way; angle by inverse-transform sampling on a cumulative
F(Z,q)² table in x≡q², then a (1+cos²θ)/2 rejection (≥50% acceptance, the same two-stage scheme as EGS5;
[docs/plan_rayleigh_compton_importance_sampling.md](../../docs/plan_rayleigh_compton_importance_sampling.md)).
Electron range is neglected (kerma approximation, local absorption), except that photoelectric absorption samples K-shell
fluorescence (`sample_fluorescence` in physics.py; K-shell only, no cascade/L-shell; lines below 5 keV absorbed
locally). An emitted photon continues isotropically at the line energy. `physics.fluorescence` in scene.yaml (default
`true`); off reproduces local absorption. Design and checks: [docs/plan_fluorescence.md](../../docs/plan_fluorescence.md).
Tests: [tests/test_transport.py](../../tests/test_transport.py) (Beer-Lambert `exp(-μt)`),
[tests/test_fluorescence.py](../../tests/test_fluorescence.py) (K-edge data vs xraylib, energy conservation on/off,
emission rate vs K-shell fraction × ω_K).

## Dose/H*(10) tallying

Tallying is separate from transport. [tally.py](../../chatcarlo/tally.py)'s `VoxelGrid` is a uniform scoring grid
independent of the geometry. Two estimators are cross-validated in [tests/test_tally.py](../../tests/test_tally.py): a
collision estimator (`energy_deposited`) and a track-length kerma estimator. H*(10) is fluence-based (not kerma),
from the track-length integral normalised by voxel volume (`VoxelGrid.h10_map_pSv`).
`accumulate_track_length` computes the **exact analytic overlap length of each flight segment with every voxel it
crosses** (3D DDA generalising the audited single-box `_segment_box_overlap_cm`) — no sampling, no discretisation, no
boundary-phase artifact. Implementation notes that must not regress (all bit-identical to commit `0c97ab4`):
`accumulate_track_length_multi` shares one traversal between kerma and H*(10); `_segment_grid_traversal_accumulate`
chunks segments so no sort exceeds `_CHUNK_TARGET_INTERSECTIONS` (1,000,000) and runs each chunk's `np.add.at`
immediately (peak memory); `_argsort_within_segment` replaces `np.lexsort` with a single-key `np.argsort`
(key = segment_id + 0.5×fractional position; a fuzz test in `tests/test_tally.py` covers the boundary-collision case the
0.5 scaling exists for). History and measurements: [docs/plan_tally_speedup.md](../../docs/plan_tally_speedup.md) and
[docs/speedup_baseline/tally_exact_resolution_growth.txt](../../docs/speedup_baseline/tally_exact_resolution_growth.txt)
(single measurement script: `docs/speedup_baseline/tally_speedup_timing.py`).
Before fluorescence was modeled the two estimators disagreed by design in high-Z materials (NIST μen/ρ already subtracts
the mean fluorescence escape); with it, the lead track-length/collision ratio went from ~0.38 to ~0.92 (plan_fluorescence).

## Units and calibration

Relative output is `Gy/history` / `pSv/history`. With `source.mas`, `photon_count_through_field` (source.py) uses
SpekPy's absolute fluence for the real photon count through the field, and per-history values are scaled by that count
(not divided again by `n_histories` — see the mAs double-division bug in [docs/lessons_learned.md](../../docs/lessons_learned.md)).

## Statistical uncertainty

First-class output ([docs/plan_statistical_uncertainty.md](../../docs/plan_statistical_uncertainty.md)). Batch statistics
(batch ≡ transport batch of `batch_size` histories) give R = SEM/mean per voxel and per material, on by default
(`track_uncertainty=True`), off with `run --no-uncertainty`. A snapshot-diff accumulator keeps totals bit-identical with
tracking on or off (it only adds `kerma_sum2`/`h10_sum2`/`n_batches_hit`). `run --dose-grid` prints R and the
contributing-batch count next to the max values plus a grid summary; `--dose-out` `.npz` carries
`rel_err_dose`/`rel_err_h10`/`sem_*`/`n_batches`/`n_batches_hit`; `plot --quantity relerr-dose`/`relerr-h10` renders it
(linear 0–0.5, own colormap, zero-batch voxels masked grey). With the default `-n 1e5` and `batch_size` 200,000 there is
M=1 batch and R comes back as an actionable "increase -n or lower --batch-size" message — intentional, not a bug to fix
by raising the default n_histories.
