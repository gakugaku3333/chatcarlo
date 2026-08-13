# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Codex併用時の計画運用

`docs/ai/plans/` に `状態: approved` の計画ファイルがある場合、実装はその「対象範囲」「受入条件」「テストコマンド」に厳密に従うこと。計画に書かれていない変更をしない。計画の書き方は [docs/ai/PLAN_TEMPLATE.md](docs/ai/PLAN_TEMPLATE.md) 参照。

## 報告時のルール

作業完了時にユーザーへ報告する際は、進捗の説明だけで終わらせず、成果物そのものを
提示する（生成したHTML/PNG/npzファイルを`SendUserFile`等で送る、重要な数値結果は
本文に具体的に書く、など）。「〜を実装しました」で終わらせて中身を見せないのは
避けること。

## What this is

ChatCarlo is a Monte Carlo photon transport code for diagnostic X-ray energies (10–150 keV). Scenes are declared
in `scene.yaml` — the system is designed for an AI (Claude Code) to write and iterate on scene files declaratively,
then drive validation/preview/run non-interactively via the CLI. Current status is research/education only: doses
and H*(10) must be cross-checked against an established code (EGS5 — PHITS was considered but was not adopted;
see [docs/plan_egs5_crosscheck.md](docs/plan_egs5_crosscheck.md)) before being used for real patient-dose or
shielding decisions (see the warning banner in [README.md](README.md)).

## Commands

```bash
# setup (venv is project-local, per the parent Projects/CLAUDE.md rule — don't pip install globally)
python3 -m venv .venv
.venv/bin/pip install numpy pyyaml matplotlib xraylib pytest spekpy scipy numba

# validate a scene (physical sanity checks, not just schema)
.venv/bin/python -m chatcarlo validate examples/chest_room.yaml

# 3D geometry preview -> self-contained HTML (no external deps)
.venv/bin/python -m chatcarlo preview examples/chest_room.yaml -o preview.html

# cross-section curves
.venv/bin/python -m chatcarlo xs water bone lead -o xs.png

# run transport (prints per-material absorbed energy, absorbed/escaped fractions)
.venv/bin/python -m chatcarlo run examples/chest_room.yaml -n 1e6 --seed 42

# same, plus voxel absorbed-dose/H*(10) tally written to .npz
.venv/bin/python -m chatcarlo run examples/chest_room.yaml -n 1e6 --seed 42 \
    --dose-grid --resolution 5 --dose-out dose.npz

# multiprocess parallel run for large n (shielding-scale n=1e7+; --workers 0 = auto-detect cpu count).
# Same seed with a different --workers value does NOT reproduce bit-for-bit (independent RNG
# streams per worker via SeedSequence.spawn), only statistically — see docs/plan_phase3_parallel.md.
# Worker startup (~0.8s each: reimport + rebuild cross-section tables) makes this net-negative
# below roughly n=1e6; it pays off at shielding-evaluation scale (n=1e7+, ~3x at workers=4).
.venv/bin/python -m chatcarlo run examples/chest_room.yaml -n 1e7 --seed 42 --workers 4

# opt-in Numba kernel engine (Phase 1: box-only + monochromatic source.spectrum + parallel field +
# effective workers=1; ~2.4x faster on water_phantom_pdd_ocr at n=1e6). Incompatible scenes fail fast
# with a specific reason rather than silently falling back. R/SEM uncertainty is supported by default;
# use --no-uncertainty to disable it.
.venv/bin/python -m chatcarlo run examples/water_phantom_pdd_ocr.yaml --engine kernel -n 1e6 --seed 42

# photon trajectory 3D visualization (small n; overlays onto the preview HTML template)
.venv/bin/python -m chatcarlo trace examples/chest_room.yaml -n 200 --seed 42 -o trace.html

# educational photon-trajectory ANIMATION (small n; playback with orbit/relay/first-person
# cameras, time compression + slow-mo at interactions, HUD, raw-data mode — see
# docs/plan_photon_animation.md). Separate from `trace` (a static overlay); `animate` replays
# the same recorder log over a timeline instead.
.venv/bin/python -m chatcarlo animate examples/chest_room.yaml -n 200 --seed 42 -o anim.html

# cross-section slices through a dose/H*(10) map (default: 3 planes through the max-value voxel)
.venv/bin/python -m chatcarlo plot dose.npz --scene examples/chest_room.yaml -o maps.png

# relative-error map (R) for the same run — see "Statistical uncertainty" below.
# Requires a .npz written without --no-uncertainty.
.venv/bin/python -m chatcarlo plot dose.npz --scene examples/chest_room.yaml --quantity relerr-dose -o relerr.png

# tests (spot-checked against published NIST reference values)
.venv/bin/python -m pytest tests/ -q
```

Run a single test: `.venv/bin/python -m pytest tests/test_transport.py::test_name -q`

There's also `.claude/skills/vive-check/`, a gated workflow skill that runs the four CLI steps in order
(geometry preview → trajectory preview → full run → results) with human approval at each gate. Invoke it for
"walk through the scene with me" style requests rather than chaining the raw CLI calls yourself.

When the user asks for a simulation but no scene.yaml exists yet (or the request is vague), do NOT start writing
a scene directly — invoke `.claude/skills/vive-interview/` first. It elicits the requirements in stages
(purpose → exposure parameters → geometry → run settings) via AskUserQuestion, confirming intent and pinning down
ambiguities before drafting scene.yaml, then hands off to vive-check.

## Architecture

**Transport is not voxelized.** Geometry stays as analytic primitives (box/cylinder/sphere in
[geometry.py](chatcarlo/geometry.py)); each photon steps to the next material boundary by computing an analytic
ray/primitive intersection distance ([transport.py](chatcarlo/transport.py)). This is "analytic surface tracking,"
not Woodcock delta-tracking — since each segment has one homogeneous material, μ is constant along a step, so no
virtual-collision rejection is needed. This scales well for room-size scenes with thin shielding without a
voxel-resolution/memory tradeoff. Overlapping objects resolve by list order (later wins); open space not inside
any object is `background` (default air).

**Module layout around the kernel**: [transport.py](chatcarlo/transport.py) is only the transport loop +
`run_transport`. Spectrum generation (SpekPy/Kramers, heel off-axis spectra) lives in
[spectrum.py](chatcarlo/spectrum.py); source/field sampling and the mAs photon-count calibration in
[source.py](chatcarlo/source.py); interaction angle/energy sampling in [physics.py](chatcarlo/physics.py);
trajectory recording for `trace` in [trajectory.py](chatcarlo/trajectory.py); dose-map conversion and the
non-physical-max warnings in [diagnostics.py](chatcarlo/diagnostics.py); the terminal planar detector tally
(primary/scatter discrimination for the scatter-correction research line) in [detector.py](chatcarlo/detector.py) —
see [docs/plan_scatter_correction_feasibility.md](docs/plan_scatter_correction_feasibility.md) and
[docs/ai/plans/2026-08-03-scatter-phase0-detector-tally.md](docs/ai/plans/2026-08-03-scatter-phase0-detector-tally.md).

**Experimental parallel implementation**: [kernel.py](chatcarlo/kernel.py) is a
from-scratch Numba-compiled per-history scalar transport kernel (Phase B of
[docs/plan_chatcarlo_speedup_post_egs5.md](docs/plan_chatcarlo_speedup_post_egs5.md); box-shaped geometry only,
no cylinder/sphere yet). Its dose-grid path uses the Numba scalar DDA in
[tally_njit.py](chatcarlo/tally_njit.py) by default, with the audited NumPy
`tally.accumulate_track_length_multi` retained as the `use_njit_dda=False` reference/fallback.
`transport.py` remains the production path and the permanent reference implementation
that `kernel.py` is statistically cross-checked against (RNG algorithms differ — MT19937 vs PCG64 — so bit-identity
is not the verification method; see the plan doc's "Phase Bの検証戦略"). Before extending or duplicating
transport logic, check whether it already exists in `kernel.py`.

`chatcarlo run --engine kernel` is an opt-in Phase 1 adapter for box-only, monochromatic,
parallel-field scenes with one effective worker. It samples source origins through the existing
`sample_source_photons` path and supports the same batch R/SEM tracking as numpy; `--engine numpy` remains the default.

**GPU speed ceiling (measured, not `kernel.py`-related)**: before committing to a CUDA port of
`kernel.py` (a hypothetical "route 2"), [docs/gpu_speedup/route1_mcgpu/RESULTS.md](docs/gpu_speedup/route1_mcgpu/RESULTS.md)
measured an *existing, unmodified* diagnostic-energy GPU code (DIDSR/MCGPUv1.3_PCD) on a free Colab
Tesla T4 to get a real number for "how much would GPU buy us." Sustained throughput was ≈70M
histories/s (with a ≈190M cold-start peak that decays within a few consecutive same-size runs —
mechanism not established), which is **≈8× the existing 8-process EGS5 baseline**, not the ≈46×
a naive single-thread comparison would suggest. Read together with the ~10× cumulative EGS5-side
ceiling already measured (compiler flags + process parallelism + RNG replacement,
[docs/egs5_crosscheck/speed_comparison/RESULTS.md](docs/egs5_crosscheck/speed_comparison/RESULTS.md)),
GPU headroom beyond an already-parallelized CPU is roughly one order of magnitude, not two.

**Physics**: photoelectric / Compton (bound Compton — free-electron Klein-Nishina via Kahn rejection sampling,
then an additional S(Z,q)/Z rejection from the incoherent scattering function via `xraylib.SF_Compt`; compounds
sampled by mass-fraction-weighted element pick, same pattern as Rayleigh, before the angular distribution) /
Rayleigh (atomic form factor F(Z,q) via `xraylib.FF_Rayl`, compounds sampled by mass-fraction-weighted element
pick before the angular distribution; the angle itself is drawn via inverse-transform sampling on a cumulative
F(Z,q)² table in x≡q² space followed by a (1+cos²θ)/2 rejection — ≥50% acceptance guaranteed, same two-stage
scheme EGS5 itself uses — replacing a uniform-cosθ proposal whose acceptance rate collapsed to <1% for light
elements at high energy; see [docs/plan_rayleigh_compton_importance_sampling.md](docs/plan_rayleigh_compton_importance_sampling.md)).
Electron range is neglected (kerma approximation — local absorption at
the interaction point), except that photoelectric absorption samples K-shell fluorescence emission
(`sample_fluorescence` in [physics.py](chatcarlo/physics.py); K-shell only, no cascade/L-shell, line energies
below 5 keV are absorbed locally instead of emitted) — when emitted, the photon continues transport at the
fluorescence line energy with an isotropic direction rather than being annihilated. Controlled by
`physics.fluorescence` in scene.yaml (default `true`); toggling it off reproduces the pre-fluorescence local-absorption
behavior. See [docs/plan_fluorescence.md](docs/plan_fluorescence.md) for the design rationale and verification.
[tests/test_transport.py](tests/test_transport.py) checks primary transmission against the analytic Beer-Lambert
law (`exp(-μt)`); [tests/test_fluorescence.py](tests/test_fluorescence.py) checks K-edge data against xraylib,
energy conservation with fluorescence on/off, and the emission rate against the analytic K-shell-fraction×ω_K
expectation.

**Dose/H*(10) tallying is a separate concern from transport.** [tally.py](chatcarlo/tally.py)'s `VoxelGrid` lays a
uniform grid independently of the transport geometry, purely for scoring. Two independent estimators are
cross-validated against each other in [tests/test_tally.py](tests/test_tally.py): a collision estimator
(`energy_deposited`, scored at interaction points) and a track-length kerma estimator (path-integral over the
grid). H*(10) is a fluence-based protection quantity (different from kerma), computed by normalizing the
track-length integral by voxel volume (`VoxelGrid.h10_map_pSv`). `accumulate_track_length` computes, for each
flight segment, the **exact analytic overlap length with every voxel it crosses** (a 3D grid-traversal/DDA
generalizing the single-box `_segment_box_overlap_cm` used and audited in the EGS5 crosscheck scripts to a
segment that crosses an arbitrary number of voxels) — no random sampling, no discretization, zero spatial-binning
variance. This replaced an earlier substep+stratified-random-point Monte Carlo scheme (itself a fix for a
decisive-midpoint scheme that systematically under-scored voxels when many segments started exactly on a voxel
boundary, e.g. a `field.shape: parallel` beam entering a phantom face — see lessons_learned for that bug); the
exact method has no such boundary-phase artifact by construction and needed no `max_substeps` clamp. **This was
initially slower than the old clamped-substep method** (up to ~4.7× at the default `--batch-size` 200,000 with a
2cm grid and n=2e5) but has since been fixed — see [docs/plan_tally_speedup.md](docs/plan_tally_speedup.md)
(Phase 0-3 complete). Phase-by-phase profiling at the regression condition found `np.lexsort` inside
`_segment_grid_traversal` responsible for 84.1% of wall time, and confirmed (contrary to an earlier hedge in this
file) that the cost really was dominated by the sort itself rather than by memory-bandwidth-bound gathers — an
isolated benchmark showed `np.lexsort`'s per-element cost roughly tripling once the array exceeds ~2–4M elements
(a cache cliff), and the largest single traversal call in the regression condition (200,000 segments → ~18M
intersection points) was being computed **twice** (once each for kerma and H\*(10), on identical geometry).
Four fixes landed: (1) `accumulate_track_length_multi` shares one traversal between kerma and H\*(10) instead of
computing it twice; (2) `_segment_grid_traversal_accumulate` (renamed from `_segment_grid_traversal`) chunks
segments so no single sort call exceeds `_CHUNK_TARGET_INTERSECTIONS` (1,000,000), keeping each chunk's sort
arrays cache-resident; (3) the sort itself was replaced with a single-key `np.argsort`
(`_argsort_within_segment`, key = segment_id + 0.5×fractional-position) instead of the two-pass `np.lexsort`,
with a dedicated fuzz test (`tests/test_tally.py`) verifying it reproduces the same segment/overlap decomposition
— including a boundary-collision edge case (two segments' t-values landing exactly on each other) the 0.5 scaling
exists specifically to avoid; (4) each chunk's `np.add.at` now runs immediately (instead of concatenating all
chunks' results before one final scatter-add) — a `/furikaeri` self-review caught that fix (2)'s own pre-registered
memory acceptance criterion had been left unmeasured, and the concatenate-then-add-at design meant chunking hadn't
actually reduced peak memory at all. All four changes are bit-identical with the pre-fix code (verified via
git-worktree A/B against commit `0c97ab4`). Net result (measured with the single consistent script
`docs/speedup_baseline/tally_speedup_timing.py`, since the memory figures originally reported for this regression
turned out not to reproduce under that script and were corrected — see the file for the discrepancy): the
regression condition (chest_room, `--dose-grid`, res=2cm, n=2e5, batch=2e5) went from ~26.5s/4.59GB down to
**~5.5s/~1.7GB**, actually *beating* the original substep method's ~5.9s/2.31GB on both wall-time and peak memory
(interleaved 2–3-rep A/B, alternating arm order, values stable across repeats). Per-history cost is now flat
across batch sizes (was 40µs→149µs superlinear, now ~28–36µs regardless of batch size). Physics results (total
kerma, per-material energy) remain bit-identical throughout — only wall-time/memory changed. Full profiling data
and methodology, plus the reusable measurement scripts: [docs/speedup_baseline/tally_exact_resolution_growth.txt](docs/speedup_baseline/tally_exact_resolution_growth.txt).
The exact method also reaches a somewhat larger voxel set than the old one (same chest_room scene, same seed:
442,111→477,913 nonzero voxels, +8.1%) since it catches thin corner/edge crossings the substep sampling could
miss — this shows up as a slightly higher `n_batches_hit` count on marginal voxels and does not change any
total. Before K-shell fluorescence was modeled, the two estimators disagreed
by design in high-Z materials — the collision estimator deposited the full photoelectric energy locally while
NIST μen/ρ (used by the track-length estimator) already subtracts the mean fluorescence escape fraction. Modeling
fluorescence brought the two into much closer agreement for lead (spot-checked with a 100 keV beam into a thick
lead slab: track-length/collision ratio improved from ~0.38 without fluorescence to ~0.92 with it — see
[docs/plan_fluorescence.md](docs/plan_fluorescence.md) for the verification script and numbers).

**Units and calibration**: relative output is `Gy/history` / `pSv/history`. When `scene.yaml`'s `source.mas` is
set, `photon_count_through_field` (in source.py) uses SpekPy's absolute fluence to get the real photon count
through the field, and per-history values are scaled by that count (not divided again by `n_histories` — see the
mAs double-division bug writeup in [docs/lessons_learned.md](docs/lessons_learned.md) if touching this path).

**Cross-section/dose-coefficient data provenance** (do not "improve" these from memory — see lessons learned):
| quantity | source | used for |
|---|---|---|
| μ/ρ, photoelectric/Compton/Rayleigh split | `xraylib` (EPDL-based, matches NIST XCOM) | transport free-path/interaction sampling |
| μen/ρ (mass energy-absorption coefficient) | NIST XAAMDI, bundled CSV (`chatcarlo/data/nist_xaamdi/`) | kerma/absorbed-dose tally |
| h*(10)/Φ (ambient dose equivalent conversion) | ICRP Publication 74 / ICRU Report 57, bundled CSV (`chatcarlo/data/h_star_10/`) | H*(10) tally |

`xraylib`'s `CS_Energy` diverges from NIST-published μen/ρ by up to ~17% and must not be used for dose — this is
covered by a regression test in `tests/test_materials.py`. Re-fetch data with `scripts/fetch_nist_xaamdi.py` /
`scripts/fetch_h_star_10.py`, never by hand-typing values.

**Statistical uncertainty is a first-class output, not an afterthought** ([tally.py](chatcarlo/tally.py),
[docs/plan_statistical_uncertainty.md](docs/plan_statistical_uncertainty.md), Phase 0-4 complete). Batch statistics
(batch ≡ transport batch, `batch_size` histories) give an unbiased relative-error estimator R = SEM/mean per voxel
and per material, on by default (`track_uncertainty=True`), toggled off with `run --no-uncertainty`. The estimator
uses a snapshot-diff accumulator so totals are bit-identical whether tracking is on or off — turning it on never
changes a physics result, only adds `kerma_sum2`/`h10_sum2`/`n_batches_hit` arrays alongside the existing tally.
`run --dose-grid` prints R and contributing-batch-count next to the max dose/H\*(10), plus a grid-wide reliability
summary; `--dose-out`'s `.npz` carries `rel_err_dose`/`rel_err_h10`/`sem_*`/`n_batches`/`n_batches_hit`; `plot
--quantity relerr-dose`/`relerr-h10` renders it (linear 0–0.5, distinct colormap from dose/H\*(10), voxels with
zero contributing batches masked in grey rather than colored — see "Rの解釈ガイド" below on why that mask matters).
With the default `-n 1e5` and `batch_size` 200,000, M=1 batch and R comes back as an actionable "increase -n or
lower --batch-size" message rather than a silent NaN — this is intentional (see plan doc design judgment 3), not a
bug to "fix" by raising the default n_histories.

**R interpretation guide (necessary, not sufficient)**: R<0.05 generally trustworthy, 0.05–0.10 reasonably
trustworthy, 0.10–0.20 questionable, >0.20 meaningless (MCNP convention). **Low R alone does not mean a voxel's
value is reliable** — a voxel with very few contributing batches can show a deceptively small R (it just hasn't
drawn a large contribution yet). Always check the contributing-batch-count alongside R; `plot`'s grey mask and
`diagnostics.unreliable_max_warning` both encode this rule so it isn't only documentation.

## Known sharp edges (read before trusting "max dose"/"max H*(10)" output)

`chatcarlo run --dose-grid`'s reported max absorbed dose / max H*(10) can still land in a non-physical spot: a
background (air) voxel near the 1/r² point-source singularity, or an air voxel just outside a material boundary
due to backscatter. The CLI detects and prints a `[警告]` when this happens
(`background_medium_warning`/`near_source_air_warning` in diagnostics.py). For a real exposure-point estimate (patient
surface, operator position, etc.), lay a fine grid directly at that position rather than trusting the global max.

A related but distinct effect: at fine resolution the reported max can still grow as resolution gets finer, then
plateau. This is **not** the same bug as a since-fixed systematic bias where flight segments starting exactly on
a voxel boundary (all photons of a `field.shape: parallel` beam, for instance) were under-scored in that boundary
voxel — that one was a real bug in the track-length tally's substep scoring and is fixed (see
`accumulate_track_length` above). **Re-verified 2026-07-26** after replacing the tally with the exact
analytic-overlap method (which removed the `max_substeps` clamp entirely): the growth-then-plateau is **not** a
`max_substeps` artifact — a side-by-side run of the old (clamped-substep) and new (exact) tally at the same
resolutions down to 0.0625cm agreed within statistical noise at every point (the old method only carried a
somewhat larger R, from its own extra spatial-sampling variance, not a different mean). The actual mechanism is
ordinary voxel-averaging of a spatially-varying but *finite-width* beam/field: while the voxel is larger than the
field's local transverse footprint, the peak per-voxel density keeps rising as the voxel shrinks; once the voxel
becomes smaller than that footprint, the value plateaus. Widening the field by 4× moved the plateau's onset
resolution by almost exactly 4× coarser, confirming the mechanism quantitatively. This is physically correct
behavior, not a tally bug. Raw data, the pre-registered hypotheses, and the widened-field confirmation run are in
[docs/speedup_baseline/tally_exact_resolution_growth.txt](docs/speedup_baseline/tally_exact_resolution_growth.txt).
**Not covered by this re-verification**: a true point-source r→0 singularity (evaluating literally adjacent to
the source position, where the beam footprint itself collapses toward zero) — the scenario used to probe this
turned out to still have a finite (if small) footprint at the tested distance, so it exercised the same
finite-footprint mechanism rather than the singularity `near_source_air_warning` is meant to catch. That specific
case remains untested and open. Both warnings above only fire when the max lands on `background`/air, so they
don't catch this class of issue when both neighboring voxels are legitimately the declared material — treat
single-voxel maxima at declared-material boundaries with the same caution. Full writeup:
[docs/lessons_learned.md](docs/lessons_learned.md).

**R (relative error) still matters for triage**: if the max-value voxel's R is high or its contributing-batch-count
is low, an apparent resolution-to-resolution change may simply be statistical noise from an under-sampled voxel,
not the finite-footprint mechanism above — re-run at higher `-n` (or lower `--batch-size`) before drawing
conclusions from a single run.

## Scene files

`scene.yaml`: cm units, z-axis up, floor at z=0. See [chatcarlo/scene.py](chatcarlo/scene.py) for the validator
(`load_scene`/`validate_scene`) — it's designed to produce actionable errors (`geometry[2].size_cm: ...`) for an AI
self-correction loop, plus non-fatal physics-sanity warnings (e.g. filtration below legal minimum, source position
inside a solid). [examples/chest_room.yaml](examples/chest_room.yaml) is the canonical worked example (standing
chest X-ray room).

**Non-clinical/research sources**: `source.kvp` (polychromatic SpekPy/Kramers spectrum) is the default and required
unless `source.spectrum` is given instead — an explicit `[{energy_keV, weight}, ...]` list (e.g. a single entry for
a monoenergetic beam), used for physics cross-checks against reference codes rather than clinical scenes. `kvp` and
`spectrum` are mutually exclusive (validated), as are `spectrum` with `mas`/`ctdi_vol_mGy`/`heel_effect` (those are
fixed to the kvp-based SpekPy path and would silently disagree with an overridden spectrum) — such scenes only
produce relative `Gy/history` output. Similarly `source.field.shape: parallel` (non-divergent beam, `size_cm` only,
no `sid_cm`) is available alongside `rect`/`cone` for the same reference-code-matching use case, with the same
`mas`/`heel_effect` restriction. See [examples/water_phantom_pdd_ocr.yaml](examples/water_phantom_pdd_ocr.yaml).
