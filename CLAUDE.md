# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
It is read in full at the start of every session, so keep it under 200 lines: only rules that apply everywhere.

## Codex併用時の計画運用

`docs/ai/plans/` に `状態: approved` の計画ファイルがある場合、実装はその「対象範囲」「受入条件」「テストコマンド」に厳密に従うこと。計画に書かれていない変更をしない。計画の書き方は [docs/ai/PLAN_TEMPLATE.md](docs/ai/PLAN_TEMPLATE.md) 参照。

## 報告時のルール

作業完了時にユーザーへ報告する際は、進捗の説明だけで終わらせず、成果物そのものを
提示する（生成したHTML/PNG/npzファイルを`SendUserFile`等で送る、重要な数値結果は
本文に具体的に書く、など）。「〜を実装しました」で終わらせて中身を見せないのは
避けること。

## 記録の置き場所（2026-10-19）

- **CLAUDE.md**: どの作業にも効く決まりだけ。段階の経過・数値・注意書きは書かない。
- **`.claude/rules/*.md`**: 場面ごとの常設ルール。`paths:` に合うファイルに触れたときだけ読み込まれる。
  [mv.md](.claude/rules/mv.md)（治療領域の試作）、[chatcarlo.md](.claude/rules/chatcarlo.md)（本体の設計）。
- **段階の記録**: 各段階の RESULTS / DISCUSSION が正典。治療領域は終わるたびに
  [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md) に 1 行足す。教訓は [docs/lessons_learned.md](docs/lessons_learned.md)。
- **自動メモリー**: ユーザーの好みと進め方だけ。PC ごとで共有されないので、両方の PC で要る事実はリポジトリに書く。

## What this is

ChatCarlo is a Monte Carlo photon transport code for diagnostic X-ray energies (10–150 keV). Scenes are declared
in `scene.yaml` — the system is designed for an AI (Claude Code) to write and iterate on scene files declaratively,
then drive validation/preview/run non-interactively via the CLI. Current status is research/education only: doses
and H*(10) must be cross-checked against an established code (EGS5 — PHITS was considered but was not adopted;
see [docs/plan_egs5_crosscheck.md](docs/plan_egs5_crosscheck.md)) before being used for real patient-dose or
shielding decisions (see the warning banner in [README.md](README.md)).

**Latest validation (2026-10-07)**: production NumPy transport at commit `9a91953`, 60 keV in water vs EGS5 — a scoped
benchmark, not blanket validation. Keep these caveats whenever citing it: the historical 2σ/2% criterion gives 42 pass,
1 fail, 4 insufficient-statistics bins among 47 PDD/OCR bins (the Bonferroni auxiliary criterion does not replace that
failure), and EGS5's Rayleigh fit warning is an unquantified reference uncertainty. Numbers:
[docs/egs5_crosscheck/revalidation_2026_10_07/RESULTS.md](docs/egs5_crosscheck/revalidation_2026_10_07/RESULTS.md); the
old BSF/PDD JSON, figures and reports are historical records, not regenerated data.

**Therapy-energy (10 MV) extension**: a prototype in `prototypes/mv_electron/`, outside `chatcarlo/` except one dispatch in
`chatcarlo/__main__.py` (scenes with `domain: therapy_mv` go to `prototypes/mv_electron/entry/`, 2026-10-21); the 150 keV limits and
the diagnostic path are unchanged. Its standing rules are in
[.claude/rules/mv.md](.claude/rules/mv.md) — **read it first when starting MV work from anywhere** (cache folders,
Colab/Drive results, plans), since it only auto-loads when a matching repo file is touched. Current stage, passed
configurations and what comes next: [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md) (not repeated here).

**Scatter correction moved out (2026-08-20)**: the MC scatter-correction research line now lives in `~/Projects/mcscatter`
(engine: unmodified `DIDSR/MCGPUv1.3_PCD`, user decision). **`chatcarlo/detector.py`, its two test files and its plan
`docs/ai/plans/2026-08-03-scatter-phase0-detector-tally.md` stay here** (integrated and audited; an independent
cross-check for mcscatter) — do not delete them as "unused". `docs/ai/plans/2026-08-03-scatter-variance-reduction.md`
(NEE/forced collision, `draft`) also stays, now as a ChatCarlo variance-reduction idea.

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

# GPU G1 CLI (Colab requires preview --engine gpu --gpu-device colab, then approve RUN_DIR)
.venv/bin/python -m chatcarlo run docs/validation/gpu/g4/scene_portrait_detector.yaml --engine gpu --seed 2040
.venv/bin/python -m chatcarlo run docs/validation/gpu/g4/scene_portrait_detector.yaml --engine gpu --gpu-device colab --seed 2040

# photon trajectory 3D visualization (small n; overlays onto the preview HTML template)
.venv/bin/python -m chatcarlo trace examples/chest_room.yaml -n 200 --seed 42 -o trace.html

# educational photon-trajectory ANIMATION (small n; orbit/relay/first-person cameras, time compression +
# slow-mo at interactions, HUD, raw-data mode — see docs/plan_photon_animation.md). Separate from `trace`
# (a static overlay); `animate` replays the same recorder log over a timeline instead.
.venv/bin/python -m chatcarlo animate examples/chest_room.yaml -n 200 --seed 42 -o anim.html

# cross-section slices through a dose/H*(10) map (default: 3 planes through the max-value voxel)
.venv/bin/python -m chatcarlo plot dose.npz --scene examples/chest_room.yaml -o maps.png

# relative-error map (R) for the same run. Requires a .npz written without --no-uncertainty.
.venv/bin/python -m chatcarlo plot dose.npz --scene examples/chest_room.yaml --quantity relerr-dose -o relerr.png

# tests (spot-checked against published NIST reference values)
.venv/bin/python -m pytest tests/ -q
```

Run a single test: `.venv/bin/python -m pytest tests/test_transport.py::test_name -q`. The MV prototype's tests run
separately: `pytest prototypes/mv_electron/tests -q` (one pytest call over both directories fails on duplicate module names).

**Skills are AI-agnostic (2026-10-07).** `vive-interview`, `vive-check` and `vive-audit` live in `.claude/skills/`
(single source); `.agents/skills/<name>` are symlinks so Codex and Antigravity pick them up. They do not assume
Claude-only features. [AGENTS.md](AGENTS.md) is the shared entry point for every AI agent. `vive-crosscheck` and the
`egs5-operator` agent are research-only and Claude Code-specific; the published system must not depend on EGS5.
- No scene.yaml yet, or a vague request → invoke `vive-interview` first (purpose → method and what the system can't do,
  shown up front → output format → exposure parameters → geometry → run settings, confirming at each stage); do NOT start writing a scene directly. It hands off to `vive-check`.
- "Walk through the scene with me" → `vive-check` (geometry preview → trajectory preview → full run → results, with
  human approval at each gate) rather than chaining the raw CLI calls yourself.

## Architecture (summary)

Analytic surface tracking on box/cylinder/sphere primitives (not voxelized; later objects win overlaps; open space is
`background`). Tallying is a separate uniform `VoxelGrid` with an exact analytic segment–voxel overlap. Physics:
photoelectric (with K-shell fluorescence, `physics.fluorescence`), bound Compton, Rayleigh; kerma approximation.
`kernel.py` is an experimental Numba kernel cross-checked statistically against the production `transport.py`.
`chatcarlo/gpu/` (G1, 2026-10-21) is a CPU-twin/CUDA engine reached only through `run_transport(engine="gpu")` (no CLI yet); see the rules file.
Details: [.claude/rules/chatcarlo.md](.claude/rules/chatcarlo.md) (auto-loads when you touch `chatcarlo/`, `tests/`, `examples/`;
**read it first when planning or reviewing package changes from a plan file** in `docs/ai/plans/`).

**Cross-section/dose-coefficient data provenance** (do not "improve" these from memory — see lessons learned):
| quantity | source | used for |
|---|---|---|
| μ/ρ, photoelectric/Compton/Rayleigh split | `xraylib` (EPDL-based, matches NIST XCOM) | transport free-path/interaction sampling |
| μen/ρ (mass energy-absorption coefficient) | NIST XAAMDI, bundled CSV (`chatcarlo/data/nist_xaamdi/`) | kerma/absorbed-dose tally |
| h*(10)/Φ (ambient dose equivalent conversion) | ICRP Publication 74 / ICRU Report 57, bundled CSV (`chatcarlo/data/h_star_10/`) | H*(10) tally |

`xraylib`'s `CS_Energy` diverges from NIST-published μen/ρ by up to ~17% and must not be used for dose — covered by a
regression test in `tests/test_materials.py`. Re-fetch data with `scripts/fetch_nist_xaamdi.py` /
`scripts/fetch_h_star_10.py`, never by hand-typing values.

**R interpretation guide (necessary, not sufficient)**: R = SEM/mean from batch statistics. R<0.05 generally
trustworthy, 0.05–0.10 reasonably trustworthy, 0.10–0.20 questionable, >0.20 meaningless (MCNP convention). **Low R alone
does not mean a voxel's value is reliable** — a voxel with very few contributing batches can show a deceptively small R.
Always check the contributing-batch count alongside R; `plot`'s grey mask and `diagnostics.unreliable_max_warning`
encode this rule.

## Known sharp edges (read before trusting "max dose"/"max H*(10)" output)

- The reported max absorbed dose / max H*(10) of `run --dose-grid` can land in a non-physical spot: a background (air)
  voxel near the 1/r² point-source singularity, or an air voxel just outside a material boundary due to backscatter.
  The CLI prints a `[警告]` then (`background_medium_warning`/`near_source_air_warning` in diagnostics.py). For a real
  exposure-point estimate (patient surface, operator position), lay a fine grid at that position instead of trusting
  the global max.
- At fine resolution the max can grow as resolution gets finer, then plateau. This is ordinary voxel-averaging of a
  finite-width beam (plateau once the voxel is smaller than the field's local footprint; a 4× wider field moved the
  onset 4× coarser) — physically correct, not a tally bug, and not the since-fixed boundary-voxel under-scoring of the
  old substep tally (re-verified 2026-07-26 with the exact tally). **Not covered**: a true point-source r→0
  singularity (still untested and open).
- Both warnings fire only when the max lands on `background`/air; treat single-voxel maxima at declared-material
  boundaries with the same caution. If the max voxel's R is high or its contributing-batch count low, re-run at higher
  `-n` (or lower `--batch-size`) before drawing conclusions.
- Full writeups: [docs/lessons_learned.md](docs/lessons_learned.md),
  [docs/speedup_baseline/tally_exact_resolution_growth.txt](docs/speedup_baseline/tally_exact_resolution_growth.txt).

## Scene files

`scene.yaml`: cm units, z-axis up, floor at z=0. See [chatcarlo/scene.py](chatcarlo/scene.py) for the validator
(`load_scene`/`validate_scene`) — it produces actionable errors (`geometry[2].size_cm: ...`) for an AI self-correction
loop, plus non-fatal physics-sanity warnings (e.g. filtration below legal minimum, source position inside a solid).
[examples/chest_room.yaml](examples/chest_room.yaml) is the canonical worked example (standing chest X-ray room).

**Non-clinical/research sources**: `source.kvp` (polychromatic SpekPy/Kramers spectrum) is the default and required
unless `source.spectrum` is given instead — an explicit `[{energy_keV, weight}, ...]` list (e.g. a single entry for a
monoenergetic beam), used for physics cross-checks against reference codes. `kvp` and `spectrum` are mutually exclusive
(validated), as are `spectrum` with `mas`/`ctdi_vol_mGy`/`heel_effect` (those are fixed to the kvp-based SpekPy path) —
such scenes only produce relative `Gy/history` output. Similarly `source.field.shape: parallel` (non-divergent beam,
`size_cm` only, no `sid_cm`) is available alongside `rect`/`cone` for reference-code matching, with the same
`mas`/`heel_effect` restriction. See [examples/water_phantom_pdd_ocr.yaml](examples/water_phantom_pdd_ocr.yaml).

## Where moved sections went (2026-10-19)

Older documents cite CLAUDE.md sections by name. They now live here:

| old section | now |
|---|---|
| "Therapy-energy extension" (stage history) | [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md) (original text kept at the end); rules in [.claude/rules/mv.md](.claude/rules/mv.md) |
| "Architecture": Physics, "Dose/H\*(10)タリー", module layout, kernel.py, GPU ceiling, Units, Statistical uncertainty | [.claude/rules/chatcarlo.md](.claude/rules/chatcarlo.md) |
| "Scatter correction moved out" | "What this is" above |
| "Known sharp edges", "Commands", data provenance, R guide | this file (condensed; history in the linked writeups) |
