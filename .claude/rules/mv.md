---
paths:
  - "prototypes/mv_electron/**"
  - "docs/validation/mv/**"
  - "docs/ai/plans/*mv*"
  - "docs/ai/plans/reviews/**"
  - "scripts/archive_run.sh"
---

# Therapy-energy (MV) extension — standing rules

Coupled photon/electron/positron transport for nominal 10 MV, **outside `chatcarlo/`** (prototype only;
nothing in the production package or its 150 keV limits changes) except the dispatch in `chatcarlo/__main__.py`: a scene
with `domain: therapy_mv` goes to `prototypes/mv_electron/entry/` (validate/preview/approve/run; one allow-listed condition,
P7 kernel with the `'p6'` layout, Colab CLI; user approval of the 3D preview is required before `run`). Results: `docs/validation/mv/entry/RESULTS.md`. Stage-by-stage status and links:
[docs/validation/mv/STATUS.md](../../docs/validation/mv/STATUS.md) (add one row per finished stage there, write the
stage's RESULTS; do not add stage history to CLAUDE.md). Standing decisions and known approximations:
[KNOWN_APPROXIMATIONS.md](../../docs/validation/mv/KNOWN_APPROXIMATIONS.md). Roadmap (draft):
[docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md](../../docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md).

## Standing decisions (user)

- Electron cutoff stays at EGS5's 10 keV — do not propose raising it. Final target is GPU (Google Colab free tier).
- No positron-specific stopping/Bhabha/elastic data (estimated effect ≤ 0.1%). In-flight positron annihilation
  (Heitler, analytic) is always on. Positron ELSEPA tables are generated and checked but not used in transport.
- Bremsstrahlung photons above 10 keV are discrete events and are transported (P3r), normalised to ESTAR's radiative
  stopping power by our own factor (no EGS5 correction table; user decision "B").
- Multiple-scattering step rule (P3s, adopted 2026-10-12): K1 = 0.1 for T ≤ 0.02 MeV, 0.05 for T ≤ 1 MeV, 0.025 above.
- Judging criteria v2 (P2b, literature-based: Rogers & Mohan ICCR 2000, AAPM TG-105): fixed 5 cm depth blocks beyond the
  peak window, block-mean difference ≤ 1% of a fixed `Dref` **and** SEM ≤ 0.33%; build-up and peak are reported, not
  judged. Field runs (P3b-3 onward): deep blocks start at a fixed 5 cm; symmetry is checked on production data;
  P7 central means use bins within 80% of the geometric half-width at that depth. Results of these comparisons cover
  block/central means and 50% positions only — not penumbra shape, build-up, per-bin agreement, nor equivalence.
- Elastic-DCS source: the `elsepa` backend (self-generated ELSEPA tables, `IHEF=0`, 10 keV–10 MeV; refuses to run above
  10 MeV). Geant4 G4EMLOW `dpwa` is **not** used (its README: not for commercial use, only within Geant4). ELSEPA is
  elsepa-2020 (Mendeley Data doi:10.17632/w4hm5vymym.1, CC BY-NC 3.0 per the repository metadata; licence text itself not
  yet read); how to distribute the tables is an **open user decision**. `dcslib` (EGS5's own DCS) is a research-only
  control, never to be shipped; `mf` (McKinley–Feshbach) is a data-free independent check, valid for low Z only (H, O).
- Speed-up of the GPU kernel is paused at P5k variant F (user decision 2026-10-18); the next stage is decided with the user.

## Caveats to keep when citing results

Details in [STATUS.md](../../docs/validation/mv/STATUS.md) and each stage's RESULTS; recorded verdicts are never
overwritten by later diagnostics.
- P1c is formally **undetermined** (pre-registered K9(i) stop). Read P1b's "transport mechanics are sound" with the
  caveat of a real systematic in the hinge's single-step lateral spread (mechanism a hypothesis).
- P3s's electron-beam before/after check (S4) is formally "insufficient statistics" (the step rule was still adopted).
- P3b-1's symmetry check B8 stays **failed** on record; later diagnostics and P3b-3 production data found no asymmetry.
- P4a's Fano pass holds by construction in that kernel: it checks density scaling and boundary crossing, not the
  accuracy of the hinge approximation, and only for a laterally infinite slab scored in depth.
- MV comparisons vs EGS5 judge block/central means and 50% positions only (see "Standing decisions"); a small unexplained
  build-up excess (≈ +0.2–0.3% of `Dref`, P3a–P6; not seen in P7) is report-only.

## Which code to use

- Slab depth dose: `photon_transport_p3t.py` (P3r loop minus redundant work, byte-identical outputs).
  `photon_transport_p3r.py` is hash-frozen by `run_p3r.py` — do not edit it.
- **Slab kernels up to P3t do not carry lateral position for photons or most secondaries** — harmless for depth-only
  scoring, wrong for any lateral scoring. Use the P3b (`photon_transport_p3b.py`) or P4 (`photon_transport_p4.py`,
  byte-identical to P3b when no layers are given) kernels for anything lateral.
- Production runs go through the P3u runner: `batch_runner_p3u.run_batches` with a named kernel from
  `batch_kernels_p3u.KERNELS` (add a kernel rather than a new batch loop). Max 2 worker processes is the standing
  operating rule (more is unmeasured, not a technical limit).
- Slab thickness must be a multiple of the 0.2 cm scoring bin (otherwise deposits fall outside the dose array; the
  scoring-violation counters flag it).
- EGS5 field runs: `run_p3b2.run_egs` (not `run_p3r.run_egs`). EGS5 start-up is ≈ 78 s, so size runs with
  `CPU_seconds`, not wall time. P7 reuses `run_p3b2.run_egs` unchanged with scoped paths/spectrum SHA, one process per
  condition; EGS5 results are sealed (hashes) at completion.
- Production kerma scoring was off (`score_kerma=False`) in all runs before P4: no earlier saved run contains kerma.
- GPU: the current passed configuration is **P5k variant F** (slab). Field kernels: `gpu/p6_*` (P6), `gpu/p7_*` (P7; field
  half-width/SSD as arguments, common ±20 cm layout). P7 entry: `run_p7.py` (`local`/`gate1`/`pilot`/`egs`/`export`/`analyse`).
- Multi-material depth-only GPU kernel: `gpu/p8f_*` (P8f passed 2026-10-21; entry `gpu/p8f.py` `local`/`stack`/`gate1`/`reference`/`export`/`analyse`). One history-completing function, no state machine (276 µs/history on T4). Field branches not implemented.
- Colab CLI (`colab`, installed with uv, signed in with the gmail account) can replace the browser/Drive steps: `colab new --gpu T4`, `colab upload` (files over ~150 MB fail: split to 50 MB parts and reassemble on the VM), run the job with `nohup` on the VM, and **poll it with `colab exec` every 10 minutes** (user decision 2026-10-21, changed from 20; P8f used 4 minutes) — the first P8f VM, left unpolled, was gone after ~40 min (cause inferred, not confirmed: idle kernel; n=1), and the entry run's VM polled every 20 min was also gone after 96 of 100 batches (cause unknown; n=1). Collect finished results at every poll so a lost VM loses at most one interval. Scripts: `gpu/p8f_cli/`. Do not `pip install cupy-cuda12x` on images that ship `cupy-cuda13x`. Always `colab stop` at the end.
- Run the prototype tests separately: `pytest prototypes/mv_electron/tests -q` (collecting it together with `tests/`
  fails: both contain `test_transport.py`).
- ELSEPA tables and archive live outside the repo (`~/.cache/chatcarlo/elsepa/`); regenerate with
  `prototypes/mv_electron/generate_elsepa_p1f.py`.

## GPU verification and measurement

- One C++ source (`prototypes/mv_electron/gpu/`) is compiled by `clang++ -ffp-contract=off` locally (ctypes) and by
  NVRTC `--fmad=false` on Colab. Chain 0: the C++ kernel with MT19937 is byte-identical to the Numba kernel (same machine
  only). Chain 1: Philox (counter-based, keyed by seed and global history number) vs saved MT runs, statistically.
  Chain 2: C++ CPU vs GPU history by history (pass/fail on paired differences; matched fraction is report-only).
- GPU math functions differ in the last bit from CPU for many inputs, and so do x86 g++ vs M3 clang — compare across
  machines statistically or history by history, not bitwise on math results.
- `brems[1]` is accumulated mid-history, so per-history merging changes its summation order (≈1e-13 MeV): compare it
  within a computed bound, everything else bitwise.
- Any speed-up that changes the passed configuration (P5k F) needs its own plan and goes back through chain 2.
- Values the canonical kernel receives as input (bundle tables) are sent unchanged as hash-verified data, including
  P8e's NumPy-generated `log_stop_T`. Values the canonical kernel computes during transport are computed on each
  device with the same functions as transport, never precomputed on the host and transferred to the GPU. P5i's
  per-device table-log initialization is an adopted optimization for its kernel; P8f adds no such optimization.
- Colab g++ needs `p5d_gxx_compat.h` (the frozen kernel calls bare `isfinite`).
- Compare speeds only within one Colab session (the same P5i measured 34.41 and 33.12 µs/history in two sessions);
  cross-session ratios are rough.
- Keep raw ncu exports and parse them locally (CSV parsing inside Colab failed every time).
- Never fingerprint launch/measurement tools (notebooks, measurement scripts) in gate records; read the real artefact
  shape locally before shipping a Colab step.
- Do not rank performance from lane counts, instruction counts or the CPU equal-cost emulation (P5h/P5k/P5l: it
  predicted the opposite). If speed-up resumes, first compute resident blocks from registers (allocation granularity),
  shared memory and launch — P5l at 130 registers was still 1 block/SM.

## Data, freezing and deletion

- The Mohan spectra (NRC EGSnrc distribution) are untracked (redistribution terms unconfirmed); never copy their values
  into tracked files. `mohan10.spectrum` and `mohan10_bins.csv` are in `docs/validation/mv/p0/data/` (git-ignored);
  `mohan4.spectrum` lives in `~/.cache/chatcarlo/p7/spectra/` (keep that folder; P7 code reads it).
- `docs/validation/mv/p2/manifest.json` (`protected`) is checked on every P2 input load; before deleting or moving
  anything, check it is not listed there or in another manifest/config, move it aside, and run the checks first.
  The 137 P1b–P1f `.npz` entries removed on 2026-10-16 keep their hashes in
  `docs/validation/mv/p2/removed_2026-10-16.json`.
- `docs/validation/mv/p1,p1b,p1c,p1d/` are frozen by P1e/P1f (`p1e_common.targets()` hashes every file there): do not
  edit or add files in them. Put later notes in STATUS.md.
- Finished-stage cache data: archive large run directories under `~/.cache/chatcarlo` with `scripts/archive_run.sh <dir>`
  (lossless tar + zstd, every SHA256 checked after extraction, original moved to the Trash). What is archived and which
  commands need extraction first (P2 input loads, `p6 analyse`, `run_p7 analyse`, pilot-lock checks) is listed in
  STATUS.md. Do not change `np.savez` in the shared saving helpers instead: P5-era frozen code uses them. A Drive-copied
  export bundle may be moved to the Trash after its manifest hashes are checked against the Drive copy.

## Codex and Colab hand-off

- Codex review originals are kept in `docs/ai/plans/reviews/`. Codex's sandbox cannot write outside the repository
  (`~/.cache`); checks that need it are run by Claude.
- Colab input bundles go to the user's gmail Drive `MyDrive/viveMonte/p5/` (or the stage's folder) via the locally
  synced folder (`~/Library/CloudStorage/GoogleDrive-<gmail>/マイドライブ/...`); results are read back from it. Bundles
  are never committed. Check which account the Drive connector is signed into (it was the university one once) before
  concluding a file is missing.
- Browser steps are done by the user (user decision 2026-10-17): give the Colab URL, "T4", "Run all" and the Drive
  permission (gmail account) in a few lines, then watch the synced Drive folder. Keep every Colab step a single
  "Run all". The Drive mount often times out while the permission is pending; rerunning "Run all" fixes it.
- Large bundles (P7: ~750 MB) take minutes to upload: name the files the user should see in the Drive folder and ask
  them to wait until all are listed before "Run all" (a partial file stops the run safely, but the run is wasted).
