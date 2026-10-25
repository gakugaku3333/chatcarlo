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
with `domain: therapy_mv` goes to `prototypes/mv_electron/entry/` (validate/preview/approve/run; two allow-listed spectra (Mohan 10 MV, 4 MV; always 10×10 cm², SSD 100 cm, water 30 cm; the 4 MV entry output has no EGS5 comparison); a second branch `scoring: seed_ratio` (2026-10-26: the P9c prostate/58-seed 7-field pair, `entry/seed_*.py`, example `examples/therapy/prostate_seed_7field.yaml`, results `docs/validation/mv/entry_seed/RESULTS.md`; allow-list values only, reuses the P9c export/T0 in `~/.cache`, the preview approval of the 2026-10-26 run was given by Claude on the user's instruction),
P7 kernel with the `'p6'` layout, Colab CLI with bounded calls (`colab_cli.TIMEOUTS`, poll retried 3 times); user approval of the 3D preview is required before `run`). To show the preview with SendUserFile, copy `preview.html` from `~/.cache/chatcarlo/mv_entry/<hash>/` into the session folder first (e.g. gitignored `runs/<name>/`); a file under `~/.cache` did not render (2026-10-24, n=1). Watch a background `chatcarlo run` by its PID (`$!` and `kill -0`), not `pgrep -f` (the process name is `Python`, and the waiting shell matches its own pattern). Results: `docs/validation/mv/entry/RESULTS.md`. Stage-by-stage status and links:
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
- Speed-up of the water GPU kernel is paused at P5k variant F (user decision 2026-10-18). The multi-material stack kernel is being sped up as P8g (user decision 2026-10-24): order A1 (precomputed table-constant logs, P8g-1 passed, byte-identical to P8f) → B (state machine + refill; P8g-2 passed, byte-identical, 42.5 µs/history on T4 at 40×256 and 200k histories, ×6.5 vs P8f) → A2 (regridding only where needed; not started, on hold — the remaining binary searches are a recorded speed-up candidate: rationale and entry conditions in the plan's §9). Verification no longer requires bit equality where it is not free (table accuracy, component checks, per-history health, block test vs EGS5/P8e, fault injection); use exact match wherever it holds at no cost. Plan: `docs/ai/plans/2026-10-24-mv-p8g-fast-stack-kernel.md`.
- Check-build counters that use `atomicAdd` must point at global memory: a test kernel that kept them in a thread-local array failed with `CUDA_ERROR_LAUNCH_FAILED` on T4 (P8g-1, 2026-10-24); CPU builds never show this.

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

- Seven fixed patient-coordinate fields and 58 seeds (P9c, completed 2026-10-25 for the prostate-shaped field, 600 batches): `gpu/p9c.py` (`local`/`export`), shared `p9c_*` headers and entries; Colab `gpu/p9c_colab.py --mode t0` or `--mode production --histories-per-field N --t0 DIR [--beams prostate] [--seed S]`, supervised by `gpu/p9c_cli.py run` (10-minute polls, heartbeat-based stall detection, automatic resume from collected checkpoints, `supervisor.json`). Figures and numbers: `prototypes/mv_electron/p9c_figures/`. Paired checkpoints commit complete seven-field batches only; log markers `P9C_BATCH <number>` and `P9C_DONE`. Hand-off and CPU checks: `docs/validation/mv/p9c/README.md`. Keep P9b's failed verdict and the absence of a near-seed EGS5 comparison when reporting P9c results.

- 3D layer boxes and noncontact y-aligned cylindrical seeds (P9b): `gpu/p9b_*`, entry `gpu/p9b.py` (`local`/`export`/`analyse`); G0 (byte-identical to P8g-2), G3–G5, G1-y and G2 passed; G1-x failed on the 0.1 cm seed-material plate block (+1.2 ± 0.3%; P9a-GPU direct comparison shows no orientation error) — stage verdict fail on record. Colab execution is a single `gpu/p9b_colab.py` invocation (G3→G4→G5→G1→G2; final log marker `P9B_DONE`); Colab CLI driver: `gpu/p9b_cli.py` (start/watch); `analyse` reads `<work>/results`. Results and hand-off: `docs/validation/mv/p9b/`.

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
- Seed material (P9a, 2026-10-24): `seed_agti` (Ag 0.5955 / Ti 0.4045 by mass, 2.47 g/cm³, stopping power from the PEGS5 general formula by user decision) is a fifth bundle appended after the four P8a materials; data and checks in `p9a_data.py`, EGS5 `p9a_egs.f` (`media_set` 4/5, parallel 0.05 cm tally), GPU driver `gpu/p9a*.py` with the Colab CLI driver `gpu/p9a_cli.py` (start/watch). It reuses the P8g-2 kernel source unchanged. The requirements for the 7-field seed study (P9b onward) are in `docs/validation/mv/p9a/REQUIREMENTS.md`.
- Fastest passed multi-material kernel: `gpu/p8g2_*` (P8g-2; no production-run driver yet). Multi-material depth-only GPU kernel: `gpu/p8f_*` (P8f passed 2026-10-21; entry `gpu/p8f.py` `local`/`stack`/`gate1`/`reference`/`export`/`analyse`). One history-completing function, no state machine (276 µs/history on T4). Field branches not implemented.
- Colab CLI (`colab`, installed with uv, signed in with the gmail account) can replace the browser/Drive steps: `colab new --gpu T4`, `colab upload` (files over ~150 MB fail: split to 50 MB parts and reassemble on the VM), run the job with `nohup` on the VM, and **poll it with `colab exec` every 10 minutes** (user decision 2026-10-21, changed from 20; P8f used 4 minutes) — the first P8f VM, left unpolled, was gone after ~40 min (cause inferred, not confirmed: idle kernel; n=1), and the entry run's VM polled every 20 min was also gone after 96 of 100 batches (cause unknown; n=1). A third VM (stats10mv GPU run, 2026-10-23, polled every 10 min) was lost mid-download after 16 batches (cause unknown). The entry's poll therefore lists session (gate) records and the fault observation **before** batches, and `run` refuses to resend batches whose session record is missing locally (`colab_cli.require_sessions`; move such batches aside and rerun). Collect finished results at every poll so a lost VM loses at most one interval. Scripts: `gpu/p8f_cli/`. Do not `pip install cupy-cuda12x` on images that ship `cupy-cuda13x`. Always `colab stop` at the end. Watchers: decide completion from a log marker written by the job (never `pgrep -f`, which matched its own shell twice on 2026-10-24), put a time limit on every `colab` call (a hung `colab exec` froze a watcher), and on any failure still collect and `colab stop` (`p8g2_cli`/`p8g3_cli` are the reference implementation). Before starting a supervisor, check that no other supervisor uses the same output folder or session prefix (`pgrep -fl p9c_cli`): on 2026-10-25 a leftover supervisor from an earlier turn, with a wrong seed, fought over the same VM names and log and broke a resume. Pass seeds as an explicit list, never built inside a loop (`9401${s}00` produced 9401300 where 940300 was written in the plan). Do not retry a failure that is deterministic for the same data (a summary check failing after the last batch was retried on 5 fresh VMs): stop when two resumes add no committed batch. Wait conditions that start a second job must be tested once before relying on them (an awk condition fired immediately on 2026-10-25).
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
