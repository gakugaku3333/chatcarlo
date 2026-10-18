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

**Latest validation (2026-10-07)**: current production NumPy transport at commit `9a91953`
was rechecked at 60 keV in water, with matched density 1.000, EGS5 bound-Compton
IBOUND=1/INCOH=1, and explicit near-vacuum ChatCarlo boundaries. BSF_w passes
(difference +0.378%, 0.833σ); central PDD passes all 15 bins. Across PDD/OCR's 47 bins,
the historical 2σ/2% criterion gives 42 pass, 1 fail, 4 insufficient-statistics bins.
The preregistered Bonferroni auxiliary criterion passes all 43 eligible bins; it does
not replace the historical failure. EGS5's Rayleigh distribution fit warning remains
an unquantified reference uncertainty. This is a scoped benchmark, not blanket validation.
Use [current results](docs/egs5_crosscheck/revalidation_2026_10_07/RESULTS.md) for current
numbers; the old BSF/PDD JSON, figures and reports are historical records, not regenerated data.

## Therapy-energy extension (in progress, prototype only — 2026-10-14)

Work toward coupled photon/electron/positron transport for nominal 10 MV is under way, **entirely outside
`chatcarlo/`**. Nothing in the production package or the 150 keV limits has changed.

- Roadmap (draft): [docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md).
  Presentation story it serves: [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md).
- P0 feasibility (done): [docs/validation/mv/p0/RESULTS.md](docs/validation/mv/p0/RESULTS.md) — xraylib stops at
  ~800 keV, NIST XCOM chosen as the MV photon data path, EGS5 MV reference with electron transport established.
- P1 electron-only prototype (done): `prototypes/mv_electron/` (class-II condensed history, dual random hinge,
  Numba). [docs/validation/mv/p1/RESULTS.md](docs/validation/mv/p1/RESULTS.md): all EGS5-independent checks pass;
  the 2 MeV EGS5 comparison missed the 3% tolerance (3.02%) with a screened-Rutherford elastic DCS.
- P1b elastic-DCS experiment (done): [docs/validation/mv/p1b/RESULTS.md](docs/validation/mv/p1b/RESULTS.md).
  Feeding the prototype the same DCS EGS5 uses (research-only control, never to be shipped) brings the difference
  to 0.54%, so the transport mechanics are sound and the DCS approximation was the cause. EEDL fails badly at
  1–2 MeV (its angular tables jump from 256 keV to 10 MeV). The production elastic-DCS source is still undecided.
- P1c analytic spin correction (done, verdict open): [docs/validation/mv/p1c/RESULTS.md](docs/validation/mv/p1c/RESULTS.md).
  A new `mf` backend multiplies the screened-Rutherford DCS by the McKinley–Feshbach Mott/Rutherford ratio — no
  external data file; valid for low Z only (tested: H and O). **The EGS5 comparison is formally undetermined, not
  passed**: the pre-registered sanity check K9(i) (lateral spread after one 0.5 mm step at K=0.05, against
  single-scattering reference) gave −2.02% against a 2% tolerance, and the stop rule forbids judging the dose
  comparison after that. Facts: the same condition gave −1.96% with the `dcslib` control in P1b (each SEM ≈ 0.29%),
  so `mf` did not make it worse (0.06 points apart, inside 1σ), and both are ~7σ from zero — a real systematic in
  the hinge's single-step lateral spread at that step size, already present in P1b, which landed just inside the
  threshold. K9(i) at K=0.025 passes for both. The mechanism is a hypothesis, not established. So read P1b's
  "transport mechanics are sound" with this caveat. Measured values outside the formal verdict: the saved 2 MeV `mf`
  run differs from EGS5 by 0.94% of Dref at most (tolerance 3%); `mf` vs `dcslib` differ by up to 0.71% near R50
  (29 of 81 bins beyond 2σ — a real model difference).
- P1d half-step retest (done): [docs/validation/mv/p1d/RESULTS.md](docs/validation/mv/p1d/RESULTS.md). User decision
  after P1c: rerun with the multiple-scattering step limit halved (`K1_max` = 0.025; one halving only, chosen on the
  EGS5-independent check, new fixed seeds, P1–P1c verdicts untouched). All pre-registered checks pass for both
  backends; 2 MeV max dose difference vs EGS5 is 0.45% (`dcslib` control) and 0.82% (`mf`, analytic, low Z only),
  tolerance 3%. **`K1_max` = 0.025 is now the adopted step.** Caveats to keep: (1) the run stopped once on a hash
  mismatch caused by Claude editing an unrelated plan file mid-run and was resumed under a reviewed amendment —
  no dose result existed at that point; (2) the change in EGS5 residual from halving the step is not statistically
  distinguishable from zero; (3) repeated K9(i) values with new seeds differ from the earlier ones by ~2σ, so the
  P1c "fail by 0.02 points" was within seed-to-seed scatter — the cause of the offset is still a hypothesis.
- Elastic-DCS source for production (user decisions, 2026-10-08): Geant4's G4EMLOW `dpwa` tables were chosen first,
  then **dropped** — the dataset's top-level README says "DATA ARE *NOT* FOR COMMERCIAL USE AND MUST BE USED WITHIN
  GEANT4". The user then chose **self-generating tables with ELSEPA** (elsepa-2020, Mendeley Data
  doi:10.17632/w4hm5vymym.1, CC BY-NC 3.0 per the repository metadata; sha256 `1066d4d4…a7e5`). A trial build with
  gfortran outside the repo reproduces EGS5's `dcslib` transport cross sections for oxygen to ~0.01% and hydrogen to
  ~1% (options not yet matched). Neither archive nor generated tables are in this repository. `mf` stays as a
  data-free independent check.
- ELSEPA tables (P1e failed, P1f passed — 2026-10-09): [P1e](docs/validation/mv/p1e/RESULTS.md) generated H/O
  electron and positron tables with ELSEPA's default high-energy factorization (`IHEF=1`) and **failed** its own
  table check (interpolation vs direct calculation, 52 of 228 cases; the factorization switch at 20.1 keV × Z makes
  the DCS jump by up to ~1.5%, and part of the check compared against the wrong reference).
  [P1f](docs/validation/mv/p1f/RESULTS.md) regenerates them with pure partial waves (`IHEF=0`, 10 keV–10 MeV only)
  and passes every pre-registered check: interpolation error ≤ 0.13%, and with the new `elsepa` backend the 2 MeV
  max dose difference vs EGS5 is 0.75% (tolerance 3%; `dcslib` control 0.44%, `mf` 0.82%, same step). Caveats:
  P1f is a redesign made after seeing P1e fail (basis: EGS5-independent checks only); Codex ran out of quota mid-run
  and Claude finished the remaining runs with the frozen resume command; one check (K5b) first recorded a "fail" that
  was a test-harness JIT error and was redone under a reviewed amendment, at a point when the dose result was already
  visible; ELSEPA's "dcs table is not consistent" warning (58 cases, total cross section series-vs-table ≈0.25%) is
  accepted under a limit set in advance with that value known; hydrogen differs from EGS5's table by ~1% for an
  unknown reason (≈0.06% on water's G1). Positron tables are generated and checked but not yet used in transport.
  Tables and the ELSEPA archive live outside the repo (`~/.cache/chatcarlo/elsepa/`); regenerate with
  `prototypes/mv_electron/generate_elsepa_p1f.py`. Above 10 MeV the `elsepa` backend refuses to run.
  How to distribute the tables (CC BY-NC 3.0 program; licence text itself not yet read) is an open user decision.
- The prototype's tests are run separately from the main suite (`pytest prototypes/mv_electron/tests -q`);
  collecting both directories in one pytest call fails because both contain `test_transport.py`.
- P2 coupled photon/electron/positron slab (done, 2026-10-10): [docs/validation/mv/p2/RESULTS.md](docs/validation/mv/p2/RESULTS.md).
  Monoenergetic 1.25/2/6/10 MeV photons on a 30 cm water slab vs EGS5. **Formal verdict stays as recorded: Q0–Q4 pass,
  Q6 undetermined, Q7 insufficient statistics** (the pre-registered precision had no margin: 0.3345% vs 0.3333%).
  Run notes incl. a disk-full stop and a 2-process helper: [EXECUTION_NOTES.md](docs/validation/mv/p2/EXECUTION_NOTES.md).
- P2b criteria v2 (2026-10-11, user decision): literature-based redesign (Rogers & Mohan ICCR 2000, AAPM TG-105). Judge
  fixed 5 cm depth blocks beyond the peak window: block-mean difference ≤ 1% of a fixed `Dref` **and** SEM ≤ 0.33%;
  build-up and peak are reported, not judged. Applied post hoc to P2 (all 12 comparisons within 1%, max 0.38%) and to
  later stages. Plan: `docs/ai/plans/2026-10-11-mv-p2b-criteria-redesign.md`. A 6 MeV "+1.05%" difference between two
  EGS5 settings, found post hoc, did **not** reproduce at 5× statistics (+0.23 ± 0.13%).
- P2c positron annihilation in flight (done): [docs/validation/mv/p2c/DISCUSSION.md](docs/validation/mv/p2c/DISCUSSION.md).
  Heitler cross section, analytic, no data file. In-flight probability matches theory within 0.3% (1–9 MeV). It removed
  a systematic offset vs EGS5 (whole-slab deposit +0.11% → −0.04% at 6 MeV, +0.26% → −0.05% at 10 MeV). Now always on.
- P3a Mohan 10 MV spectrum on the slab (done): [docs/validation/mv/p3a/DISCUSSION.md](docs/validation/mv/p3a/DISCUSSION.md).
  All five deep blocks within 1% of EGS5 (max 0.29 ± 0.11%). **This is a broad parallel beam on a slab, not a field
  PDD/OCR** (that is P3b, not started). The spectrum file is untracked (redistribution terms unconfirmed); never
  copy its values into tracked files. Pair-production counts differ by −1.2% because EGS5's piecewise-linear fit smears
  the pair threshold (total attenuation still agrees within 0.06%). A small build-up excess of the prototype
  (≈ +0.3 ± 0.1% of `Dref`) persists across energies and settings; cause unknown.
- P3s step-size rule (done; **adopted by user 2026-10-12**): [docs/validation/mv/p3s/DISCUSSION.md](docs/validation/mv/p3s/DISCUSSION.md).
  Multiple-scattering step K1 = 0.1 for T ≤ 0.02 MeV, 0.05 for T ≤ 1 MeV, 0.025 above (was 0.025 everywhere). ~1.35×
  faster, dose unchanged within statistics. Caveat: the electron-beam before/after check (S4) is formally "insufficient
  statistics" (no difference seen; the precision condition was set without sizing it). Prototype is still ~3× slower
  than EGS5; ~100 µs/history of non-step overhead is unexplained.
- P3r bremsstrahlung photon transport (done, 2026-10-12): [docs/validation/mv/p3r/DISCUSSION.md](docs/validation/mv/p3r/DISCUSSION.md).
  Same Bethe–Heitler formula as EGS5, normalised to ESTAR's radiative stopping power by our own factor (no EGS5 correction
  table; user decision "B"). Photons above 10 keV are discrete events and are transported; the EGS5 reference was rerun
  with them transported too. Mohan 10 MV slab: all five deep blocks within 1% (−0.04 to +0.04%, SEM ≈ 0.1%); the dose
  change from transporting them (≈ −0.85% near the peak, ≈ 0 beyond 20 cm) is the same in both codes. Making radiative
  loss discrete did not move the build-up excess (still ≈ +0.3%). Caveats: Codex hit its usage limit and Claude wrote the
  check/production/analysis part of `run_p3r.py`; three check amendments (one chance failure, two harness/plan mistakes);
  bremsstrahlung count/energy created are 0.7 ± 0.2% / 1.2 ± 0.4% lower than EGS5, unexplained (not the formula: rates agree within 0.07%; the
  −1.4% pair count covers only ≈ 0.1%; EGS5's 1%-tolerance piecewise-linear electron tables remain the candidate, not confirmed);
  the prototype got ~40% slower (≈ 310 µs/history, 4× EGS5) — about half of that remains with discrete events switched
  off, so it is overhead in the copied loop (cause found and removed in P3t, below). Slab thickness must be a multiple of the 0.2 cm scoring
  bin (otherwise deposits fall outside the dose array; the scoring-violation counters flag it).
- P3t loop speed recovery (done, 2026-10-12): [docs/validation/mv/p3t/RESULTS.md](docs/validation/mv/p3t/RESULTS.md).
  `photon_transport_p3t.py` is the P3r loop minus two redundant operations (an unused per-step `event_order` call that
  allocated an array, ≈ 23–28 µs; a duplicated bremsstrahlung table lookup per energy update, ≈ 36–57 µs). Outputs are
  byte-identical to P3r (and to P3s with discrete events off). Mohan slab, 50k histories, single process: P3s 201,
  events off 201 (was 254), photons transported 228 µs/history (was 301) — ≈ 3× EGS5 again. **Use P3t for new work**;
  `photon_transport_p3r.py` is left untouched because it is hash-frozen by `run_p3r.py`. The remaining ≈ 23 µs with
  discrete events on is probably the table lookup itself (not separated; changing it could change bits).
- P3u parallel batches (done, 2026-10-12): [docs/validation/mv/p3u/RESULTS.md](docs/validation/mv/p3u/RESULTS.md).
  `batch_runner_p3u.run_batches` runs independent-seed batches of a named kernel (`batch_kernels_p3u.KERNELS`; currently
  the P3t slab) in up to 2 spawn processes, saves each batch from the parent only, resumes, and stacks in batch-index
  order. For a fixed config the arrays are byte-identical for 1 or 2 workers and across stop/resume, and each batch equals
  a direct P3r call with the same seed. Mohan slab, 2M histories: 492 s → 258 s (×1.90). The worker cap of 2 is the
  standing operating rule, not a technical limit (more is unmeasured). **New production runs (P3b onward) should go
  through this runner**; add a kernel rather than a new batch loop.
- Standing decisions and known approximations live in
  [docs/validation/mv/KNOWN_APPROXIMATIONS.md](docs/validation/mv/KNOWN_APPROXIMATIONS.md): no positron-specific
  stopping/Bhabha/elastic data (estimated effect ≤ 0.1%); electron cutoff stays at EGS5's 10 keV (do not propose raising
  it); bremsstrahlung photons are transported since P3r; GPU target
  is Google Colab free tier.
- P3b-1 prototype field kernel (done, 2026-10-12; **no EGS5 comparison yet**):
  [docs/validation/mv/p3b/RESULTS_P3B1.md](docs/validation/mv/p3b/RESULTS_P3B1.md). `photon_transport_p3b.py` carries
  full xyz for every particle, has the divergent 10×10 cm² source and scores an axis box, the whole slab and lateral
  profiles by straight-segment apportioning (legacy depth outputs untouched and byte-identical to P3t without a field).
  Registered in the P3u runner as `'p3b_field'`. All deterministic checks pass. **The pre-registered symmetry check B8
  failed** (1 of 9 contrasts, z = −3.59, 2M histories; the test ran at its designed ≈5% family-wise level, so a coarse
  SEM (≈ 2% per contrast) does not explain a false alarm — it only means a pass would have shown little); a diagnostic
  with a rule fixed beforehand (≈60× the volume, new seed) found no bias (max |z| 1.61, asymmetries ≤ 0.75 ± 0.5%), so
  the failure is consistent with a ≈1% chance event — B8 stays recorded as failed and symmetry is to be rechecked in the P3b-3 pilot.
  50%-dose positions match the geometric field edge (5.25/5.51/6.05 cm at 5/10/20 cm depth). Speed with regions:
  308 µs/history (1.32× the P3t slab) after removing a per-step allocation (was ≈ 700).
- P3b-2 EGS5 field user code (done, 2026-10-12; **still no prototype-vs-EGS5 dose comparison**):
  [docs/validation/mv/p3b/RESULTS_P3B2.md](docs/validation/mv/p3b/RESULTS_P3B2.md). `p3b_egs.f` = `p3r_egs.f` plus the same
  divergent source and the same region scoring (boxes are written by Python from `expand_regions`; the apportioning code
  `p3b_egs_score.inc` and the `ausgab` in `p3b_egs_ausgab.inc` are the very files the standalone gfortran test drivers
  include). All pre-fixed checks pass: legacy output identical to `p3r_egs.f` without field/regions, scorer equal to the
  prototype's on the prototype's own recorded deposits (≤ 1e-13 MeV), per-history region ledger 0 violations, source
  uniformity p = 0.23, stop/resume byte-identical. Region outputs have the prototype's names, shapes and unit (MeV per
  batch). Run through `run_p3b2.run_egs` (not `run_p3r.run_egs`). Loop CPU: 68 → 124 µs/history with regions (×1.8);
  EGS5 start-up is ≈ 78 s, so use `CPU_seconds`, not wall time, for sizing.
- P3b-3 field PDD/OCR vs EGS5 (done, 2026-10-13): [RESULTS](docs/validation/mv/p3b/RESULTS_P3B3.md),
  [DISCUSSION](docs/validation/mv/p3b/DISCUSSION_P3B3.md). Mohan 10 MV, 10×10 cm² at the surface, divergent, 46M histories
  per code (count fixed from a 4M pilot ×1.5). **All 11 pre-fixed quantities are inside tolerance with the precision
  condition met**: 2×2 cm axis-box 5 cm blocks from 5 to 30 cm −0.44 to +0.28% of `Dref` (SEM 0.17–0.27%; tolerance 1%),
  central (±4 cm) profile means at 5/10/20 cm −0.18 to +0.05% (SEM 0.16–0.22%), 50% positions within 0.002 cm (SEM
  0.002–0.004 cm; tolerance 0.1 cm). Symmetry on the production data: not detected in either code (max |z| 1.85/1.91;
  detectable size 0.8–2.2% of the central mean) — P3b-1's B8 stays recorded as failed. Scope: block/central means and
  50% positions only; penumbra shape, build-up and per-bin agreement are not judged, and it is not an equivalence test.
  User decisions (2026-10-13): deep blocks start at a fixed 5 cm (the 97%-window rule is unstable for the small box:
  its end moved 2.25–3.75 cm at 2M), symmetry checked on production data. Report-only: whole-slab 0–2 cm deposit
  +0.18 ± 0.08% (the known build-up excess, still unexplained); 20–80% widths 0.006–0.010 cm narrower in the prototype
  (each within its ≈0.01 cm error). Pilot differences of ≈ −1.5% (≈2σ at 4M) shrank to −0.2/−0.3% in production. Runs go
  through `run_p3b3.py` (`pilot` / `production` / `analyse`; statistics in `analysis_p3b3.py`). One harness stop (a wrong
  shape assumption for the legacy depth arrays, fixed before any comparison value was seen; plan amendment 1). Time:
  prototype 167 µs/history wall on 2 processes, EGS5 123 µs CPU.
- P4 EGS5-independent checks (done, 2026-10-14): [P4a Fano](docs/validation/mv/p4/RESULTS_P4A.md),
  [P4b/P4c kerma and range](docs/validation/mv/p4/RESULTS_P4BC.md); skeleton and user decisions in
  [docs/ai/plans/2026-10-13-mv-p4-independent-checks.md](docs/ai/plans/2026-10-13-mv-p4-independent-checks.md).
  **All pre-fixed quantities pass.** P4a: new kernel `photon_transport_p4.py` (= P3b + per-layer density along depth,
  mass-thickness scoring, an isotropic uniform-per-mass charged source, optional per-history RNG; byte-identical to P3b
  when no layers are given). Fano test with electrons of 0.1/1/10 MeV in two layered systems (density ratios
  1:1/2:2:1/4:1 and 1:1/1024:1): 18 layer doses within −0.46 to +0.35% of the analytic value (SEM 0.10–0.29%; tolerance
  1%, user decision). With identical random numbers, layered and uniform systems agree history by history
  (20,000/20,000 in all 12 pairs), and deliberately omitting the density factor is detected. **Read this narrowly: in
  this kernel the Fano test holds by construction** (boundaries only cut straight segments; no boundary-dependent step),
  so it checks density scaling and boundary crossing, *not* the accuracy of the hinge approximation, and only for a
  laterally infinite slab scored in depth. P4b: in effectively infinite water (2,000 cm slab, photon started at the
  centre, existing P3b kernel untouched) the summed collision kerma from NIST μen/ρ equals the incident energy within
  +0.02 to +0.24% at 0.5–10 MeV (SEM ≈ 0.08%; tolerance 1%); all five are positive, weighted mean +0.09 ± 0.04%
  with no visible energy dependence (χ² 4.3 / 4) — a small common offset of unknown origin, not a radiative-yield error
  (that would grow with energy; brems radiated fraction is 0.07% at 0.5 MeV and 2.6% at 10 MeV).
  Report-only: dose / collision kerma on the Mohan slab is 1.012 ± 0.001 over 5–25 cm. P4c: CSDA range from the P1
  electron kernel (hard collisions off, straight) vs the integral of 1/S: −0.009/−0.008/−0.001%; that integral vs NIST
  ESTAR's range table: within 0.02% (tolerance 0.1%). Not done: the interaction-point energy-transfer cross-check (B3),
  a photon-source Fano test, non-slab cavities, published measurements (user decision). Production kerma scoring had
  been switched off in all earlier runs (`score_kerma=False`), so no earlier saved run contains kerma.
- P5 GPU-oriented rewrite (in progress, 2026-10-14). Skeleton and user decisions:
  [docs/ai/plans/2026-10-14-mv-p5-gpu-rewrite.md](docs/ai/plans/2026-10-14-mv-p5-gpu-rewrite.md) — **CuPy RawKernel (CUDA C++)**,
  first target the Mohan slab depth dose only, speed report-only, Colab is run by the user. One C++ source
  (`prototypes/mv_electron/gpu/`) is compiled by `clang++ -ffp-contract=off` locally (ctypes) and by NVRTC `--fmad=false` on
  Colab. Verification chain: chain 0 = the C++ kernel with MT19937 is byte-identical to the Numba kernel (only on the same
  machine); chain 1 = Philox (counter-based, keyed by seed and global history number) vs saved MT runs, statistically;
  chain 2 = C++ CPU vs GPU history by history (pass/fail on paired differences; the matched fraction is report-only).
  - [P5a0](docs/validation/mv/p5/RESULTS_P5A0.md) (done): small trial on M3 and Colab T4. RNG, table read-out, stack and
    atomic sums are bit-identical CPU↔GPU; `--fmad=false` works (without it 16% of an arithmetic expression differs);
    GPU math functions differ in the last bit for 0.3–27% of inputs, and so do x86 g++ vs M3 clang. Toy speed GPU/CPU ≈ 4.7×.
  - [P5a](docs/validation/mv/p5/RESULTS_P5A.md) (done): the full kernel ported to C++. Byte-identical to the Numba kernel
    (2×20k histories) **and reproduces saved P3r production batches 0 and 2 byte for byte**. Philox 10M histories vs saved
    P3r 10M: deep 5 blocks −0.003 to +0.139% of `Dref` (SEM 0.09–0.13%; tolerance 1%). Report-only: build-up 0–2 cm
    −0.33 ± 0.15% (2.2σ, single observation). CPU speed 164 vs 284 µs/history (Numba), cause not examined.
  - [P5b](docs/validation/mv/p5/RESULTS_P5B.md) (done): GPU wrapper without editing the P5a kernel (each thread runs one
    history into its own local output, then merges by atomics), NumPy-only judging shared by Mac and Colab, Drive bundle.
    Local checks L0–L3 pass (incl. planted faults rejected as pre-registered); all 7 kernels compile on Colab T4 (no warnings).
    Note: `brems[1]` is accumulated mid-history, so per-history merging changes its summation order (≈1e-13 MeV) — compare it
    within a computed bound, everything else bitwise.
  - [P5c](docs/validation/mv/p5/RESULTS_P5C.md) (done, Colab T4): all pre-fixed checks pass. 10M histories, same seed and
    history numbers as the saved P5a Philox run: **every history took the same path** (counts and steps identical in all 100
    batches); dose bins differ only in the last bits (max relative 2.9e-11), deep-block paired differences ~1e-15 % of `Dref`.
    So Numba → C++ CPU → T4 GPU is chained for the slab depth dose. Rough speed from file times: ≈80 µs/history on T4 vs
    164 on M3 C++ (superseded by P5d).
  - [P5d](docs/validation/mv/p5/RESULTS_P5D.md) (done, report only): every timed run passed its correctness gate. µs/history:
    M3 Numba 257, M3 C++ 122 (P5a's 284/164 did not reproduce — use these), Colab Xeon C++ 358, **T4 79 (100k) / 77 (1M)**.
    T4 vs same-machine CPU ×4.5–4.7; vs M3 C++ only ×1.5–1.6. Diagnostics: atomic merge costs ~0 (0.1%), block size 64/128/256
    within 2.6%; the kernel uses 255 registers (the cap) and 30 KB local memory per thread → only 256 threads/SM resident.
    Claude's reading (not proven): register/local-memory pressure, with FP64, limits it. Speeding up changes the passed
    configuration → separate plan back through chain 2. Colab g++ needs `p5d_gxx_compat.h` (frozen kernel calls bare `isfinite`).
  - [P5e](docs/validation/mv/p5/RESULTS_P5E.md) (done, exploration only): 12 wrapper/compile variants (static vs dynamic history
    fetch, stack 64/16, register cap none/168/128), all bit-identical to V0 per history, **all within ±2% of V0**. Dynamic fetch −1.7%
    (selected by the pre-fixed rule, **not adopted** by user decision; V0 stays the passed configuration). Doubling resident blocks
    did not help; sorting histories by step count made it 33% slower. Claude's reading: a shared throughput limit (local-memory
    traffic or FP64), not warp imbalance or latency (superseded by P5f-0). User decision 2026-10-15: rewrite the kernel itself.
  - [P5f-0](docs/validation/mv/p5/RESULTS_P5F0.md) (done): ncu works on Colab T4. V0: **FP64 pipe 83% busy but only 2.1 of 32
    threads active per warp** (branch divergence: each lane follows a different particle/process); memory 5%, L2 hit 99%. Whole-kernel
    switches: moving the 72 MB `ys` reads into a cache-sized panel −0.4% (table is not the bottleneck); shrinking the dose path −8.6%.
    First rewrite candidate: make lanes in a warp do the same work.
  - [P5f-1](docs/validation/mv/p5/RESULTS_P5F1.md) (done; **P5g adopted as the passed GPU configuration**, user decision 2026-10-16): kernel rewritten as a per-history state
    machine (`p5g_*`, P/C/E/H/D states) with the GPU driver advancing all lanes of one state together each round (ballot masks).
    Each history's operation order is unchanged: **CPU build byte-identical to the 100 saved P5a Philox batches; on T4, 20k histories
    bit-identical to V0 per history; P5c's C0–C4 (10M) all pass. T4 speed 79.2 → 42.5 µs/history (×1.87)**, same-session alternation.
    V0 divergence map (ncu source counters): 2.08 lanes/32 overall, 2.04 on FP64, and ~2 even on lines every history runs (Philox,
    step length) — consistent with lanes not reconverging inside the nested loops. P5g's own map not taken (gate-2 record fingerprints
    the measurement script). Lesson: never fingerprint launch/measurement tools in gate records; read the real artefact shape locally
    before shipping a Colab step. (Superseded as the passed configuration by P5i, then P5j — below.) P5g's map, taken later:
    4.80 lanes/32, warp instructions 9.36e10 → 4.49e10.
  - [P5h](docs/validation/mv/p5/RESULTS_P5H.md) (done, **not adopted**, user decision 2026-10-16): binary searches replaced by a
    bucket table + verified short search; all gates pass (bit-identical) but T4 **≈1.5% slower**. Search lines were 20.3% of warp
    instructions, yet removing integer work did not buy time.
  - [P5i](docs/validation/mv/p5/RESULTS_P5I.md) (done, **adopted** 2026-10-16): logs of table constants computed once per device
    with the same `log` (never host-computed). Gates 0–3 pass, bit-identical to P5g. Same-session T4 42.98 → 34.41 µs/history
    (×1.25); FP64 warp instructions −22%, all instructions −13%.
  - [P5j](docs/validation/mv/p5/RESULTS_P5J.md) (done, adopted 2026-10-17; superseded by P5k F):
    removes the Møller constant-argument logs, `log1p(-d/tau)` (always exactly −0.5, guarded by an exact `==` check) and repeated
    `log(ev)` per energy update. Gates 0–3 pass, bit-identical to P5i. Same-session T4 33.12 → 30.83 µs/history (×1.074); FP64
    −12.7%, all instructions −4.0%. Claude's reading (n=3, hypothesis): time tracks FP64 instructions (P5h/P5i), less so once FP64
    share drops (P5j). Bit-preserving FP64 savings are now mostly used up; next candidate is cross-warp event queues.
    Compare speeds only within one session (the same P5i measured 34.41 and 33.12 in two sessions); "≈2.6× over V0" (79.2 → 30.8)
    is a cross-session ratio, roughly 2.5–2.7×. ncu CSV parsing inside Colab failed in all four runs; keep the raw export and
    parse locally.
  - [P5k](docs/validation/mv/p5/RESULTS_P5K.md) (done, **variant F adopted as the current passed configuration**, user decision
    2026-10-17): new GPU drivers only, the P5j transition functions are `#include`d unchanged. F = when a lane's history ends it writes
    its outputs and fetches the next history index (`atomicAdd`), keeping the per-warp ballot; Q1/Q4/Q8/Q1-512 = block-local slots
    in global memory regrouped by state each round. Gates 0–3 pass (gate 1: every history of all 5 variants bit-identical to P5j over
    the 100 saved batches; gate 3 on F). Same-session T4: P5j 30.27, **F 16.51 µs/history (×1.83)**, Q8 19.20, Q4 19.71, Q1 23.98.
    F map: lanes 5.30 → 9.94, warp instructions −48.5%, FP64 −53.6% — the lost lanes were finished histories idling until the warp's
    longest one ended. Q beat F in the equal-cost CPU emulation (0.74 vs 0.39 utilisation) but lost on T4; not profiled (candidates:
    scattered slot access, block barriers, reloading values that F keeps in registers). ≈4.8× over V0 is a cross-session ratio.
    Q map taken later (report only): Q is slow because of the end-of-round `__syncthreads` — 37–47% of stall samples, issue-active
    5.29% (F) → 3.40% (Q1), although Q executes 7–14% fewer warp instructions with more lanes (12.98); memory is not the cause.
    Codex hit its usage limit at the final test run; Claude finished it by user instruction (tests, gates, docs; no source change).
  - [P5l](docs/validation/mv/p5/RESULTS_P5L.md) (done, report only; **no variant beat F; not adopted**, user decision 2026-10-18 — P5k F stays the passed configuration): warp-local
    repacking (`__syncwarp` only, 64/128 slots per warp) and C repeated K=4 times. Gates 0–3 pass (gate 3 on W4K4). Same-session T4:
    F 17.41, FK4 19.32 (+11%), W2K4/W4K4 19.98, GK4 20.07, G1 20.93 µs/history. Map: FK4 lanes 9.94 → 8.29 (lanes that leave C wait
    inside the K loop; +11% warp instructions); W4K4 lanes 9.62 (repacking gain cancelled) and global slots add lg-throttle stalls
    (24%, RNG/driver lines). Equal-cost emulation predicted the opposite for both — do not trust it for K or pool size.
  - **GPU speed-up paused at P5k F** (user decision 2026-10-18). Codex consultation `data/logs/codex_p5_next2.md` (not tracked); lessons in docs/lessons_learned.md (2026-10-18). If resumed: compute resident blocks from registers (allocation granularity), shared memory and launch first — P5l at 130 registers was still 1 block/SM (registers and shared memory). Next stage (e.g. field PDD/OCR on GPU) to be decided with the user.
  - [P6](docs/validation/mv/p6/RESULTS.md) field PDD/OCR on GPU (done, 2026-10-18; plan `docs/ai/plans/2026-10-18-mv-p6-gpu-field.md`): `gpu/p6_*` = P5j transitions + P5k F driver + divergent 10×10 source + P3b region scoring (P5 already carried xyz). Gate 1: the C++ MT build reproduces saved P3b-3 production batches 0/2/99 byte for byte (all 21 outputs). Gates 2–3 on T4 pass. **46M GPU histories vs saved EGS5 and vs saved P3b-3 MT: all 11 pre-fixed quantities inside tolerance** (axis-box blocks −0.25 to +0.31% of `Dref`, SEM ≤ 0.27%; 50% positions ≤ 0.005 cm). T4 24.8 µs/history (slab F 16.8 in the same session). Two plan amendments (rounding bound counts deposits; site 5 is checked by a fixed-state test — 0 hits in 1.38M histories). Colab first stopped because the notebook was fingerprinted (P5g lesson recurred); launch tools are now excluded from identity. Whole-slab deposit ratio GPU/EGS5 (report only, same method as P3b-3): 0–2 cm +0.18 ± 0.08% (the known build-up excess), deeper blocks within ±0.08%.
  - [P7](docs/validation/mv/p7/RESULTS.md) field size / energy variants (done, 2026-10-18; plan `docs/ai/plans/2026-10-18-mv-p7-field-energy-variants.md`): 10 MV 5×5 (A), 10 MV 20×20 (B), Mohan 4 MV 10×10 (C; spectrum `mohan4.spectrum` from NRC EGSnrc, untracked in `~/.cache/chatcarlo/p7/spectra/`). `gpu/p7_*` = P6 + field half-width/SSD as arguments + common ±20 cm layout (3,420 bins); gate 1 replays saved P3b-3 batches byte for byte with the P6 layout. Central means use bins within 80% of the geometric half-width at that depth (user decision; differs from P6's ±4 cm at 10/20 cm). **GPU vs newly run EGS5: all 11 quantities inside tolerance for each condition** (N 20M/138M/26M from 4M pilots ×1.5; cap raised to 240M by user decision). Axis-box blocks −0.28 to +0.62% of `Dref` (B 10–20 cm +0.6% at ≈2.4σ each, not seen in the whole slab). Whole-slab 0–2 cm ratio A +0.03 ± 0.15, B +0.05 ± 0.05, C −0.07 ± 0.09% — the P3a–P6 build-up excess is not seen here (report only). Codex hit its usage limit; Claude implemented by user instruction, judgement code reviewed by Codex in 3 rounds before any verdict. Entry: `run_p7.py` (`local`/`gate1`/`pilot`/`egs`/`export`/`analyse`); EGS5 runs reuse `run_p3b2.run_egs` unchanged with scoped paths/spectrum SHA, one process per condition; EGS5 results are sealed (hashes) at completion.
  - P2 freeze list (2026-10-16): 137 regenerable P1b–P1f output `.npz` files were deleted (user ran `rm` on Claude's
    recommendation, without checking that `docs/validation/mv/p2/manifest.json` `protected` listed them), which stopped every
    input load. By user decision the code was left untouched and those 137 entries were removed from `protected`; their hashes
    are kept in `docs/validation/mv/p2/removed_2026-10-16.json` (831 files still checked). See docs/lessons_learned.md.
  - Colab hand-off: input bundles go to the user's gmail Drive `MyDrive/viveMonte/p5/` via the locally synced folder
    (`~/Library/CloudStorage/GoogleDrive-<gmail>/マイドライブ/...`); results are read back from it. Bundles are never committed.
    Check which account the Drive connector is signed into (it was the university one once) before concluding a file is missing.
    **Browser steps are done by the user** (user decision 2026-10-17, to save tokens): Claude gives the Colab URL, "T4", "Run all"
    and the Drive permission (gmail account) in a few lines, then watches the synced Drive folder for results. Keep every Colab
    step a single "Run all". The Drive mount often times out while the permission is pending; rerunning "Run all" fixes it.
- P3b skeleton and user decisions (2026-10-12; implemented in P3b-1 to P3b-3):
  [docs/ai/plans/2026-10-12-mv-p3b-field-pdd-ocr.md](docs/ai/plans/2026-10-12-mv-p3b-field-pdd-ocr.md) — 10×10 cm² at the
  surface, divergent, 100 cm source-to-surface in vacuum, 2×2 cm axis box, profiles at 5/10/20 cm, depth bins 0.25 cm
  (coarse) / 0.1 cm (fine). **The slab kernels up to P3t do not carry lateral position for photons or for most
  secondaries** (only charged steps and Møller secondaries) — harmless for depth-only scoring, wrong for any lateral
  scoring; use the P3b/P4 kernels for anything lateral.
- Codex review originals for these plans are kept in `docs/ai/plans/reviews/`. Codex's sandbox cannot write outside the
  repository (`~/.cache`); checks that need it are run by Claude.

## Scatter correction moved out (2026-08-20)

The MC scatter-correction research line (Virtual-Grid-like primary/scatter separation and subtraction)
**is no longer part of this repository.** It became its own project at `~/Projects/mcscatter` when its
compute engine was switched from ChatCarlo's own transport to an unmodified `DIDSR/MCGPUv1.3_PCD`
(user decision, 2026-08-20). Moved out: the feasibility study, the roadmap, and the engine-selection plan.

**What stays here, and why**: `chatcarlo/detector.py` (terminal planar detector tally with
primary/scatter discrimination) plus its two test files. It is integrated into `geometry.py`/`source.py`/
`transport.py` and audited (403 tests), so extracting it would be a pure regression risk — and living in a
*separate* codebase is what makes it useful to mcscatter as an **independent** cross-check of MC-GPU's image
tally. Its implementation plan (`docs/ai/plans/2026-08-03-scatter-phase0-detector-tally.md`) also stays,
since it documents a ChatCarlo feature. Do not delete either when tidying up "unused" scatter artifacts.

`docs/ai/plans/2026-08-03-scatter-variance-reduction.md` (NEE/forced-collision, still `draft`) stays too,
but note its original driver moved out: it is now a *ChatCarlo* variance-reduction idea, not a dependency
of the scatter project.

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

**Skills are written to be AI-agnostic (2026-10-07).** `vive-interview`, `vive-check` and `vive-audit` live in
`.claude/skills/` (single source); `.agents/skills/<name>` are symlinks to the same folders so that Codex and
Antigravity pick them up. They do not assume Claude-only features (no hard dependency on AskUserQuestion or
subagents — each skill states what to do when those are unavailable). [AGENTS.md](AGENTS.md) is the shared entry
point for every AI agent and points back to this file as the technical canon. `vive-crosscheck` and the
`egs5-operator` agent are research-only and stay Claude Code-specific; the published system must not depend on EGS5.

There's also `.claude/skills/vive-check/`, a gated workflow skill that runs the four CLI steps in order
(geometry preview → trajectory preview → full run → results) with human approval at each gate. Invoke it for
"walk through the scene with me" style requests rather than chaining the raw CLI calls yourself.

When the user asks for a simulation but no scene.yaml exists yet (or the request is vague), do NOT start writing
a scene directly — invoke `.claude/skills/vive-interview/` first. It elicits the requirements in stages
(purpose → exposure parameters → geometry → run settings), asking the user at each stage, confirming intent and pinning down
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
(primary/scatter discrimination) in [detector.py](chatcarlo/detector.py) — its implementation plan is
[docs/ai/plans/2026-08-03-scatter-phase0-detector-tally.md](docs/ai/plans/2026-08-03-scatter-phase0-detector-tally.md).
**The scatter-correction research line that motivated it now lives in a separate project**
(`~/Projects/mcscatter`, split off 2026-08-20 — see "Scatter correction moved out" below);
`detector.py` stays here as a ChatCarlo feature and serves that project as an *independent*
cross-check of its MC-GPU image tally.

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
mechanism not established), which is **≈46× EGS5's single-threaded `-O` build** on kernel time
(≈39× end-to-end) — the headline comparison basis (user decision, 2026-08-13; the same
document also records ≈8× against an already-parallelized 8-process EGS5 run and ≈25× against
an RNG-replacement upper bound, for readers who want the parallel-CPU baseline instead).
[docs/egs5_crosscheck/speed_comparison/RESULTS.md](docs/egs5_crosscheck/speed_comparison/RESULTS.md)
has the underlying EGS5 throughput measurements.

**Route 2 scoping (2026-08-20, not implemented)**: before actually starting a CUDA port,
[docs/gpu_speedup/route2_scoping/RESULTS.md](docs/gpu_speedup/route2_scoping/RESULTS.md) sized the
work. Key finding: `kernel.py`'s current box/monochromatic/parallel-field/single-worker restriction
blocks *every* real scene under `examples/` for multiple simultaneous reasons (cylinder/sphere
geometry, polychromatic `kvp`, divergent `rect` field, and `mas`/`ctdi_vol_mGy`/`heel_effect`/
`rotation` calibration) — not just geometry as an earlier session's memory note suggested. A CUDA
port therefore isn't the next increment after `kernel.py`; it needs several independent extensions
first, plus a new heightfield primitive in `geometry.py` (not yet built anywhere) before it could
even help the scatter-correction line. No throughput number exists yet for ChatCarlo's own kernel on
GPU — route1's ≈46×/≈8× are an upper-bound reference from an unrelated, mature production code, not
a promise for this codebase.

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
