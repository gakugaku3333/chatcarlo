# Route 1: MC-GPU v1.3_PCD baseline results

Status: **complete** (P1--P4 all executed, 2026-08-13, Google Colab free tier).
Every value below is transcribed from the Drive artifact written by
`route1_mcgpu_colab.ipynb` (revision 6); nothing here is estimated or extrapolated.
The allocated GPU really was a T4, so the headline numbers are labelled as T4 numbers.

**Repository note (2026-08-04, addendum 2 of the plan)**: the originally pinned
`DIDSR/MCGPU` (v1.3) repo ships source code only — no sample data, no material files —
so P1's shipped-sample gate could not pass. Switched (with user approval) to
`DIDSR/MCGPUv1.3_PCD` (same FDA/DIDSR org, CC0, same v1.3 lineage extended for photon
counting detectors), pinned commit `af5fa2888ebbc71ed3aaaf73e8ea8c1b24aea846`, which
ships real pencil/fan-beam samples and real water/air PENELOPE-derived material files.
This route always sets `Nbin<=0` in the IMAGE DETECTOR section, which the source
(`report_image()`) confirms falls back to plain v1.3 Energy-Integrated-Detector
behaviour — the PCD-specific energy-binned output path is never exercised.

## P1 — acquisition, build, and specification confirmation

- Result: **pass** (built with `-arch=sm_75` and provably unchanged MC-GPU source)
- Official repository URL and fixed commit SHA:
  `https://github.com/DIDSR/MCGPUv1.3_PCD.git` @ `af5fa2888ebbc71ed3aaaf73e8ea8c1b24aea846`
  (verified at runtime by `git rev-parse HEAD` against the pinned constant)
- Material-data version and SHA-256: water/air copied unmodified from
  `Sample_Fan_Beam/inputs/{water,air}.mcgpu` in the pinned commit
  - `water.mcgpu`: `7e0501e7e3bf27e918d2f1063b77efebe134b90c3727e485db24bb156d11b458`
  - `air.mcgpu`: `0fe164f58b513d535d1f5ef9bcf0e477c9edb7cc6e2412ab5ce8732e5e282fec`
- Input-file SHA-256:
  - `mono60keV.spc`: `aae657c83974ea35445e627deff9b3d51cb9709b99d0ef2bd1da6c59929baa53`
  - `air10cm.vox`: `bc34a8f5e39dd2491e079c5bd11f20d803837c0f97f669ef03ff34a52d5ab0d5`
  - `water10cm.vox`: `eaafe32114d9a6754887f3313f9430bb702451e76186872d3dc0aede4f782592`
  - `air_control.in`: `3e44044f65df392efca3eaf3567ffa7d87b0c6a62a01928f1858c070f5fc4199`
  - `water_gate.in`: `76e9c183a116c58cf51a7f2a952d7951ad5b840f992c7e947f557290b57085a3`
  - `bench.in`: `974bfbaa0469abae020a05cd7abf184dcc09071a6e615fafd17652ccc82f2498`
- `nvcc --version`: CUDA compilation tools, release 12.8, V12.8.93
  (build `cuda_12.8.r12.8/compiler.35583870_0`, built Fri Feb 21 2025)
- GPU name / compute capability / driver (`nvidia-smi`): **Tesla T4, 7.5, 580.82.07**
- Complete build command:
  ```
  nvcc -m64 -DUSING_CUDA MC-GPU_v1.3_PCD.cu -o MC-GPU_v1.3_PCD.x -O3 -use_fast_math \
       -I. -lz --ptxas-options=-v -I/content/route1_mcgpu/cuda-samples/Common -arch=sm_75
  ```
  Return code 0. `source_unchanged: true` — the SHA-256 of every `*.cu`/`*.h` in the
  repo is identical before and after the build, so no MC-GPU source was edited.
  `ptxas` reports the transport kernel at 70 registers, 5648 B shared memory,
  160 B stack frame, **0 bytes spill stores / 0 bytes spill loads**.
- **Missing-header fallback was required (plan branch 4, 2026-08-04)**: the first build
  attempt failed with
  `./MC-GPU_v1.3_PCD.h:117:12: fatal error: helper_functions.h: No such file or directory`.
  This header belongs to NVIDIA's separately distributed CUDA Samples, not to MC-GPU.
  `NVIDIA/cuda-samples` was cloned `--depth 1` and its `Common/` added as an extra `-I`
  path; the actually-fetched commit is recorded as required by the plan (no commit could
  be pre-registered because the dependency itself was unforeseen):
  **`b7c5481c556c3fe98db060207ecaa41a4b9a9abc`**. The unchanged-source-hash check still
  gated this second attempt and still passed, so the "do not modify MC-GPU" constraint
  held.
- Sample-input run: **pass** — the shipped `Sample_Fan_Beam/fan_beam_simulation.in`
  executed against its own shipped data with only the history count (1e5) and projection
  count (1) reduced by regex; every referenced data file was used unmodified.
- Input format / monoenergetic spectrum / voxel geometry / material energy range /
  RNG and reproducibility: recorded in `state.json` → `p1_spec` (and `P1_specification.txt`,
  the verbatim upstream README). RNG is RANECU with per-thread sequences seeded from
  `seedsMLCG`.
- Source geometry: **confirmed** — point source, rectangular collimated cone beam;
  aperture `0 0` gives an exact zero-divergence pencil beam (verified against the
  shipped `Sample_Pencil_Beam` example; a negative aperture means "cover the whole
  detector", the opposite of narrow — this was misread in an earlier notebook revision)
- Primary tally definition and units: **confirmed** — eV/cm² **per history**
  (`report_image()`'s `NORM` already divides by `total_histories`; an earlier notebook
  revision divided by N a second time, a real bug, fixed before any Colab run)
- Count conversion formula: `T = raw[1] * pixel_area_cm2 / 60000` (no division by N)

## P2 — scenario and physical gate

- Scenario: 60 keV monoenergetic (exact, zero-width spectrum bin), 10 cm water slab,
  zero-divergence pencil beam, detector-centre pixel (1×1 px, 1×1 cm)
- Cone-beam path-length error budget: **not applicable** — the source is an exact
  pencil beam (aperture 0 0), not a cone; an earlier notebook revision had this
  backwards and built an unnecessary error-budget term, since removed
- **Air-only N=1e6 count-conversion control: pass (0.64σ)**

  | quantity | value |
  |---|---|
  | MC-GPU air transmission | 0.99771322 ± 0.00004777 |
  | xraylib analytic Beer--Lambert | 0.99774389 |
  | z | **0.64σ** |

  The control does NOT assert air transmission ≈ 1.0. Air's total interaction cross
  section at 60 keV is Compton-dominated (xraylib `CS_Total_CP` gives μ/ρ=0.1875 cm²/g,
  not the much smaller energy-absorption coefficient), so a 10 cm / 0.00120479 g/cm³ air
  path has a genuine ~0.226% non-scattered loss. An earlier notebook revision asserted
  T≈1.0 and failed a real Colab run at T=0.99771 — a false failure, not a bug in MC-GPU.
- **Water gate: pass** (both gating criteria, N=1e7)

  | comparison | reference T | MC-GPU T | z | gating? |
  |---|---|---|---|---|
  | MC-GPU self-consistency (its own `water.mcgpu` MFP table → μ=0.20693174 /cm) | 0.12627194 | 0.12619692 ± 0.00010501 | **0.71σ** | yes — pass |
  | EGS5 `water60_bound` (IBOUND=1, 5e5 histories) | 0.1270 ± 0.00047 | 0.12619692 ± 0.00010501 | **1.67σ** | yes — pass |
  | xraylib/EPDL analytic (μ=0.20587349 /cm) | 0.12761531 | 0.12619692 ± 0.00010501 | 13.51σ | **no — advisory only** |

- On the xraylib comparison being advisory rather than gating (decided 2026-08-05,
  user-approved, Codex-reviewed): MC-GPU's own PENELOPE-2006 total cross section for
  water at 60 keV differs from xraylib's EPDL-based value by ~0.5% in μ (0.206932 vs
  0.205873 /cm), compounding to ~1.1% in 10 cm transmission. At N=1e7 MC-GPU's own
  statistical SE is tight enough (~0.08% relative) that this known inter-database
  difference alone exceeds 3σ. All three other pre-registered candidate causes were
  independently ruled out: pencil beam confirmed from the shipped sample, the air
  control passes cleanly (which exercises the unit-conversion formula itself), and a
  straight-through path length is geometrically exact. Recorded in `p2_gate.json` as
  `analytic_xraylib_reference_only`.

## P3 — throughput measurement (T4, executed after P2 passed)

One warm-up per N plus three retained repeats. The warm-up rows were originally dropped
(the notebook discards `timed_run()`'s return value for them, so they never reached
`p3_measurements.json`); they are recovered from `execution.log` and reported below,
because they turned out to carry the information that settles how the spread should be
read. Kernel time is MC-GPU's own `>>> Time spent in the Monte Carlo transport only:` report
line — a host-side timer MC-GPU itself brackets around the CUDA kernel launch, printed
with 3 decimal places. End-to-end is process wall clock measured by the notebook.
`context_init_and_io_residual_s` = end-to-end − kernel, bundling host I/O + CUDA context
init + uninstrumentable host work as a single honest residual (MC-GPU is an unmodified
black box; an strace-based split was considered and dropped because strace's own
overhead would bias the timing under measurement).

| N | repeat | kernel s | kernel hist/s | end-to-end s | effective hist/s | residual s | primary T | primary R | FOM = 1/(R²T) | GPU proof (util before→after) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1e6 | 1 | 0.022 | 45.45 M | 0.2843 | 3.518 M | 0.2623 | 0.1256633 | 0.0026378 | 5.06e5 | 14 % → 15 % |
| 1e6 | 2 | 0.020 | 50.00 M | 0.3462 | 2.889 M | 0.3262 | 0.1256633 | 0.0026378 | 4.15e5 | 15 % → 15 % |
| 1e6 | 3 | 0.021 | 47.62 M | 0.3319 | 3.013 M | 0.3109 | 0.1256633 | 0.0026378 | 4.33e5 | 15 % → 17 % |
| 1e7 | 1 | 0.116 | 86.21 M | 0.6255 | 15.987 M | 0.5095 | 0.1261969 | 0.0008321 | 2.31e6 | 35 % → 64 % |
| 1e7 | 2 | 0.142 | 70.42 M | 0.7556 | 13.235 M | 0.6136 | 0.1261969 | 0.0008321 | 1.91e6 | 64 % → 28 % |
| 1e7 | 3 | 0.050 | 200.00 M | 0.8758 | 11.419 M | 0.8258 | 0.1261969 | 0.0008321 | 1.65e6 | 1 % → 16 % |
| 1e8 | 1 | 0.811 | 123.30 M | 1.7772 | 56.269 M | 0.9662 | 0.1262682 | 0.0002631 | 8.13e6 | 1 % → 87 % |
| 1e8 | 2 | 1.430 | 69.93 M | 1.7010 | 58.788 M | 0.2710 | 0.1262682 | 0.0002631 | 8.50e6 | 87 % → 59 % |
| 1e8 | 3 | 1.431 | 69.88 M | 1.7057 | 58.628 M | 0.2747 | 0.1262682 | 0.0002631 | 8.47e6 | 59 % → 47 % |

**All 15 runs of the session, from MC-GPU's own three internal timers** (recovered from
`execution.log`, 2026-08-13 post-review; `total = transport + init/report`, all printed by
MC-GPU itself). The warm-up rows and the `total`/`init+report` split are not in
`p3_measurements.json` and were missing from the first version of this file:

| run (execution order) | total s | transport s | init+report s | transport hist/s |
|---|---:|---:|---:|---:|
| P1 smoke (1e5) | 0.488 | 0.001 | 0.487 | — |
| P2 air (1e6) | 0.411 | 0.002 | 0.410 | — |
| P2 water (1e7) | 0.472 | 0.191 | 0.280 | — |
| 1e6 warm-up | 0.270 | 0.022 | 0.247 | 45.45 M |
| 1e6 r1 / r2 / r3 | 0.261 / 0.275 / 0.289 | 0.022 / 0.020 / 0.021 | 0.239 / 0.255 / 0.268 | 45.45 / 50.00 / 47.62 M |
| 1e7 warm-up | 0.466 | 0.168 | 0.298 | 59.52 M |
| 1e7 r1 / r2 / r3 | 0.384 / 0.417 / 0.330 | 0.116 / 0.142 / 0.050 | 0.269 / 0.275 / 0.280 | 86.21 / 70.42 / 200.00 M |
| **1e8 warm-up** | 0.796 | **0.529** | 0.267 | **189.04 M** |
| **1e8 r1 / r2 / r3** | 1.072 / 1.676 / 1.678 | **0.811 / 1.430 / 1.431** | 0.261 / 0.246 / 0.247 | 123.30 / 69.93 / 69.88 M |

Two things this table shows that the nine-row table above cannot:

- MC-GPU's **own `init+report` timer is essentially constant** (0.239–0.298 s across every
  run from 1e6 to 1e8). That is the trustworthy figure for MC-GPU-internal fixed cost.
  The notebook's `context_init_and_io_residual_s` column is noisier because it also
  contains process spawn, CUDA driver init and Drive-backed file I/O — for `1e8 r1` that
  extra process-level cost was ~0.7 s, against ~0.03 s for r2/r3.
- The N=1e8 slowdown is **real transport time, not timer mis-attribution**: MC-GPU's own
  `total` rises monotonically (0.796 → 1.072 → 1.676 → 1.678) while its `init+report`
  stays flat. (The notebook's end-to-end column happens to look flat at N=1e8 only
  because r1's extra ~0.7 s of process overhead offset its shorter transport.)

Medians and ranges over the three retained repeats:

| N | kernel median | kernel range | effective (end-to-end) median | FOM median | residual median |
|---:|---:|---|---:|---:|---:|
| 1e6 | 47.62 M hist/s | 45.45 – 50.00 M | 3.013 M hist/s | 4.33e5 | 0.311 s |
| 1e7 | 86.21 M hist/s | 70.42 – 200.00 M | 13.235 M hist/s | 1.91e6 | 0.614 s |
| 1e8 | **69.93 M hist/s** | 69.88 – 123.30 M | **58.628 M hist/s** | 8.47e6 | 0.275 s |

Ratios versus the existing local-M3 EGS5 measurements (the pre-registered three denominators;
Colab's CPU is deliberately **not** used as a denominator):

| N | basis | vs EGS5 `-O` 1 proc (1.52 M/s) | vs EGS5 `-O2` 8 proc (8.8 M/s) | vs EGS5+LCG upper bound (2.8 M/s) |
|---:|---|---:|---:|---:|
| 1e6 | kernel median | 31.3× | 5.41× | 17.0× |
| 1e6 | end-to-end median | 2.0× | 0.34× | 1.1× |
| 1e7 | kernel median | 56.7× | 9.80× | 30.8× |
| 1e7 | end-to-end median | 8.7× | 1.50× | 4.7× |
| 1e8 | kernel median | **46.0×** | **7.95×** | **25.0×** |
| 1e8 | end-to-end median | **38.6×** | **6.66×** | **20.9×** |

- GPU-execution confirmation: `gpu_execution_confirmed: true` for all nine retained runs.
  **Correction (post-audit, 2026-08-13)**: an earlier draft of this section claimed the
  `Time spent in the Monte Carlo transport only` line itself was CPU-fallback-proof
  evidence of GPU execution. An independent audit checked the pinned commit's actual
  source and found that `printf` (`MC-GPU_v1.3_PCD.cu:1268`) sits *outside* the
  `#ifdef USING_CUDA` block, so it is printed by a CPU/MPI-only build too — that claim
  was wrong. The real evidence is: the build command recorded in P1 includes
  `-DUSING_CUDA` and `source_unchanged: true` confirms the compiled binary matches that
  command (i.e. the CUDA code path really was compiled in), corroborated by the
  `nvidia-smi` utilization snapshots (column above) changing across each run.
  `memory.used` reads 0 MiB in every snapshot because it is sampled outside the MC-GPU
  process lifetime, so memory is not usable as proof here.
- N=1e8 interruptions or separately resumed allocations: **none** — the whole P1–P4 run
  completed in one Colab session without a disconnect, so no run had to be discarded or
  recorded separately.

### Reliability of the kernel-time numbers (read before quoting any speed-up)

**Revised 2026-08-13 after a post-audit review recovered the warm-up runs from
`execution.log`.** Two earlier framings of this section are now withdrawn: it is *not*
"two random outliers", and the uncertainty is *not* a symmetric 70–90 M hist/s band.
Both were artefacts of reasoning from the nine rows in `p3_measurements.json` alone.

**The finding: at N=1e8, transport time rises monotonically across consecutive runs of
identical work.**

```
1e8 transport:  0.529 s  →  0.811 s  →  1.430 s  →  1.431 s
       rate:   189.0 M   → 123.3 M   →  69.9 M   →  69.9 M   hist/s
               (warm-up)     (r1)        (r2)        (r3)
```

Total spread 2.71×, and it settles: the last two runs agree to 0.07%. As shown in the
15-run table above, MC-GPU's own `total` timer rises with it while its `init+report`
timer stays flat, so this is genuine transport slowdown rather than an artefact of which
timer the work is charged to.

**The honest summary is peak ≈190 M hist/s, sustained ≈70 M hist/s** — not a symmetric
error bar. Which one to quote depends on the question:

- For a **long production run** — the realistic use of an MC speed-up — you get the
  sustained rate, ≈70 M hist/s. **The headline ratios below deliberately use this.**
- For a **short burst on a cold GPU**, ≈190 M hist/s is reachable.

**The mechanism is not established, and the obvious explanation does not fully fit.**
Thermal/power throttling on a passively-cooled 70 W T4, or contention from another tenant
on shared free-tier hardware, would both produce progressive slowdown. But **N=1e7 shows
no such pattern** (0.168 → 0.116 → 0.142 → 0.050 s; the *last* run is the fastest of the
group), and N=1e6 is flat (0.020–0.022 s). A simple "the GPU heats up as the session
proceeds" story would predict a slowdown there too. This route logged neither SM clock
nor temperature, so it cannot separate the candidates — see
"未解明・今後の検証候補" below. `nvidia-smi` reported `memory.used` = 0 MiB in every
snapshot, weak evidence against another tenant holding memory, but those snapshots are
taken outside the MC-GPU process lifetime and settle nothing either way.

Two further points:

1. **The medians are non-monotonic in N** (N=1e7 median 86.21 M > N=1e8 median 69.93 M).
   Under the progressive-slowdown reading this is no longer paradoxical: the N=1e8 group
   ran last and longest, so it is the most affected.
2. **Timer quantization matters at small N but explains none of the above.** MC-GPU
   prints the transport time to 3 decimals, so at N=1e6 (0.020–0.022 s) quantization
   alone is ±2.5% — orders of magnitude too small for a 2.71× spread.

**End-to-end numbers look steadier, but partly by coincidence.** At N=1e8 the three
repeats agree to within 4% (1.701–1.777 s) — however the 15-run table shows why: r1
combined the *shortest* transport (0.811 s) with ~0.7 s of extra process-level overhead,
which happened to land it at the same total as r2/r3. Do not read that 4% agreement as
evidence that the underlying rate was stable; it was not. What the end-to-end column
does support is the practical point about fixed cost: a roughly constant ~0.25 s of
MC-GPU-internal init/report (plus process spawn and Drive I/O on top) per invocation
means the GPU advantage collapses at small N — at N=1e6 the effective rate, 3.0 M hist/s,
is *slower* than EGS5 with 8 processes.

### One statistical caveat about the R and FOM columns

`bench.in` fixes the RANECU seed at `1234567890` for every run, so within each N all
three repeats are bit-identical simulations: `primary_T` and `primary_R` repeat exactly
and the repeats measure **wall-time variance only, not statistical variance**. `primary_R`
is the analytic binomial value √((1−T)/(N·T)), not a batch-based estimator like
ChatCarlo's. FOM therefore varies across repeats only through the end-to-end time.

## P4 — record and audit

- Drive artifact directory: `MyDrive/viveMonte/route1_mcgpu/` on the maintainer's
  personal Google account (the Colab runtime must be mounted from that same account, not
  the institutional one, or the artifacts land in a different Drive), SHA-256 manifest in
  `artifact_sha256.json`
- Raw logs, input files, and JSON retained alongside the notebook in that directory:
  `execution.log`, `state.json`, `p2_gate.json`, `p3_measurements.json`,
  `sha256_manifest.json`, `P1_specification.txt`, and the six generated inputs
  (`mono60keV.spc`, `air10cm.vox`, `water10cm.vox`, `air_control.in`, `water_gate.in`,
  `bench.in`). Key artifact hashes:
  - `p3_measurements.json`: `9c9f3cba6b97a62c447a8785235119a4946a6e71c974f4100e299ffc1d49753e`
  - `p2_gate.json`: `abe4d21994d007edbd20545cca82e3392d1a229a7670585590f6dfa2d4ba13aa`
  - `state.json`: `f5ace44e0147c42e7a5cbddfe5507672bf8f917e5cdc7fda90b26b23c218aac3`
  - `execution.log`: `54c5248b87d88cdc686e344cb12c559d1850f4e4590c513cebb7f01c855c404e`
- Cross-reference added to `docs/egs5_crosscheck/speed_comparison/RESULTS.md`: done
- vive-audit: **passed** (2026-08-13). The auditor independently re-derived every σ in P2,
  re-ran `analytic_reference.py`, re-fetched `water.mcgpu`/`air.mcgpu` from the pinned
  commit and confirmed their SHA-256, re-interpolated the PENELOPE MFP table to 60000 eV,
  and recomputed all P3 medians, ratios, FOM and R — all matching. Two 要注意 findings
  were raised and fixed before the pass: (a) the GPU-execution evidence originally cited
  a `printf` that a CPU-only build also emits (verified against the source: it sits
  outside the `#ifdef USING_CUDA` block), and (b) the outlier-exclusion argument was
  one-directional. Both corrections are marked inline above.
- Post-audit self-review (`/furikaeri`, 2026-08-13) found a third issue the audit could
  not: the warm-up runs were never in `p3_measurements.json`, so both the auditor and this
  document were reasoning from an incomplete dataset. Recovering them from `execution.log`
  changed the uncertainty story from "70–90 M, unresolved" to "peak ≈190 M / sustained
  ≈70 M". The lesson — hand an auditor the raw log location, not only the processed JSON —
  is recorded in `docs/lessons_learned.md`.
- Notebook revision note: the run recorded here was produced by the notebook at SHA-256
  `40bbabb440fcadc0de9bf917fd294f62fcfdb14f30e161f242bf548b21c4929d`. The version now in
  this directory differs only by recording warm-up rows into `p3_measurements.json`
  (so the omission above cannot recur); the transport, timing and gate logic are
  untouched.

## Scope and physics comparability

MC-GPU v1.3_PCD (run in energy-integrated-detector mode, `Nbin<=0`) uses
PENELOPE-2006-derived material data and the RANECU generator; ChatCarlo uses
xraylib-derived attenuation data with bound-Compton and Rayleigh form-factor modelling;
the EGS5 reference uses PEGS5/EGS5 settings with IBOUND=1. The ~0.5% difference in the
water total cross section at 60 keV between PENELOPE-2006 and xraylib/EPDL, quantified in
P2 above, is a concrete instance of this: the three codes are not running identical
physics data even on an identical scenario.

**This is therefore a fixed-scenario throughput comparison gated on a primary
transmission agreement, not a general accuracy-equivalence claim.** The observable is the
primary (non-scattered) transmission only; no scatter-image or dose comparison was made.
The scenario is a single monoenergetic pencil beam through a homogeneous 10 cm water
slab onto a one-pixel detector — the simplest possible geometry, chosen so the physics
gate is unambiguous. Nothing here supports extrapolating the speed-up to polychromatic
spectra, realistic voxelized phantoms, large detectors, or scatter tallies.

## 未解明・今後の検証候補

事前登録した観測量（一次透過率とスループット）は取り切ったので**本計画は完了**である。
以下は本計画のスコープ外として残った項目で、路線2（`kernel.py`のCUDA移植）を検討する
場合にのみ着手すればよい。

1. **N=1e8の単調な性能低下の機序**（上記「Reliability」節）。
   - **仮説**: T4のサーマル/電力スロットリング。
   - **検証方法**: 同一のN=1e8を6反復以上連続実行し、各実行の前後で
     `nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw` を記録する。
   - **判定基準（事前登録）**: SMクロックが実行を追うごとに単調低下し定常値へ漸近すれば
     仮説を支持。**クロックがほぼ一定なのに輸送時間だけ伸びるなら反証**で、その場合は
     他テナントとの競合か、MC-GPU内部の状態依存（メモリ断片化等）を疑う。
   - **既知の反証材料**: N=1e7では同じ低下が見られず（0.168→0.116→0.142→0.050 s）、
     N=1e6は横ばい。単純な発熱蓄積説はこの点を説明できない。この不整合を説明できるまで
     機序を確定と書かないこと。
2. **`execution.log`がセッションごとにtruncateされる点**。ノートブックのcell 1が
   `LOG.write_text()` で開始時に切り詰めるため、前回実行のログは残らない。今回は
   偶然1セッションで完走したため全15実行が揃ったが、途中で再実行していれば
   ウォームアップの証拠は失われていた。追記モードにするか、実行ごとに別名で
   保存する方が安全。
3. **N=1e8での無料枠切断**は今回発生しなかったため、計画が用意していた「切断時は
   別実行として記録するか欠測扱いにする」手順は一度も行使していない（未検証のまま）。

## Bottom line for the research question

On a free Colab **Tesla T4**, an established, unmodified diagnostic-energy photon
transport GPU code sustains roughly **70 M histories/s of pure transport** on this
scenario, with a cold-GPU peak of about **190 M histories/s** that decays to the
sustained rate within three consecutive N=1e8 runs (see "Reliability of the kernel-time
numbers" above — the mechanism is not established). End-to-end at N=1e8 is about
**59 M histories/s**. Quoting the sustained figure (69.93 M hist/s, the N=1e8 median)
against the same scenario's measured EGS5 numbers on the local M3, that is **≈46× the
single-threaded `-O` build, ≈8× the 8-process run, and ≈25× the RNG-replacement upper
bound** on kernel time (≈39× / ≈6.7× / ≈21× end-to-end). The sustained rate is the right
one for production-scale runs; a burst-oriented reading would be up to ≈2.7× more
favourable to the GPU.

**Headline comparison basis (user decision, 2026-08-13): EGS5's single-threaded `-O`
build, i.e. ≈46× on kernel time / ≈39× end-to-end.** This is the number a general EGS5
user is most likely to see reproduced on their own hardware, since EGS5's own default
build/run path (`egs5run`) is single-process — the 8-process figure below reflects this
project's own parallelized usage, not a documented general-EGS5-community default, and no
data was collected on cluster/MPI-scale EGS5 usage (see the "一般的なEGS計算者との比較"
discussion this session — that broader question remains open and would need either a
literature/community survey or a new HPC-scale measurement to answer). The ≈8×/≈25×
figures against the 8-process run and the RNG-replacement upper bound remain recorded
above for readers who want the already-parallelized-CPU baseline instead.

Read against this project's own earlier finding that the EGS5-side ceiling is about 10×
cumulative (compiler flags + process parallelism + RNG replacement), the GPU headroom
beyond an already-parallelized CPU code would be roughly one order of magnitude, not two,
*if* that parallelized baseline is the comparison of interest. Against the single-threaded
default build, GPU headroom is closer to two orders of magnitude (≈46×). Whether either
number justifies planning route 2 (a CUDA port of `kernel.py`) is a decision for a
separate plan; this plan's obligation was to produce the numbers, and they are now
measured rather than assumed.
