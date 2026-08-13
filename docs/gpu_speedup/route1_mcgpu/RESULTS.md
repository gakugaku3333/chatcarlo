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

One warm-up per N (excluded from the table) plus three retained repeats.
Kernel time is MC-GPU's own `>>> Time spent in the Monte Carlo transport only:` report
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

**The kernel-time medians are not trustworthy at better than roughly a factor of two.**
Three concrete problems, all visible in the table above:

1. **The medians are non-monotonic in N.** N=1e7 gives 86.21 M hist/s but N=1e8 gives
   69.93 M hist/s. If the kernel's throughput were stable, the larger N should be at
   least as fast (fixed per-launch costs amortize further), so the ordering is
   physically implausible and indicates timing noise rather than a real N-dependence.
2. **Two single-repeat outliers exist, but which side is biased is not resolved.**
   N=1e7 repeat 3 (0.050 s vs 0.116/0.142 s in repeats 1–2) and N=1e8 repeat 1
   (0.811 s vs 1.430/1.431 s in repeats 2–3) are each ~2–3× faster than their siblings.
   **Correction (post-audit, 2026-08-13)**: an earlier draft of this section treated
   these two as the outliers to discard and named N=1e8's remaining pair (which agree to
   0.07%: 1.430 vs 1.431 s) as "the best-supported point." An independent audit pointed
   out this is not the only reading: the two excluded runs are exactly the two whose
   `gpu_before` utilization read 1% (idle before the run started), while the two runs
   used to support the N=1e8 median started at 59% and 87% utilization (busy). It is
   equally plausible that the idle-start runs are the clean, uncontended measurements and
   the busy-start runs are the ones inflated by contention on a shared free-tier GPU —
   the opposite of what the original wording implied. **This measurement alone cannot
   decide which direction is correct.** The honest statement is: T4 kernel-time
   throughput for this scenario sits somewhere in a 70–90 M hist/s range with roughly a
   factor-of-two uncertainty, and resolving which end is the true uncontended rate would
   need more repeats on a dedicated (non-shared) GPU, which is out of this plan's scope.
3. **Timer quantization matters at small N but does not explain the outliers.** MC-GPU
   prints the transport time to 3 decimals, so at N=1e6 (0.020–0.022 s) the quantization
   alone is ±2.5%. That is far too small to account for the 2–3× outliers above.

**End-to-end numbers do not suffer from this** — they are wall-clock measured by the
notebook, and at N=1e8 the three repeats agree to within 4% (1.701–1.777 s). The
end-to-end rate is also the honest number for an iterative workflow: a fixed ~0.3 s of
context init + I/O per invocation means the GPU advantage collapses at small N (at
N=1e6 the effective rate, 3.0 M hist/s, is *slower* than EGS5 with 8 processes).

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
- vive-audit: pending

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

## Bottom line for the research question

On a free Colab **Tesla T4**, an established, unmodified diagnostic-energy photon
transport GPU code reaches roughly **70–90 M histories/s of pure transport** on this
scenario (a factor-of-two uncertainty that this measurement cannot resolve further —
see "Reliability of the kernel-time numbers" above), and about **59 M histories/s
end-to-end** at N=1e8, which is not subject to that same uncertainty. Using the N=1e8
kernel-time median (69.93 M hist/s, the more conservative end of that range) against the
same scenario's measured EGS5 numbers on the local M3, that is **≈46× the
single-threaded `-O` build, ≈8× the 8-process run, and ≈25× the RNG-replacement upper
bound** on kernel time (≈39× / ≈6.7× / ≈21× end-to-end).

Read against this project's own earlier finding that the EGS5-side ceiling is about 10×
cumulative, the GPU headroom beyond an already-parallelized CPU code is therefore
roughly **one order of magnitude, not two** — the honest comparison is the ≈8× against
8-process EGS5, not the ≈46× against a single thread. Whether that justifies planning
route 2 (a CUDA port of `kernel.py`) is a decision for a separate plan; this plan's
obligation was to produce the number, and the number is now measured rather than assumed.
