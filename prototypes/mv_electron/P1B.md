# P1b elastic DCS intervention

Approved scope: `docs/ai/plans/2026-10-08-mv-p1b-elastic-dcs.md`.
The fixed before-change source and 10⁴-history M1 run are in `docs/validation/mv/p1b/`.
Do not run the old P1 report writer for this experiment: it writes p1. Never rebuild or rerun EGS5.

- `rutherford`: default, original analytic rates/moments/sampling preserved.
- `dcslib`: reads original EGS5 atomic DCS in place; compound formed before the natural log-log energy spline and log-DCS/RMU spline. Original file angular grid reconstructed from INIGRD. Tables excluded from Git by p1b/.gitignore. Research-only control.
- `eedl`: verified EPICS2025 element bytes. MF23 INT=2, MF26 INT=2 and LANG=12 read and checked. Large-angle conditional density and analytic forward branch are kept separately. Seltzer original equation 7.8 sets eta; no seam continuity adjustment.

The transport event/hinge/stack/loss/scoring rules are unchanged. The only runtime branches select provider rates and single-CDF sampling, with all GS tables from that provider. Every Legendre order used is numerically integrated. Rates and single CDFs use a separate numerical energy grid containing native data knots and converged against the same provider, with a probability mesh resolving the rare single-scatter tails. No atomic interpolation law is changed. In table backends, the exact one-collision GS term is evaluated from the DCS and subtracted from the Legendre series before reconstruction; this is the same GS solution and retains the zero-collision atom. EEDL knots and one-sided branch values are included in angular integration.

```bash
.venv/bin/python -m prototypes.mv_electron.fetch_eedl
.venv/bin/python prototypes/mv_electron/run_p1b.py --backend both
# Wait for all validation computations to finish before timing:
.venv/bin/python prototypes/mv_electron/run_p1b.py --backend both --benchmark-only
.venv/bin/python prototypes/mv_electron/run_checks_p1b.py --from-saved
.venv/bin/python prototypes/mv_electron/run_checks_p1b.py --verify-p1-untouched
```

Existing raw runs are reused. M1 mismatch raises immediately. M2/M4/M5 failure stops that backend before M6/M7; the other backend continues. Statistical failures are reported, not corrected by parameter selection. Results and all input/output hashes are regenerated from saved data without transport or network by `--from-saved`. M4 recorded Monte Carlo and deterministic diagnostics are saved, not silently resimulated during replay.

P1 K5b original test assertions are rerun unchanged with each selected provider. Compiled poison functions assert that no Rutherford rates/single/GS entry points are called by table-backed transport or the fixed-energy reference. Replaying a frozen baseline source directly from its snapshot folder requires resolving the original physics.py ROOT path; the before-change run was recorded in its original location with every source hash verified.

This experiment does not select a production elastic data library and does not validate photon-electron coupled transport.
