# P1 water electron transport prototype

Units: MeV and cm; kinetic cutoff/class-II Δ: 0.01 MeV. The [approved preregistration](../../docs/validation/mv/p1/PREREGISTRATION.md) fixes acceptance. No production module imports this prototype.

Collision stopping uses Berger/Seltzer (SLAC-R-730 §§2.13, eqs 2.257–2.268), with I and density effect from fetched NIST ESTAR. The hard part is the analytic first moment of the exact Møller DCS (§2.10), checked by independent quadrature. Molière screening (§2.14), H/O number densities and Z(Z+1) soft scattering are combined before GS Legendre exponentiation (§2.14.2). Q derivatives use a stable continued fraction, not fitted parameters.

The Numba scalar kernel has independent loss/scattering-strength hinges (§§2.15.3–5). Each frame stores position, direction, ledger energy, coefficient energy, initial/residual loss and scattering strength, hard-event optical depth, region, lineage and final-residual flag. Score uses straight-segment overlap with 1D depth bins. All descendants contribute to their primary history and equal-sized independent batches. Radiation is continuous and locally deposited for this comparison only.

## Reproduce saved results

From repository root:

    .venv/bin/python -m pytest tests/ -q
    git diff --stat -- chatcarlo tests scripts examples docs/validation/mv/p0
    .venv/bin/python -m pytest prototypes/mv_electron/tests -q
    .venv/bin/python prototypes/mv_electron/run_checks.py --from-saved

The last command performs no transport/network calls. It reproduces RESULTS.md from saved raw NPZ/EGS5 text and component sampling summaries. Exit 0 means reproducible analysis, not all acceptance gates passed.

## Fresh calculations

Prerequisites: project .venv with NumPy, SciPy, Numba, xraylib, matplotlib, pytest; installed read-only EGS5 distribution/gfortran. Preregistration precedes calculation; do not rewrite it on reruns.

    .venv/bin/python prototypes/mv_electron/fetch_data.py
    .venv/bin/python -m prototypes.mv_electron.component_checks
    .venv/bin/python -m prototypes.mv_electron.gs
    .venv/bin/python -m prototypes.mv_electron.check_gs
    .venv/bin/python -m prototypes.mv_electron.check_fixed
    .venv/bin/python -m pytest prototypes/mv_electron/tests -q
    .venv/bin/python -m prototypes.mv_electron.run_transport new_base_2 -n 1000000
    .venv/bin/python docs/validation/mv/p1/egs5/prepare.py new_egs_base_2 -n 1000000
    .venv/bin/python docs/validation/mv/p1/egs5/run_egs5.py new_egs_base_2

Existing run names are refused. Use new names. Registered analysis names: base_2, base_10, ms_half, energy_half, split_aligned, split_shifted, reference_2.
Adopted f_E=K1_max=0.05. Halves tested independently. 2 MeV: 1.2 cm slab, 0.01 cm bins; 10 MeV: 6 cm slab, 0.05 cm bins. Splits: --split .005 and additionally --shift .002. T3(ii): --mode 1 (individual elastic events with identical loss/Møller/ledger/cutoff model). Diagnostics: --no-hard; --no-boundaries --no-scoring for genuinely unbounded water (zero surface queries). Supplemental extended-slab controls retain a front surface.

EGS5: AP=source kinetic energy, UP=21, UE=20.511 MeV, IUNRST=0. --gs 0 selects Molière; --ms .5 halves CHARD (GS ignores K1HSCL/K1LSCL); --es .5 halves ESTEPR; --nale 100 changes fit interval count. --ap .001 is the single radiation-fluctuation sensitivity. The EPE/NIPE-only test remains an ineffective perturbation. Compilation uses egs5run comp in fresh scratch, then direct foreground exe and os.wait4. The EGS5 tree is never written.

Run python -m prototypes.mv_electron.benchmark after every other calculation has completed. It warms each signature and records three JIT-excluded replicates per energy/control/scoring condition.

Raw NIST HTML/binary third-party PEGS data are ignored; retrieval scripts and hashes remain. manifest.json identifies source, inputs, data and saved results. No commit is made.

Sources: [NIST ESTAR](https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html), [SLAC-R-730](https://rcwww.kek.jp/research/egs/egs5.html) (installed manual), CODATA via scipy.constants, elemental atomic weights via xraylib.
