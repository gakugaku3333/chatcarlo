# EPICS2025 EEDL inputs

ZA001000 and ZA008000 are unmodified EPICS2025 Evaluated Electron Data Library (EEDL) files, evaluated by D. E. Cullen, distributed by Lawrence Livermore National Laboratory; EPICS is maintained by Caleb Mattoon (LLNL).

Source: https://nuclear.llnl.gov/EPICS/ (element links under ENDF format EEDL).
License: CC Attribution 4.0 International, https://nuclear.llnl.gov/EPICS/CC%20Attribution%204.0%20Intl%20Public%20License.pdf (local license.pdf).

License body checked: section 2(a)(1) permits reproduction and sharing; section 3(a) requires attribution, license reference, and modification notice; section 4 covers database reuse. Original element bytes are unchanged. Parsed/interpolated effective DCS and numerical results are derived calculations, not replacements of the original evaluations. No endorsement by LLNL or the evaluators is implied. Checksums and exact download URLs are in eedl_provenance.json.

EEDL1991.pdf was read locally for provenance; it is excluded from tracking (separate report copyright, not assumed licensed as the data library).

# Interpolation / screening provenance

ENDF-102, MF26 LAW=2 and LANG=12 specify tabulated linear-linear angular interpolation. MF23 TAB1 and MF26 TAB2 in these files explicitly specify INT=2 (linear-linear energy interpolation). Both distributions are interpolated on the union of cosine grids without rescaling.

EEDL1991, Elastic Scattering Cross Sections (printed page ix), identifies Molière screening with Seltzer's empirical correction and cites Seltzer, *An Overview of ETRAN Monte Carlo Methods*, 1988, chapter 7. The original cited chapter was obtained and Eq. 7.8 on printed page 160 visually checked (reference_provenance.json); the same explicit formula is reproduced in the primary LANL MCNP5 manual, LA-UR-03-1987 (revised 2008), page 2-77:
https://mcnp.lanl.gov/pdf_files/TechReport_2003_LANL_LA-UR-03-1987Revised212008_SweezyBoothEtAl.pdf

eta_M = (alpha*m/(0.885*p))^2 * Z^(2/3) / 4 * [1.13 + 3.76*(alpha*Z/beta)^2 * sqrt(tau/(tau+1))].

Here tau=T/m, p²=T(T+2m), beta²=p²/(T+m)². The EEDL convention A/(eta+1-mu)^2 uses eta=2*eta_M. This empirical factor differs from P1's unmodified Molière screening. No EGS5-based parameter adjustment is used. The older MCNP4C manual prints a different empirical factor; it is not used as an alternative fit. EEDL1991 does not itself print the explicit correction formula, the explicitly cited Seltzer original Eq. 7.8 was therefore read directly in addition to the ENDF contents.
