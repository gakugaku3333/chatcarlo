#!/usr/bin/env python3
"""Beer--Lambert reference for the fixed 60 keV, 10 cm water-slab scenario.

The attenuation coefficient is obtained from xraylib at run time; no cross
section or transmission value is embedded in this file.
"""
from __future__ import annotations

import math

import xraylib


ENERGY_KEV = 60.0
THICKNESS_CM = 10.0
WATER_DENSITY_G_CM3 = 1.0


def main() -> None:
    mu_over_rho_cm2_g = xraylib.CS_Total_CP("Water, Liquid", ENERGY_KEV)
    mu_cm_inv = mu_over_rho_cm2_g * WATER_DENSITY_G_CM3
    transmission = math.exp(-mu_cm_inv * THICKNESS_CM)
    print(f"energy_keV={ENERGY_KEV:.6g}")
    print(f"thickness_cm={THICKNESS_CM:.6g}")
    print(f"density_g_cm3={WATER_DENSITY_G_CM3:.6g}")
    print(f"mu_over_rho_cm2_g={mu_over_rho_cm2_g:.12g}")
    print(f"mu_cm_inv={mu_cm_inv:.12g}")
    print(f"beer_lambert_transmission={transmission:.12g}")


if __name__ == "__main__":
    main()
