"""Heitler two-photon annihilation, approved P2c §0; MeV and cm²/electron.
ζ labels the LOWER energy photon: integrate only [ζmin, 1/2].
"""
import numpy as np
from numba import njit
from .physics import M, RE, NE, rotate


@njit
def zeta_min(T):
    g = 1 + T / M
    return 1 / (g + 1 + np.sqrt(g*g - 1))


@njit
def sigma(T):
    if T <= 0:
        raise ValueError('annihilation requires positive kinetic energy')
    g = 1 + T / M
    p = np.sqrt(g*g - 1)
    return np.pi*RE**2/(g+1)*((g*g+4*g+1)/(g*g-1)*np.log(g+p)-(g+3)/p)


@njit
def annihilation_rate(T):
    return NE*sigma(T)


@njit
def differential(T, z):
    g = 1 + T/M
    A = g*g + 4*g + 1
    return np.pi*RE**2/((g+1)*(g*g-1)) * (
        -2*(g+1)**2 + A/(z*(1-z)) - 1/z**2 - 1/(1-z)**2)


@njit
def primitive(T, z):
    g = 1 + T/M
    return (-2*(g+1)**2*z + (g*g+4*g+1)*np.log(z/(1-z))
            + 1/z - 1/(1-z))


@njit
def cdf(T, z):
    lo = zeta_min(T)
    return (primitive(T, z)-primitive(T, lo))/(primitive(T, .5)-primitive(T, lo))


@njit
def sample(T):
    g = 1 + T/M
    lo = zeta_min(T)
    width = -np.log(lo/(1-lo))
    A = g*g + 4*g + 1
    # Proposal proportional to 1/[ζ(1−ζ)]; omitted terms are all negative.
    while True:
        odds = np.exp(-width + width*np.random.random())
        z = odds/(1+odds)
        acceptance = 1 - (2*(g+1)**2*z*(1-z) + (1-z)/z + z/(1-z))/A
        if np.random.random() <= acceptance:
            break
    energy = T+2*M
    mu = (g+1-1/z)/np.sqrt(g*g-1)
    return z*energy, max(-1., min(1., mu)), (1-z)*energy


@njit
def photons(T, ux, uy, uz):
    k1, mu, k2 = sample(T)
    v1 = rotate(ux, uy, uz, mu, 2*np.pi*np.random.random())
    p = np.sqrt(T*(T+2*M))
    x = p*ux-k1*v1[0]; y = p*uy-k1*v1[1]; z = p*uz-k1*v1[2]
    norm = np.sqrt(x*x+y*y+z*z)
    return k1, v1[0], v1[1], v1[2], k2, x/norm, y/norm, z/norm


@njit
def samples(T, n, seed):
    """Same sampled event checked against all three incident directions."""
    np.random.seed(seed)
    zetas = np.empty(n); phis = np.empty(n)
    maxima = np.zeros(3)  # momentum norm, energy residual, unit-length residual
    directions = np.array([[0.,0.,1.], [0.,0.,-1.], [1/3.,2/3.,2/3.]])
    p = np.sqrt(T*(T+2*M))
    for i in range(n):
        k1, mu, k2 = sample(T)
        phi = 2*np.pi*np.random.random()
        zetas[i] = min(k1, k2)/(T+2*M)
        maxima[1] = max(maxima[1], abs(k1+k2-T-2*M))
        for j in range(3):
            u = directions[j]
            v1 = rotate(u[0],u[1],u[2],mu,phi)
            q = np.array([p*u[d]-k1*v1[d] for d in range(3)])
            norm = np.sqrt(np.sum(q*q)); v2 = q/norm
            residual = np.array([k1*v1[d]+k2*v2[d]-p*u[d] for d in range(3)])
            maxima[0] = max(maxima[0], np.sqrt(np.sum(residual*residual)))
            maxima[2] = max(maxima[2], abs(np.sqrt(v1[0]**2+v1[1]**2+v1[2]**2)-1), abs(np.sqrt(np.sum(v2*v2))-1))
            if j == 0:
                phis[i] = np.arctan2(v1[1],v1[0]) % (2*np.pi)
    return zetas, phis, maxima
