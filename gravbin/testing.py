"""
Testing code
"""


import numpy as np 
import matplotlib.pyplot as plt

import gravbin as gb


def test_jacobi_extrema(theta, phi, mr, res=10**6):
    """
    Plot the radial v_eff along the passed direction and mass ratio,
    indicating the extrema found by find_jacobi_extrema. res gives
    the relative plotting resolution. Returns figure, axis.
    """
    fig, ax = plt.subplots()
    extrema = gb.find_jacobi_extrema(theta, phi, mr)
    limit = np.absolute(extrema[:, 0]).max()
    r = np.arange(-limit*1.35, limit*1.35, 2*limit/res)
    ax.plot(r, gb.v_eff(r, theta, phi, mr), alpha=0.6,
            linestyle='-', marker='', color='b')
    ax.plot(*extrema.T, marker='o', linestyle='', color='k')
    plt.show()
    return fig, ax


def test_jacobi_barriers(jacobi, theta, phi, mr, res=10**6):
    """
    Plot the radial v_eff along the passed direction and mass ratio,
    indicating the zero-velocity barrier locations found by
    find_jacobi_barriers for the given Jacobi constant. res gives
    the relative plotting resolution. Returns figure, axis.
    """
    fig, ax = plt.subplots()
    roots = gb.find_jacobi_barriers(jacobi, theta, phi, mr)
    extrema = gb.find_jacobi_extrema(theta, phi, mr)
    limit = np.absolute(list(roots) + list(extrema[:, 0])).max()
    r = np.arange(-limit*1.05, limit*1.05, 2*limit/res)
    ax.plot(r, gb.v_eff(r, theta, phi, mr), alpha=0.6,
            linestyle='-', marker='', color='b')
    ax.axhline(jacobi, color='r')
    ax.plot(roots, gb.v_eff(roots, theta, phi, mr),
            marker='o', linestyle='', color='k')
    plt.show()
    return fig, ax