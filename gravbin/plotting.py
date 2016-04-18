"""
Functions for plotting orbits. 
"""

import matplotlib.pyplot as plt


def plot_orbits_inertial(binsim, ax=None, **kwargs):
    """
    Plot particle orbits of the passed binary simulation object.

    Orbits are drawn projected onto the binary plane of the inertial
    frame.  The passed matplotlib axes is used, or a new figure is
    made if not given.  Any passed maplotilb line attributes will be
    used to draw the test particle orbits. The binaries orbits are
    drawn as a grayed solid lines with opacity ~ mass. The binary
    center-of-mass is marked by a single black dot.   
    """
    if ax is None:
        fig, ax = plt.subplots()
    orbits = binsim.track
    times = binsim.times
    ax.plot(times, orbits["test"]["pos"].T, **kwargs)
    heavy_alpha = 0.9
    light_alpha = heavy_alpha*(1.0/binsim.mr - 1.0)
        # light_alpha/heavy_alpha = lighter_mass/heavier_mass 
    ax.plot(times, orbits["binary"]["pos"][0, :],
            linestyle='-', marker='', color='k', alpha=heavy_alpha)
    ax.plot(times, orbits["binary"]["pos"][1, :],
            linestyle='-', marker='', color='k', alpha=light_alpha)
    ax.plot(*[0, 0], color='k', marker='o', linestyle='', alpha=1.0)  # COM
    return ax