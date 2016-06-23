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
    paths = binsim.paths
    times = binsim.times
    test_x = paths["pos"][2:, 0].T  # (time, particle)
    test_y = paths["pos"][2:, 1].T
    ax.plot(test_x, test_y, **kwargs)
    heavy_x = paths["pos"][0, 0].T
    heavy_y = paths["pos"][0, 1].T
    ax.plot(heavy_x, heavy_y, linestyle='-', marker='', color='k', alpha=0.8)
    light_x = paths["pos"][1, 0].T
    light_y = paths["pos"][1, 1].T 
    ax.plot(light_x, light_y, linestyle='-', marker='', color='k', alpha=0.8)
    ax.plot(*[0, 0], color='k', marker='o', linestyle='', alpha=1.0)  # COM
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    return ax