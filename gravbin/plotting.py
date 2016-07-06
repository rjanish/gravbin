"""
Functions for plotting orbits. 
"""

import matplotlib.pyplot as plt
import matplotlib.patches as ptch


def plot_orbits_inertial(binsim, ax=None, alpha=0.6, **kwargs):
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
    # plot test particles
    ax.plot(binsim.paths["pos"][:, 2:, 0], binsim.paths["pos"][:, 2:, 1],
            alpha=alpha, **kwargs)
        # binsim axes: (time, particle, coordinate) 
    # plot heavy binary
    ax.plot(binsim.paths["pos"][:, 0, 0], binsim.paths["pos"][:, 0, 1],
            linestyle='-', marker='', color='k', alpha=alpha*1.3)
    # plot light binary
    ax.plot(binsim.paths["pos"][:, 1, 0], binsim.paths["pos"][:, 1, 1],
            linestyle='-', marker='', color='k', alpha=alpha*1.3)
    ax.plot(*[0, 0], color='k', marker='o', linestyle='',
            alpha=alpha*1.6) # COM
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    return ax.get_figure(), ax

def plot_sim_verbose(binsim, ax=None, **kwargs):
    fig, ax = plot_orbits_inertial(binsim, ax, **kwargs)
    try:
        alpha = kwargs["alpha"]
    except KeyError:
        alpha = 0.6
    bin0_start, bin1_start = binsim.paths["pos"][0, :2, :]
        # binsim axes: (time, particle, coordinate) 
    ax.add_patch(ptch.Circle(bin0_start[:2], binsim.radius0, facecolor='none',
                             alpha=alpha*0.6, linestyle='--'))
    ax.add_patch(ptch.Circle(bin1_start[:2], binsim.radius1, facecolor='none',
                             alpha=alpha*0.6, linestyle='--'))
    ax.add_patch(ptch.Circle([0, 0], binsim.boundary, alpha=alpha,
                             facecolor='none', linestyle='--'))
    ax.plot(binsim.paths["pos"][0, 2:, 0], binsim.paths["pos"][0, 2:, 1], 
            marker='.', linestyle='', color='k', alpha=alpha) # tests initial 
    ax.plot(binsim.colls["test_pos"][:, 0], binsim.colls["test_pos"][:, 1], 
            color='r',marker='.', linestyle='', alpha=alpha) 
    ax.plot(binsim.escps["pos"][:, 0], binsim.escps["pos"][:, 1], 
            color='r',  marker='.', linestyle='', alpha=alpha)
    ax.set_title(binsim.label)
    return fig, ax