"""
Jacobi constant calculations
"""


import numpy as np
import scipy.optimize as opt

import utilities as utl


def v_eff(r, theta, phi, mr):
    """
    The effective potential in the co-rotating frame. The stars are
    along the x-axis, with the COM at the origin, the more massive star
    star located at x < 0, and their angular momentum along +z. The 
    unit of distance is the separation between the binary stars. Inputs
    are spherical coordinates and the mass ratio mr, the ratio of the
    larger star mass to the total mass (mr in [0.5, 1]).
    """
    chi = np.sin(theta)*np.cos(phi)  # encodes angular position
    delta0 = np.sqrt((r - (mr - 1.0)*chi)**2 + 
                     (1.0 - chi**2)*(mr - 1.0)**2)
    delta1 = np.sqrt((r - mr*chi)**2 + (1.0 - chi**2)*mr**2)
        # distance to the first (most massive) and second binary members
    return -r**2 - mr/delta0 - (1.0 - mr)/delta1


def find_jacobi_extrema(theta, phi, mr):
    """
    Returns the local minima and maxima over radius of the effective
    potential for a fixed direction in the co-rotating frame.  Inputs
    are the direction in spherical coordinates and the mass ratio. 
    Output is a 2D array, with the first column the radial location of
    the extrema and the second column the values of v_eff at those 
    locations. The radius is allowed to be negative. See function
    "v_eff" for coordinate conventions. 

    This relies on some properties of the Jacobi effective potential:
    its extrema are always located closer than a radius of 1.5 and its
    peaks are spaced between the effective star positions. See notes.
    """
    chi = np.sin(theta)*np.cos(phi)  # encodes angular position
    extrema = []
    # find peaks
    minus_v_eff_r = lambda r: -v_eff(r, theta, phi, mr) # max -> min
    dips = np.asarray([(mr - 1)*chi, mr*chi]) # star locations
    dips.sort()
    intervals = [[-1.5, dips[0]], [dips[0], dips[1]], [dips[1], 1.5]]
        # at most three peaks: always one outside each star yet closer than
        # 1.5, and possible another between the two stars. 
    for interval in intervals:
        out = utl.find_local_min(minus_v_eff_r, interval, rtol=10**-4) 
            # scipy 'bounded Brent' tol default is 10^-5, hopefully enough
        if out is not None:  # above returns None if no minimum is found
            pos, value = out # [radius, min function value]
            extrema.append([pos, -value]) # min -> max
    # find valleys
    v_eff_r = lambda r: v_eff(r, theta, phi, mr) 
    peaks = np.array(extrema)[:, 0] # peak locations
    intervals = zip(peaks[:-1], peaks[1:]) # valleys are between peaks
    for interval in intervals:
        out = utl.find_local_min(v_eff_r, interval, rtol=10**-4)
        if out is not None:
            extrema.append(out)
    extrema = np.asarray(extrema)
    extrema = extrema[np.argsort(extrema[:, 0])] # sort by radial position
    return extrema


def find_jacobi_barriers(jacobi, theta, phi, mr):
    """
    For a fixed direction in the co-rotating frame and a given Jacobi 
    constant, find the radial locations of the zero-velocity barrier.
    These are returned as a 1D array. The radius is allowed to be
    negative. See function "v_eff" for coordinate conventions. 
    """
    root_func = lambda r: jacobi - v_eff(r, theta, phi, mr) 
    intervals = [] # bracket each root
    extrema = find_jacobi_extrema(theta, phi, mr)
        # check for possibility of root in each monotonic interval of v_eff
    extrema = extrema[np.argsort(extrema[:, 0])] # sort by radial position
    # find edge intervals
    edge_peaks = extrema[[0, -1]]
    for loc, height in edge_peaks:
        if height < jacobi:
            continue
        guess = np.sqrt(-jacobi) # kosher because height is always negative
        interval = np.sort([loc, np.sign(loc)*guess])
        intervals.append(interval)
    # find central intervals
    possible_intervals = zip(extrema[:-1, 0], extrema[1:, 0])
    for left, right in possible_intervals:
        if np.sign(root_func(left)) != np.sign(root_func(right)):
            intervals.append([left, right]) # sign change => root
    roots = []
    for interval in intervals:
        root = opt.brentq(root_func, *interval)
        roots.append(root)
    return np.asarray(roots)
