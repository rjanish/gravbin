"""
Jacobi constant calculations
"""


import numpy as np
import scipy.optimize as opt

import utilities as utl


def jacobi_corotating(v, r, theta, phi, mr):
    """ Jacobi constant in the co-rotating frame """
    return v**2 + v_eff(r, theta, phi, mr)


def jacobi_inert_asymp(x, y, z, v_x, v_y, v_z):
    """
    Return the Jacobi constant in the inertial frame, assuming the
    particle's distant to be large enough that any contributions from 
    the potential of the stars can be ignored. Input is the Cartesian
    position and velocity.

    In the inertial frame I use the Jacobi constant:
        J = 2(E - L_z)/m
          = v^2 - 2 [\vec{r} \cross \vec{v}]_z for r -> infinity
    """
    pos, vel = [x, y, z], [v_x, v_y, v_z]
    return np.sum(vel**2) - 2*np.cross(pos, vel)


def jacobi_scattering(b, v, theta, phi):
    """
    Jacobi constant of particles incident from x = - infinity, with y-
    position b, velocity v, and velocity direction theta, the angle
    between the velocity and the z-axis. I assume that v_y = 0 and
    v_x > 0 (if not, could always rotate it so).
    """
    return v**2 + 2*b*v*np.sin(theta)


def v_eff(r, theta, phi, mr):
    """
    The effective potential in the co-rotating frame. The stars are
    along the x-axis, with the COM at the origin, the more massive star
    star located at x < 0, and their angular momentum along +z. The 
    unit of distance is the separation between the binary stars. Inputs
    are spherical coordinates and the mass ratio mr, the ratio of the
    larger star mass to the total mass (mr in [0.5, 1]).
    """
    r = np.asarray(r, dtype=float)
    mr = float(mr)
    chi = np.sin(theta)*np.cos(phi)  # encodes angular position
    if np.isclose(mr, 1.0): # only one star
        return -r**2.0 - 2.0/np.absolute(r)
    elif np.isclose(chi, 0.0): # along y or z axis
        delta0 = np.sqrt(r**2 + (mr - 1.0)**2)
        delta1 = np.sqrt(r**2 + mr**2)
        # distance to the first (most massive) and second binary members
    elif np.isclose(chi, 1.0): # along x axis
        delta0 = np.absolute(r - (mr - 1.0))
        delta1 = np.absolute(r - mr)
    else: 
        delta0 = np.sqrt((r - (mr - 1.0)*chi)**2 + 
                         (1.0 - chi**2)*(mr - 1.0)**2)
        delta1 = np.sqrt((r - mr*chi)**2 + (1.0 - chi**2)*mr**2)
    return -r**2.0 - 2.0*mr/delta0 - 2.0*(1.0 - mr)/delta1


def find_jacobi_extrema(theta, phi, mr, atol=10**-6):
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
    if np.isclose(mr, 1.0):
        # use analytic result
        return np.array([[-1.0, -3.0], [0.0, -np.inf], [1.0, -3.0]])
    extrema = []
    # find peaks
    minus_v_eff_r = lambda r: -v_eff(r, theta, phi, mr) # max -> min
    dips = np.asarray([(mr - 1.0)*chi, mr*chi]) # star locations
    dips.sort()
    intervals = [[-1.5, dips[0]], [dips[0], dips[1]],[dips[1], 1.5]]
        # at most three peaks: always one outside or at each star yet within
        # radius of 1.5, and possible another between the two stars. 
    for interval in intervals:
        out = utl.find_local_min(minus_v_eff_r, interval, atol) 
        if out is not None:  # above returns None if no minimum is found
            pos, value = out # [radius, min function value]
            extrema.append([pos, -value]) # min -> max
    # find valleys
    v_eff_r = lambda r: v_eff(r, theta, phi, mr) 
    peaks = np.array(extrema)[:, 0] # peak locations
    intervals = zip(peaks[:-1], peaks[1:]) # valleys are between peaks
    hole0, hole1 = (mr - 1.0)*chi, mr*chi # loc of possible singularities
    for interval in intervals:
        if np.isclose(mr, 1.0) and utl.in_linear_interval(hole0, interval):
            # when mr = 1, heavier star is at the origin and all directions
            # pass through the star => singularity at 0
            extrema.append([hole0, -np.inf])
            continue
        if np.isclose(chi, 1.0):
            # when chi = 1, direction is along x-axis and passed though both
            # stars => singularity at both holes
            if utl.in_linear_interval(hole0, interval):
                extrema.append([hole0, -np.inf])
                continue
            if utl.in_linear_interval(hole1, interval):
                extrema.append([hole1, -np.inf])
                continue
        out = utl.find_local_min(v_eff_r, interval, atol)
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
    trial_intervals = np.array(zip(extrema[:-1], extrema[1:]))
        # 3D array: intervals, endpoints, radius/veff 
    for entry in trial_intervals:
        interval = entry[:, 0]
        root_func_interval = jacobi - entry[:, 1] # root_func = J - v_eff
        if np.sign(root_func_interval[0]) != np.sign(root_func_interval[1]):
            # sign change => root
            singular = np.isinf(root_func_interval)
            if singular.any():
                singularity = interval[singular]
                midpoint = 0.5*np.sum(interval)
                interval[singular] = midpoint
                    # cut interval in half, removing the side with a singular
                    # endpoint (note: there can only be one singular endpoint)
                while (np.sign(root_func(interval[0])) == 
                       np.sign(root_func(interval[1]))):
                       # root was removed, expand to singularity to recover
                    interval[singular] += (interval[singular] + singularity)/2
            intervals.append(interval) 
    roots = []
    for interval in intervals:
        root = opt.brentq(root_func, *interval)
        roots.append(root)
    return np.asarray(roots)


def barrier_in_plane(jacobi, normal, mr, res=10**2):
    """
    Returns a 2d array shape (res, 3), giving res samples in spherical
    coordinates of the zero-velocity barrier in the plane orthogonal
    to the passed normal.
    """
    xy_uc = np.ones((res, 3)) # unit semi-circle in xy plane, spherical coords
    xy_uc[:, 1] = np.pi/2
    xy_uc[:, 2] = np.linspace(0, np.pi, res)
    xy_normal = np.array([0, 0, 1])
    desired_normal = np.asarray(normal)
    desired_normal = desired_normal/np.sqrt(np.sum(desired_normal**2))
    angle = np.arccos(np.dot(xy_normal, desired_normal))
    axis = np.cross(desired_normal, xy_normal)
    if angle > 0:
        xy_uc_cart = utl.spherical_to_cart(xy_uc)
        desired_uc_cart = utl.rotate3d(xy_uc_cart, angle, axis)
        desired_uc = utl.cart_to_spherical(desired_uc_cart)
            # unit circle in desired plane, spherical coordinates
    else: 
        desired_uc = xy_uc # already in correct plane
    barriers = np.zeros((res*6, 3))  # at most, 6 barriers per direction
    num_barriers = 0
    for theta, phi in desired_uc[:, 1:]:
        found = find_jacobi_barriers(jacobi, theta, phi, mr)
        num_found = len(found)
        barriers[num_barriers:num_barriers+num_found, 0] = found
        barriers[num_barriers:num_barriers+num_found, 1] = theta
        barriers[num_barriers:num_barriers+num_found, 2] = phi
        num_barriers += num_found
    barriers = barriers[:num_barriers, :]
    return barriers


def barrier_cutout(jacobi, mr, res=10**3):
    """
    Sample the y > 0 hemisphere of the Jacobi zero-velocity barrier
    """
    zplus = np.ones((res, 3)) # unit z > 0 hemisphere, spherical coords
    zplus[:, 1], zplus[:, 2] = utl.draw_from_hemisphere(res)
    angle = -np.pi/2
    axis = np.array([1, 0, 0]) # rotate samples to y > 0 hemisphere
    zplus_cart = utl.spherical_to_cart(zplus)
    yplus_cart = utl.rotate3d(zplus_cart, angle, axis)
    yplus = utl.cart_to_spherical(yplus_cart)
    inner_barriers = np.zeros((res*3, 3))  # at most, 3 inner barriers
    outer_barriers = np.zeros((res, 3))  # at most, 1 outer barrier
    num_inner, num_outer = 0, 0
    for theta, phi in yplus[:, 1:]:
        found = find_jacobi_barriers(jacobi, theta, phi, mr)
        y_coord = found*np.sin(theta)*np.sin(phi)
        found = found[y_coord >= 0.1] # cut out xz plane 
        total_found = len(found)
        if total_found == 0:
            continue
        outer = max(found)
        outer_barriers[num_outer, 0] = outer
        outer_barriers[num_outer, 1] = theta
        outer_barriers[num_outer, 2] = phi
        num_outer += 1
        num_inner_found = total_found - 1
        if num_inner_found > 0:
            inners = np.sort(found)[:-1]
            inner_barriers[num_inner:num_inner+num_inner_found, 0] = inners
            inner_barriers[num_inner:num_inner+num_inner_found, 1] = theta
            inner_barriers[num_inner:num_inner+num_inner_found, 2] = phi
            num_inner += num_inner_found
    outer_barriers = utl.spherical_to_cart(outer_barriers[:num_outer, :])
    inner_barriers = utl.spherical_to_cart(inner_barriers[:num_inner, :])
    return inner_barriers, outer_barriers