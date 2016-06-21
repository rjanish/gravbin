"""
Compute the phase space distribution of particles launched outward
from the radius of a binary member after escaping to infinity.
"""

import numpy as np 
import gravbin as gb 

import utilities as utl 


def get_radial_outflow(num, radius, speed):
    """
    Draw samples uniform over a sphere with the passed radius that are 
    moving randomly outward with the passed speed and uniform direction
    """
    unit_points_sph = np.ones((num, 3), dtype=float)
    unit_points_sph[:, 1:] = np.array(utl.draw_from_sphere(num)).T 
        # (point, spherical coords) on unit sphere
    unit_points = utl.spherical_to_cart(unit_points_sph) # (point, cartesian)
    points = radius*unit_points
    vel = speed*unit_points
    return points, vel


def get_random_outflow(num, radius, speed):
    """
    Draw samples uniform over a sphere with the passed radius that are 
    moving randomly outward with the passed speed and uniform direction
    """
    unit_points_sph = np.ones((num, 3), dtype=float)
    unit_points_sph[:, 1:] = np.array(utl.draw_from_sphere(num)).T 
        # (point, spherical coords) on unit sphere
    unit_points = utl.spherical_to_cart(unit_points_sph) # (point, cartesian)
    points = radius*unit_points
    unit_vel_sph = np.ones((num, 3), dtype=float)
    unit_vel_sph[:, 1:] = np.array(utl.draw_from_hemisphere(num)).T
        # velocity directions relative to local normal
    vel = speed*utl.spherical_to_cart(unit_vel_sph) # (point, cartesian)
    # generate velocities as uniform vectors over the northern hemisphere,
    # get angle and axis by which to rotate them to local normal hemisphere
    z_hat = np.array([0.0, 0.0, 1.0])
    crosses = np.cross(z_hat, unit_points)
    norms = np.sqrt(np.sum(crosses**2, axis=1))
    rot_axis = (crosses.T/norms).T
    unit_projection = unit_points[:, 2]  # unit_points dot z_hat
    rot_angle = np.arccos(unit_projection)
    utl.rotate3d(vel, rot_angle, rot_axis)
    return points, vel




