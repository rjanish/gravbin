"""
Testing functions for gravbin library, including tests
for both programming logic and accurate physics.  
"""


import time

import numpy as np
import matplotlib.pyplot as plt

import gravbin as gbin


def test_orbit(pos_init, vel_init, massratio, cycles, id='test', res=10**3):
    """
    This is a quick visual check of orbits. 

    A projectile with the given initial state in the inertial frame will
    be evolved for the passed number of binary orbital periods and plotted.
    """
    pos_init = np.asarray(pos_init)
    vel_init = np.asarray(vel_init)
    massratio = float(massratio)
    bin_init = 0.0  # initial binary angle
    ccwise = True   # binary rotates counterclockwise
    orbit = gbin.Orbit(pos_init, vel_init, bin_init, ccwise, massratio, id)
    times = np.linspace(0, cycles*np.pi*2, res*cycles)
    orbit.evolve(times)
    gbin.plot_orbit_inertial(orbit)
    plt.show()
    return orbit


def run_timereversed_pair(pos_init, vel_init, massratio,
                          cycles, res=10**3, id='test_timereversal'):
    """
    Compute a time-reversed pair of orbits.

    A projectile with the given initial conditions will be evolved
    forward for the given number of binary periods, and then evolved
    backwards from the final point for an identical amount of time. 

    Returns the forward and backward orbit objects: forward, backward.
    """
    forward_pos_init = np.asarray(pos_init)
    forward_vel_init = np.asarray(vel_init)
    massratio = float(massratio)
    times = np.linspace(0, cycles*2*np.pi, cycles*res)
    forward_bin_init = 0.0  # initial binary angle
    forward_ccwise = True   # binary rotates counterclockwise
    forward = gbin.Orbit(forward_pos_init, forward_vel_init,
                         forward_bin_init, forward_ccwise,
                         massratio, '{}-forward'.format(id))
    forward.evolve(times)
    backward_pos_init = forward.pos[-1, :]
    backward_vel_init = forward.vel[-1, :]*(-1)
    backward_bin_init = forward.binary_angle[-1]
    backward_ccwise = False
    backward = gbin.Orbit(backward_pos_init, backward_vel_init,
                          backward_bin_init, backward_ccwise,
                          massratio, '{}-backward'.format(id))
    backward.evolve(times)
    return forward, backward