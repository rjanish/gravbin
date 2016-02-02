"""
Testing functions for gravbin library, including tests
for both programming logic and accurate physics.  
"""


import numpy as np
import matplotlib.pyplot as plt

import gravbin as gbin


def test_orbit(pos_init, vel_init, massratio, cycles, id='test', res=10**4):
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
    times = np.linspace(0, cycles, res*cycles)
    orbit.evolve(times)
    gbin.plot_orbit_inertial(orbit)
    plt.show()
    return orbit


def test_res(pos_init, vel_init, bin_init, ccwise,
           cycles=3, res=None):
  """
  Check the convergence of an orbit with increasing integration resolution.
  """
  if res is None:
      res = np.array([100, 1000, 10000])
      markers = ['o', '.', '']
      linestyles = ['', '', '-']
      alphas = [0.4, 0.4, 0.8]
  else:
      res = np.asarray(res)
  orbit = gbin.Orbit(pos_init, vel_init, bin_init, ccwise, 'test')
  fig, ax = plt.subplots()
  for steps, m, ls, a in zip(res, markers, linestyles, alphas):
      t = np.linspace(0, cycles, steps*cycles)
      orbit.evolve(t)
      plot_trajectory(orbit, ax, marker=m, linestyle=ls,
                      label=steps, alpha=a)
  ax.legend(loc='best')
  plt.show()
  return orbit


def test_timereversal(pos_init, vel_init, cycles, res):
  """
  Verify time-reversal symmetry of orbit solutions by computing
  evolving the projectile forward and then backward to starting point.
  """
  t = np.linspace(0, cycles, cycles*res)
  forward = gbin.Orbit(pos_init, vel_init, 0, True, 'forward') # counterclockwise
  forward.evolve(t)
  final_state = forward.states[:, -1]
  final_binary_phi = forward.bin_pos_polar[1, -1]
  backward = gbin.Orbit(final_state[:3], -final_state[3:],
                   final_binary_phi, False, 'backward') # clockwise
  backward.evolve(t)
  delta_pos = np.sqrt(np.sum((backward.pos_cart[:, -1] -
                              forward.pos_cart[:, 0])**2))
  delta_vel = np.sqrt(np.sum((backward.vel_cart[:, -1] +
                              forward.vel_cart[:, 0])**2))
  print "delta_pos = {}".format(delta_pos)
  print "delta_vel = {}".format(delta_vel)
  fig, ax = plt.subplots()
  plot_trajectory(forward,  ax, marker='', linestyle='-', alpha=0.8)
  plot_trajectory(backward, ax, marker='.', linestyle='', alpha=0.6)
  plt.show()
  return forward, backward