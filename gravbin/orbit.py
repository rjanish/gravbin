""" 
This module computes Newtonian gravitational orbits of a
a massless projectile about a equal-mass circular binary.
"""


import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt

import utilities as utl 


class Orbit(object):
    """
    This object computes and stores the projectile orbit. It can also give
    binary positions and various dynamical properties of the projectile.  

    Distance is measured in units of the binary separation, and time in units
    of the binary period. See 'theory.md' for coordinate system and discussion.
    """
    def __init__(self, pos_init, vel_init, bin_init, ccwise, massratio, id):
        """
        Set the initial state of the binary and projectile. Inputs are to be
        given in the inertial COM frame of the binary, using Cartesian
        coordinates with binary oriented as given in 'theory.md'.

        Args:
        pos_init - ndarraylike, shape (3,)
            Initial position of projectile, (x, y, z)
        vel_init - ndarraylike, shape (3,)
            Initial velocity of projectile, (v_x, v_y, v_z)
        bin_init - float
            Initial position of the binary, phi
            (angle of the more-massive binary star in the xy-plane)
        ccwise - bool
            Rotation direction of binary - True if counterclockwise
            and False if clockwise, as viewed from +z direction. 
        massratio - float, in interval [0.5, 1]
            The ratio of the mass of the largest star
            in the binary to the total mass of the binary. 
        id - stringable
            Label for the orbit 
        """
        self.pos_0 = np.asarray(pos_init)
        self.vel_0 = np.asarray(vel_init)
        self.id = str(id)
        self.binary_phi_0 = float(bin_init) % (2*np.pi)
        self.binary_rotdir = 1.0 if bool(ccwise) else -1.0
        self.mr = float(massratio)
        if self.mr < 0.5 or self.mr > 1:
            raise ValueError("Invalid mass ratio: {}. Mass ratio "
                             "must be between 0.5 and 1".format(self.mr))
        print "orbit {}:".format(self.id)
        print ("projectile initialized at\n"
               "  x = {0[0]:0.3f}  v_x = {1[0]:0.3f}\n"
               "  y = {0[1]:0.3f}  v_y = {1[1]:0.3f}\n"
               "  z = {0[2]:0.3f}  v_z = {1[2]:0.3f}"
               "".format(self.pos_0, self.vel_0))
        print ("with binary initialized at\n"
               "  phi = {:0.3f}\n"
               "  rotation: {} (viewed from +z)\n"
               "".format(self.binary_phi_0,
                         "CCW" if self.binary_rotdir else "CW"))
        print "(in binary COM inertial frame)\n"
        # Compute initial states in co-rotating binary frame - see theory.md
        # for complete description of this coordinate system.
            # discrete rotation to put binary along x-axis
        self.corot_pos_0 = np.full(self.pos_0.shape, np.nan)
        self.corot_pos_0[2] = self.pos_0[2]  # z coordinate unchanged 
        self.corot_pos_0[:2] = utl.rotate2d(self.pos_0[:2],
                                            -self.binary_phi_0, form='cart')
        self.corot_vel_0 = np.full(self.vel_0.shape, np.nan)
        self.corot_vel_0[2] = self.vel_0[2]
        self.corot_vel_0[:2] = utl.rotate2d(self.vel_0[:2],
                                            -self.binary_phi_0, form='cart')
            # velocity shift into frame rotating with binary frequency
        self.corot_vel_0[0] +=  self.corot_pos_0[1]*self.binary_rotdir
        self.corot_vel_0[1] += -self.corot_pos_0[0]*self.binary_rotdir
        self.corot_state_0 = np.concatenate((self.corot_pos_0,
                                             self.corot_vel_0)) 

    def D_ode(self, state, time):
        """
        This is the derivative function D_ode(state, t) which
        determines orbits via ODE D[state] = D_ode(state).  
        The state is the concatenation of positions and
        velocities: (x, y, z, v_x, v_y, v_z).

        See theory.md for derivation and coordinate system.
        """
        x, y, z, v_x, v_y, v_z = state
        # compute -distance factors
        dsq_offaxis = y**2 + z**2
        d_heavy = 1 - self.mr - x 
            # distance along binary axis to more massive binary member        
        dsq_heavy = d_heavy**2 + dsq_offaxis
            # distance-squared to more massive binary member
        delta_heavy = dsq_heavy**(-1.5)
        d_light = self.mr + x  # as above, to less massive member
        dsq_light = d_light**2 + dsq_offaxis
        delta_light = dsq_light**(-1.5)
        # compute accelerations
        Dv_x = (2*self.binary_rotdir*v_y + x +
                delta_heavy*self.mr*d_heavy -
                delta_light*(1 - self.mr)*d_light)
        offaxis_forceperdist = delta_heavy*self.mr + delta_light*(1 - self.mr)
        Dv_y = -2*self.binary_rotdir*v_x + y - offaxis_forceperdist*y
        Dv_z = -offaxis_forceperdist*z
        return np.asarray([v_x, v_y, v_z, Dv_x, Dv_y, Dv_z])

    def evolve(self, times):
        """
        Evolve the projectile orbit over all times points given
        in the passed times array. times[0] must be zero. 
            Output will be stored in attributes as:
            X.corot_pos - Cartesian position with time in co-rotating frame
            X.corot_vel - Cartesian velocity with time in co-rotating frame
            X.pos - Cartesian position with time in inertial frame
            X.vel - Cartesian velocity with time in inertial frame
            X.run_stats - internal integrator statistics
        """
        times = np.asarray(times)
        corot_states, run_stats = integ.odeint(func=self.D_ode,
                                               y0=self.corot_state_0,
                                               t=times, mxstep=10000,
                                               full_output=True)
        self.run_stats = run_stats
        self.corot_pos = corot_states[:, :3]
        self.corot_vel = corot_states[:, 3:]
        # convert to inertial frame
        self.pos = np.full(self.corot_pos.shape, np.nan)
        self.pos[:, 2] = self.corot_pos[:, 2]
        self.pos[:, :2] = utl.rotate2d(self.corot_pos[:, :2], 
                                       times + self.binary_phi_0, form='cart')
        self.vel = np.full(self.corot_vel.shape, np.nan)
        self.vel[:, 2] = self.corot_vel[:, 2]
        self.vel[:, :2] = utl.rotate2d(self.corot_pos[:, :2], 
                                       times + self.binary_phi_0, form='cart')
        self.vel[:, 0] += -self.pos[:, 1]*self.binary_rotdir
        self.vel[:, 1] +=  self.pos[:, 0]*self.binary_rotdir


def plot_orbit_inertial(orbit, ax=None, **kwargs):
  """
  Plot the trajectory of the given orbit as seen in the inertial frame. 

  The orbit will be drawn onto the passed matplotlib axes, or the axis of 
  a new figure if none is given. Any maplotilb line attribute kwargs will
  be used to draw the orbit trajectory. The binaries orbit circles are
  drawn as a grayed solid lines with opacity ~ mass. The binary
  center-of-mass is marked by a single black dot.   
  """
  if ax is None:
      fig, ax = plt.subplots()
  ax.plot(orbit.pos[0], orbit.pos[1], **kwargs)
  heavy_transp = 0.8
  light_transp = heavy_transp*(1.0/orbit.mr - 1.0)
      # light_transp/heavy_transp = lighter_mass/heavier_mass 
  heavy_binary_track = plt.Circle((0, 0), 1.0 - orbit.mr, facecolor='none',
                                 edgecolor='k', alpha=heavy_transp)
  light_binary_track = plt.Circle((0, 0), orbit.mr, facecolor='none',
                                 edgecolor='k', alpha=light_transp)
  ax.add_artist(heavy_binary_track)
  ax.add_artist(light_binary_track)
  ax.plot(*[0, 0], color='k', marker='.', linestyle='', alpha=1.0)
  ax.set_aspect("equal")
  return ax


def plot_orbit_corotating(orbit, ax=None, **kwargs):
  """
  Plot the trajectory of the given orbit as seen in the co-rotating frame. 

  The orbit will be drawn onto the passed matplotlib axes, or the axis of 
  a new figure if none is given. Any maplotilb line attribute kwargs will
  be used to draw the orbit trajectory. The binaries positions are given
  by grayed dots with opacity ~ mass. The binary center-of-mass is marked
  by a single black dot. 
  """
  if ax is None:
      fig, ax = plt.subplots()
  ax.plot(orbit.corot_pos[0], orbit.corot_pos[1], **kwargs)
  heavy_transp = 0.8
  light_transp = heavy_transp*(1.0/orbit.mr - 1.0)
      # light_transp/heavy_transp = lighter_mass/heavier_mass 
  heavy_pos = [1.0 - orbit.mr, 0.0]
  light_pos = [orbit.mr, 0.0]
  ax.plot(*heavy_pos, color='k', marker='o', linestyle='', alpha=heavy_transp)
  ax.plot(*light_pos, color='k', marker='o', linestyle='', alpha=light_transp)
  ax.plot(*[0, 0], color='k', marker='.', linestyle='', alpha=1.0)
  ax.set_aspect("equal")
  return ax