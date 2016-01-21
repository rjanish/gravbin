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
	def __init__(self, pos_init, vel_init, bin_init, ccwise, id):
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
			Rotation direction of binary - True if counterclockwise and False
			if clockwise, as viewed from +z direction. 
		id - stringable
			Label for the orbit 
		"""
		self.pos_i = np.asarray(pos_init)
		self.vel_i = np.asarray(vel_init)
		self.id = str(id)
		self.binary_phi_0 = float(bin_init) % 2*np.pi
		self.binary_rotdir = 1.0 if bool(ccwise) else -1.0
		print "orbit {}:".format(self.id)
		print ("projectile initialized at\n"
			   "  x = {0[0]:0.3f}  v_x = {1[0]:0.3f}\n"
			   "  y = {0[1]:0.3f}  v_y = {1[1]:0.3f}\n"
			   "  z = {0[2]:0.3f}  v_z = {1[2]:0.3f}"
			   "".format(self.pos_i, self.vel_i))
		print ("with binary initialized at\n"
			   "  phi = {:0.3f}\n"
			   "  rotation: {} (viewed from +z)\n"
			   "".format(self.binary_phi_0,
			   	         "CCW" if self.binary_rotdir else "CW"))
		print "(in binary COM inertial frame)\n"
		# convert inputs to binary co-rotating frame
		self.corot_pos_i = self.pos_i.copy()  
		self.corot_pos_i[1] = self.corot_pos_i[1] - self.binary_phi_0
			# put binary on x-axis in the co-rotating frame
		self.corot_vel_i	= self.vel_i.copy()
		self.corot_vel_i[1] = self.corot_vel_i[1] - self.binary_rotdir*2*np.pi
		self.corot_state_i = np.concatenate((self.corot_pos_i,
											 self.corot_vel_i)) 

## EVERYTHING BELOW HERE UNCHECKED FOR CARTESIAN SOLUTIONS ##

# 	def deltas(self, r, phi, z):
# 		"""
# 		Compute the inverse cubic distance factors for D_ode.
# 		See 'theory.md' for derivation and conventions.
# 		"""
# 		pythagoras_part = 0.25 + z**2 + r**2
# 		crossterm = r*np.cos(phi)
# 		d1_inv3 = (pythagoras_part - crossterm)**(-1.5)  # \Delta_1^{-3}
# 		d2_inv3 = (pythagoras_part + crossterm)**(-1.5)  # \Delta_2^{-3}
# 		scale_factor = 2*(np.pi**2)
# 		dp_inv3 = (d2_inv3 + d1_inv3)*scale_factor
# 		dm_inv3 = (d2_inv3 - d1_inv3)*scale_factor
# 		return dp_inv3, dm_inv3

# 	def D_ode(self, state, time):
# 		"""
# 		This is the derivative function D[state, t] defining the ODE system
# 		for the projectile orbit via D[state] = ode(state).

# 		state is the concatenation of positions and velocities:
# 			state = (r, phi, z, v_r, v_phi, v_z)
# 		"""
# 		r, phi, z, r_t, phi_t, z_t = state
# 		dp_inv3, dm_inv3 = self.deltas(r, phi, z)
# 		D_r, D_phi, D_z = r_t, phi_t, z_t
# 		w_norm = self.binary_rotdir*(2*np.pi)
# 		D_r_t = -r*dp_inv3 - 0.5*np.cos(phi)*dm_inv3 + r*(phi_t + w_norm)**2
# 		D_phi_t = (0.5*np.sin(phi)*dm_inv3 - 2*r_t*(phi_t + w_norm))/r
# 		D_z_t = -z*dp_inv3
# 		return np.asarray([D_r, D_phi, D_z, D_r_t, D_phi_t, D_z_t])

# 	def evolve(self, times):
# 		"""
# 		Evolve the projectile orbit over all times points given
# 		in the passed times array. times[0] must be zero. 

# 		Output will be stored in attributes as:
# 			X.corot_states - orbital state (pos concatenated with vel) as
# 				a function of time in the binary co-rotating frame
# 			X.states - orbital state with time in inertial COM frame
# 			X.run_stats - internal integrator statistics
# 			X.pos_polar - position with time in polar coordinates
# 			X.vel_polar - velocity with time in polar coordinates
# 			X.pos_cart - position with time in Cartesian coordinates
# 			X.vel_cart - velocity with time in Cartesian coordinates
# 		"""
# 		times = np.asarray(times)
# 		corot_states, run_stats = integ.odeint(func=self.D_ode,
# 											   y0=self.corot_state_i,
# 		                                       t=times, mxstep=10000,
# 		                                       full_output=True)
# 		self.corot_states = corot_states.T  # (orbital states, time)
# 		self.run_stats = run_stats
# 		# process results
# 		states = self.corot_states.copy()  # go to inertial frame
# 		states[1, :] = states[1, :] + self.binary_rotdir*2*np.pi*times + self.binary_phi_0
# 		states[4, :] = states[4, :] + self.binary_rotdir*2*np.pi
# 		self.times = times.copy()
# 		self.states = states
# 		self.pos_polar = self.states[:3, :]
# 		self.vel_polar = self.states[3:, :]
# 		self.pos_cart = np.zeros(self.pos_polar.shape)
# 		self.pos_cart[0] = self.pos_polar[0]*np.cos(self.pos_polar[1])
# 		self.pos_cart[1] = self.pos_polar[0]*np.sin(self.pos_polar[1])
# 		self.pos_cart[2] = self.pos_polar[2]
# 		self.vel_cart = np.zeros(self.pos_cart.shape)
# 		self.vel_cart[0] = (
# 			self.vel_polar[0]*np.cos(self.pos_polar[1]) -
# 			self.pos_polar[0]*self.vel_polar[1]*np.sin(self.pos_polar[1]))
# 		self.vel_cart[1] = (
# 			self.vel_polar[0]*np.sin(self.pos_polar[1]) +
# 			self.pos_polar[0]*self.vel_polar[1]*np.cos(self.pos_polar[1]))
# 		self.vel_cart[2] = self.vel_polar[2]
# 		# save binary positions (only for 1 - other at phi + pi)
# 		self.bin_pos_polar = np.zeros(self.pos_polar.shape)
# 		self.bin_pos_polar[0] = 1.0
# 		self.bin_pos_polar[1] = self.binary_phi_0 + self.binary_rotdir*2*np.pi*times
# 		self.bin_pos_polar[2] = 0.0
# 		self.bin_pos_cart = np.zeros(self.bin_pos_polar.shape)
# 		self.bin_pos_cart[0] = np.cos(self.bin_pos_polar[1])
# 		self.bin_pos_cart[1] = np.sin(self.bin_pos_polar[1])
# 		self.bin_pos_cart[2] = 0.0


# def plot_trajectory(orbit, ax=None, marker='.', linestyle='-',
# 				    alpha=0.6, label=None):
# 	"""
# 	Plot the trajectory of the passed orbit onto the given matplotlib axes.
# 	"""
# 	if ax is None:
# 		fig, ax = plt.subplots()
# 	if label is None:
# 		label = orbit.id
# 	ax.plot(orbit.pos_cart[0], orbit.pos_cart[1], marker=marker,
# 			linestyle=linestyle, alpha=alpha, label=label)
# 	binary_trajectory = plt.Circle((0, 0), 1.0, facecolor='none',
# 								   edgecolor='k', alpha=0.8)
# 	ax.add_artist(binary_trajectory)
# 	ax.set_aspect("equal")
# 	return ax


# def test_orbit(pos_init, vel_init, bin_init, ccwise, cycles=3, res=100):
# 	"""
# 	This is a quick sanity check on the appearance of orbits. A
# 	projectile with the given initial state will be evolved for the
# 	given number of binary cycles, and resulting trajectory plotted.
# 	"""
# 	orbit = Orbit(pos_init, vel_init, bin_init, ccwise, 'test')
# 	t = np.linspace(0, cycles, res*cycles)
# 	orbit.evolve(t)
# 	plot_trajectory(orbit)
# 	plt.show()
# 	return orbit


# def test_res(pos_init, vel_init, bin_init, ccwise,
# 			 cycles=3, res=None):
# 	"""
# 	Check the convergence of an orbit with increasing integration resolution.
# 	"""
# 	if res is None:
# 		res = np.array([100, 1000, 10000])
# 		markers = ['o', '.', '']
# 		linestyles = ['', '', '-']
# 		alphas = [0.4, 0.4, 0.8]
# 	else:
# 		res = np.asarray(res)
# 	orbit = Orbit(pos_init, vel_init, bin_init, ccwise, 'test')
# 	fig, ax = plt.subplots()
# 	for steps, m, ls, a in zip(res, markers, linestyles, alphas):
# 		t = np.linspace(0, cycles, steps*cycles)
# 		orbit.evolve(t)
# 		plot_trajectory(orbit, ax, marker=m, linestyle=ls,
# 						label=steps, alpha=a)
# 	ax.legend(loc='best')
# 	plt.show()
# 	return orbit


# def test_timereversal(pos_init, vel_init, cycles, res):
# 	"""
# 	Verify time-reversal symmetry of orbit solutions by computing
# 	evolving the projectile forward and then backward to starting point.
# 	"""
# 	t = np.linspace(0, cycles, cycles*res)
# 	forward = Orbit(pos_init, vel_init, 0, True, 'forward') # counterclockwise
# 	forward.evolve(t)
# 	final_state = forward.states[:, -1]
# 	final_binary_phi = forward.bin_pos_polar[1, -1]
# 	backward = Orbit(final_state[:3], -final_state[3:],
# 					 final_binary_phi, False, 'backward') # clockwise
# 	backward.evolve(t)
# 	delta_pos = np.sqrt(np.sum((backward.pos_cart[:, -1] -
# 						        forward.pos_cart[:, 0])**2))
# 	delta_vel = np.sqrt(np.sum((backward.vel_cart[:, -1] +
# 						        forward.vel_cart[:, 0])**2))
# 	print "delta_pos = {}".format(delta_pos)
# 	print "delta_vel = {}".format(delta_vel)
# 	fig, ax = plt.subplots()
# 	plot_trajectory(forward,  ax, marker='', linestyle='-', alpha=0.8)
# 	plot_trajectory(backward, ax, marker='.', linestyle='', alpha=0.6)
# 	plt.show()
# 	return forward, backward