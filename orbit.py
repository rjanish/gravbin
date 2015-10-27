""" 
This module computes Newtonian gravitational orbits of a massless
projectile about a circular binary.
"""


import functools

import numpy as np
import scipy.integrate as integ


class Orbit(object):
	"""
	This object computes and stores the orbit. Distance is measured in units
	of the binary spacing, and time in units of the binary period. See
	'theory.md' for derivations and discussion.
	"""
	def __init__(pos_init, vel_init, id):
		"""
		Args:
		pos_init - ndarraylike, shape (3,)
			Initial position of projectile
		vel_init - ndarraylike, shape (3,)
			Initial velocity of projectile
		id - anything
			Identifying label for orbit
		"""
		self.pos_init = np.asarray(pos_init)
		self.vel_init = np.asarray(vel_init)
		self.id = id

	def ode(self, state):
		"""
		System of ODEs defined by:
		  D[state] = ode(state)
		"""
		pass

	def Dode(self, state):
		"""
		Analytic derivative of above ode func - speeds integration.
		"""
		pass

	def evolve(self, times):
		"""
		Evolve the orbit from t=0 to t=time[-1], recording the result
		at each times[i].
		"""
		times = np.asarray(times).flatten() # ensure start is 0?
		orbit, info = integ.odeint(func=self.ode, y0=state_init, t=times,
					 			   Dfun=self.Dode, col_deriv=True,
					 			   full_output=True)
		self.orbit_pos = pass
		self.orbit_vel = pass
		self.orbit_times = times
		self.run_data = info