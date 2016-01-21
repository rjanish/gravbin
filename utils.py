""" miscellaneous functions """


import numpy as np 


def to_polar(cart):
	"""
	Convert 2D Cartesian coordinates to polar coordinates.

	Args:
	cart - arraylike, shape (N, 2)
		Array of N 2D Cartesian coordinate pairs (x, y)

	Returns:
	polar - ndarray, shape (N, 2)
		Array of N 2D polar coordinate pairs (r, phi)
	"""
	cart = np.asarray(cart, dtype=float)
	x = cart[:, 0]
	y = cart[:, 1]
	r = np.sqrt(x**2 + y**2)
	phi = np.arctan2(y, x)
	return np.array([r, phi]).T