"""
Built matplotlib animations of binary-projectile orbits
"""


import matplotlib.pyplot as plt
import matplotlib.animation as ani


class Animator(object):
	"""
	"""
	def __init__(self, orbit):
		"""
		"""
		self.proj_pos = orbit.pos_cart
		self.bin_pos = orbit.bin_pos_cart
		self.times = orbit.times
		self.fig, self.ax = plt.subplots()
		bin_track = plt.Circle((0, 0), 1.0, facecolor='none',
							   edgecolor='k', alpha=0.8)
		self.ax.add_artist(bin_track)
		self.ax.set_aspect("equal")
		self.ax.set_xlim(-1.5, 1.5)
		self.ax.set_ylim(-1.5, 1.5)
		self.binary, = plt.plot([], [], marker='o', linestyle='-', color='k')
		self.proj_track, = plt.plot([], [], marker='', linestyle='-', color='b')
		self.proj_cur, = plt.plot([], [], marker='.', linestyle='', color='b')

	def init_animate(self):
		"""
		"""
		# self.binary.set_data([], [])
		self.proj_track.set_data([], [])
		self.proj_cur.set_data([], [])
		return self.binary, self.proj_track, self.proj_cur

	def animate(self, i):
		"""
		"""
		other_bin = self.bin_pos
		self.binary.set_data([self.bin_pos[0, i], -self.bin_pos[0, i]],
							 [self.bin_pos[1, i], -self.bin_pos[1, i]])
		self.proj_track.set_data(*self.proj_pos[:2, :(i + 1)])
		self.proj_cur.set_data(*self.proj_pos[:2, i])
		return self.binary, self.proj_track, self.proj_cur
	def run(self):
		"""
		"""
		animated = ani.FuncAnimation(self.fig, self.animate, frames=self.times.size,
									 blit=True, init_func=self.init_animate,
									 repeat=True, interval=.5)
		plt.show()

