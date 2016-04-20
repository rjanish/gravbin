""" 
This module computes Newtonian gravitational orbits of massless test
particles about a binary using the Rebound package.
"""


import numpy as np
import rebound as rb


class BinarySim(object):
    """
    This is a convenience interface for a Rebound simulation and
    a container for its associated data. 

    Units: G = 1, binary total mass = 1, binary reduced semi-major axis
    = 1 (if circular, this is half the binary separation). The binary
    period is 2*pi.  See 'rebound-conventions.ipynb' for details.

    Coordinates: Cartesian coordinates are used, with the origin at the
    binary COM and the binary's angular momentum along the z-axis. At 
    t=0, the more massive body is along the -x axis and the less 
    massive along the +x axis.  
    """
    def __init__(self, mass_ratio=0.5, eccentricity=0.0, radius_1=0.0,
                 radius_2=0.0, boundary_size=100.0, label=None):
        """
        Set the initial state of the binary.

        Args:
        mass_ratio - float in [0.5, 1]
            The ratio of the more massive body to the total binary mass
        eccentricity - float in [0, 1)
            The eccentricity of the binary (e=0 is a circle)
        radius_N, float
            The radius of the binary stars, where N=1 is the more
            massive star.
        boundary - float
            The outer boundary of the simulation.  Particles on
            hyperbolic escape orbits that are greater than this 
            distance from the COM are removed from the simulation.
        label - stringable
            Label for the simulation 
        """
        self.mr = float(mass_ratio)
        self.ecc = float(eccentricity)
        self.radius_1 = float(radius_1)
        self.radius_2 = float(radius_2)
        self.boundary_size = float(boundary_size)
        self.label = str(label)
        # initialize simulation
        self.space_dim = 3
        self.collision = "direct"
        self.boundary = "open"
        self.m1 = self.mr
        self.m2 = 1 - self.mr
        self.sim = rb.Simulation()
        self.sim.exit_max_distance = boundary_size
        self.sim.add(m=self.m1, r=self.radius_1)  
        self.sim.add(m=self.m2, r=self.radius_2, a=1.0, e=self.ecc)
        self.sim.move_to_com()

    def add_test_particles(self, pos, vel):
        """
        Add an array of test particles (mass = 0) to the simulation. 

        Args:
        pos - ndarraylike shape (N, 3)
            The positions of each particle to add, as a sequence
            of Cartesian coordinates (x, y, z).
        vel - ndarraylike shape (N, 3)
            The velocities of each particle to add, as a sequence
            of Cartesian velocities (v_x, v_y, v_z).
        """
        self.starting = np.hstack((pos, vel))
        for n, [(x, y, z), (v_x, v_y, v_z)] in enumerate(zip(pos, vel)):
            self.sim.add(x=x, y=y, z=z, vx=v_x, vy=v_y, vz=v_z,
                         id=(2 + n))  # binary stars have id 0 and 1
            # no mass or radius specified -> m = 0, radius = 0


    def run(self, times):
        """
        Integrate the simulation from the current time to the passed
        times, recording the states of all particles at each time.
        Results are recorded in the paths attribute.

        Args:
        times - 1D ndarraylike 
            Times at which to record particles' state
        """
        self.times = np.array(times, dtype=float)
        num_test_particles = self.sim.N - 2
        num_samples = self.times.size
        test_state_shape = (num_test_particles, self.space_dim, num_samples)
        binary_state_shape = (2, self.space_dim, num_samples)
        self.paths = {"binary":{"pos":np.full(binary_state_shape, np.nan), 
                                "vel":np.full(binary_state_shape, np.nan)},
                        "test":{"pos":np.full(test_state_shape, np.nan), 
                                "vel":np.full(test_state_shape, np.nan)}}
        for time_index, t in enumerate(self.times):
            self.sim.integrate(t) # advance simulation to time t
            particles = {"binary":self.sim.particles[:2], 
                           "test":self.sim.particles[2:]}
            for subsystem, particle_list in particles.iteritems():
                for particle_index, particle in enumerate(particle_list):
                    index = [particle_index, slice(None, None, 1), time_index]
                      # above slice object is equivalent to a ':' index
                    pos = [particle.x, particle.y, particle.z]
                    vel = [particle.vx, particle.vy, particle.vz]
                    self.paths[subsystem]["pos"][index] = pos
                    self.paths[subsystem]["vel"][index] = vel

    # def handle_collision(self):
    #     pass

    # def handle_escape(self):
    #     pass

    # def get_particle_data(self):
    #     """ Get an ndarray of particle spacial coordinates """
    #     ids = np.full(self.sim.N)
    #     coords = np.full((self.sim.N, 3))
    #     for n, p in enumerate(self.sim.particles):
    #         ids[n] = p.id
    #         coords[n, :] = [p.x, p.y, p.z]
    #     return ids, coords

    # def check_events(self):
    #     ids, coords = self.get_particle_coords()
    #     bin1_coords = coords[0]
    #     bin2_coords = coords[1]
    #     test_coords = coords[2:]
    #     test_ids = ids[2:]
    #     # check for escape
    #     test_dist = np.sqrt(np.sum(test_coords**2, axis=1))
    #     outside = test_ids[test_dist > self.boundary_size]
    #     # check for binary - test particle collision
    #     dist_to_1 = np.sqrt(np.sum((test_coords - bin1_coords)**2, axis=1))
    #     collide_1 = dist_to_1 < self.sim.particles[0].radius
    #     dist_to_2 = np.sqrt(np.sum((test_coords - bin2_coords)**2, axis=1))
    #     collide_2 = dist_to_2 < self.sim.particles[1].radius
    #     colliding = test_ids[collide_1 + collide_2]  # element-wise logical or

