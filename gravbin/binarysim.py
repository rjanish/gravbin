""" 
This module computes Newtonian gravitational orbits of massless test
particles about a binary using the Rebound package.
"""


import warnings

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
            The outer boundary of the simulation.  Particles that
            cross this boundary are removed from the simulation.
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
        self.m1 = self.mr
        self.m2 = 1 - self.mr
        self.binary_radii = [self.radius_1, self.radius_2]
        self.sim = rb.Simulation()
        self.sim.add(m=self.m1, r=self.radius_1, id=-1)  
        self.sim.add(m=self.m2, r=self.radius_2, id=-2, a=1.0, e=self.ecc)
            # id must be an integer, so binary ids of -1, -2 allows test
            # particle ids to be 0, 1, 2, etc, which simplifies indexing 
        self.sim.move_to_com()
        self.sim.exit_max_distance = self.boundary_size
        self.sim.heartbeat = self.check_for_collision  # called every timestep

    def add_test_particles(self, pos, vel):
        """
        Add an array of test particles (mass = 0) to the simulation. 
        Particle id number will be assigned by their order in the 
        passed pos and vel arrays: the particle with id n is added
        with position pos[n] and vel[n].

        Args:
        pos - ndarraylike shape (N, 3)
            The positions of each particle to add, as a sequence
            of Cartesian coordinates (x, y, z).
        vel - ndarraylike shape (N, 3)
            The velocities of each particle to add, as a sequence
            of Cartesian velocities (vx, vy, vz).
        """
        self.test_starting = np.hstack((pos, vel))
        for n, [(x, y, z), (vx, vy, vz)] in enumerate(zip(pos, vel)):
            self.sim.add(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, id=n)
                # binary stars have id -1 and -2
                # no mass nor radius specified -> m = 0, radius = 0

    def run(self, times):
        """
        Integrate the simulation from the current time to the passed
        times, recording the states of all particles at each time.
        Results are recorded in the paths attribute.

        Args:
        times - 1D ndarraylike 
            Times at which to record particles' state

        Sets:
        paths - nested dict of arrays
            paths["test"]["pos"][n] is a (3, N) array giving the
            Cartesian position coordinates as a function of time for 
            the test particle with id of n.  N is the number of time
            samples taken. "pos" -> "vel" gives analogously the
            Cartesian velocities. "test" -> "binary" gives the position
            or velocity of the binary members, with n=0 the more
            massive body and n=1 the less massive.  NaNs indicate that
            the particle has been removed from the simulation. 
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
        self.escapes = []
        self.collisions = []
        for time_index, t in enumerate(self.times):
            try:
                self.sim.integrate(t) # advance simulation to time t
            except rb.Escape:
                self.process_escape()
            particles = {"binary":self.sim.particles[:2], 
                           "test":self.sim.particles[2:]}
            for subsystem, particle_list in particles.iteritems():
                for particle in particle_list:
                    if subsystem == "binary":
                        particle_index = np.absolute(particle.id) - 1
                            # binary ids are -1 and -2, map these to 0, 1
                    elif subsystem == "test":
                        particle_index = particle.id
                            # test particle ids are 0, 1, 2, ...
                    index = [particle_index, slice(None, None, 1), time_index]
                      # above slice object is equivalent to a ':' index
                    pos = [particle.x, particle.y, particle.z]
                    vel = [particle.vx, particle.vy, particle.vz]
                    self.paths[subsystem]["pos"][index] = pos
                    self.paths[subsystem]["vel"][index] = vel

    def get_all_particle_data(self):
        """ Get array of simulation particles' ids, coords, and velocities """
        ids = np.zeros(self.sim.N, dtype=int)
        coords = np.full((self.sim.N, 3), np.nan)
        vels = np.full((self.sim.N, 3), np.nan)
        for index, particle in enumerate(self.sim.particles):
            ids[index] = particle.id
            coords[index, :] = [particle.x, particle.y, particle.z]
            vels[index, :] = [particle.vx, particle.vy, particle.vz]
        return ids, coords, vels

    def get_binary_data(self):
        """ 
        Returns array of binary members' ids, coords, and velocities.

        This is an idiot-proofing wrapper for get_all_particle_data,
        to prevent confusion of binary members and test particles.
        """
        ids, coords, vels = self.get_all_particle_data()
        return ids[:2], coords[:2], vels[:2]

    def get_test_particle_data(self):
        """ 
        Returns array of test particles' ids, coords, and velocities.

        This is an idiot-proofing wrapper for get_all_particle_data,
        to prevent confusion of binary members and test particles.
        """
        ids, coords, vels = self.get_all_particle_data()
        return ids[2:], coords[2:], vels[2:]

    def process_escape(self):
        """ 
        Rebound has detected an escaped particle - remove it from simulation
        """
        ids, coords, vels = self.get_test_particle_data()
        dist = np.sqrt(np.sum(coords**2, axis=-1))
        outside = dist >= self.boundary_size
        energy =  (0.5*np.sum(vels[outside]**2, axis=-1) - 1.0/dist[outside])
        unbound = energy >= 0.0
        for index, particle_id in enumerate(ids[outside]):
            if not unbound[index]:
                msg = ("time {}: particle {} exited the simulation "
                       "on a *bound* orbit".format(self.sim.t, particle_id))
                warnings.warn(msg, RuntimeWarning)
            escape_record = {"time":self.sim.t,
                             "id":particle_id,
                             "pos":coords[outside][index],
                             "vel":vels[outside][index]}
            self.escapes.append(escape_record)
            print "t={}: removing {} - escape".format(self.sim.t, particle_id)
            self.sim.remove(id=particle_id)

    def check_for_collision(self, internal_sim_object):
        """
        Check for test particle + binary collisions and remove from simulation
        """
        test_ids, test_coords, test_vels = self.get_test_particle_data()
        binary_ids, binary_coords, binary_vels = self.get_binary_data()
        for binary, binary_pos in enumerate(binary_coords):
            dists_sq = np.sum((test_coords - binary_pos)**2, axis=-1)
            colliding = dists_sq < self.binary_radii[binary]**2
            num_collisions = np.sum(colliding)
            if num_collisions > 0:
                for index, particle_id in enumerate(test_ids[colliding]):
                    collision_record = {"time":self.sim.t,
                                        "id":particle_id,
                                        "binary":binary,
                                        "pos":test_coords[colliding][index],
                                        "vel":test_vels[colliding][index]}
                    self.collisions.append(collision_record)
                    print ("t={}: removing {} - collision with binary {}"
                           "".format(self.sim.t, particle_id, binary))
                    self.sim.remove(id=particle_id)


