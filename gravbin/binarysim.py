""" 
This module computes Newtonian gravitational orbits of massless test
particles about a binary using the Rebound package.
"""


import warnings

import numpy as np
import rebound as rb


class BinarySim(object):
    """
    This is a convenience interface for a Rebound simulation, 
    specialized for simulating a binary + N test particles.

    Units: G = 1, binary total mass = 1, binary reduced semi-major
    axis = 1 (if circular, this is half the binary separation). The
    binary is specified by a mass ratio and eccentricity. The period
    and physical separation depends on the eccentricity, see
    'rebound-conventions.ipynb' for details.

    Coordinates: The origin of time is taken with the binary at its
    closest approach at t=0. Cartesian spacial coordinates are used,
    with the origin at the binary COM. The binary orbit is oriented to
    have its angular momentum along the +z-axis and to have the more
    massive body along the -x axis (and the less massive along +x)
    during closest approach (t=0). 
    """
    def __init__(self, mass_ratio=0.5, eccentricity=0.0, radius_0=0.0,
                 radius_1=0.0, boundary_size=100.0, label=None):
        """
        Set the initial state of the binary.

        Args:
        mass_ratio - float in [0.5, 1]
            The ratio of the more massive body to the total binary mass
        eccentricity - float in [0, 1)
            The eccentricity of the binary (e=0 is a circle)
        radius_N, float
            The radius of the binary stars, where N=0 is the more
            massive star and N=1 the lighter star.
        boundary - float
            The outer boundary of the simulation.  Particles that
            cross this boundary are removed from the simulation.
        label - stringable
            Label for the simulation 
        """
        # binary specifications
        self.mr = float(mass_ratio)
        self.ecc = float(eccentricity)
        self.period = 2*np.pi*(1.0 + self.ecc)**(-1.5)
        self.bin_sep_max = 1 + self.ecc
        self.bin_sep_min = 1 - self.ecc
        self.radius_0 = float(radius_0)
        self.radius_1 = float(radius_1)
        self.boundary_size = float(boundary_size)
        self.label = str(label)
        # initialize simulation
        self.space_dim = 3
        self.m1 = self.mr
        self.m2 = 1 - self.mr
        self.test_starting = None  # the starting states of test particles
        self.num_test_particles = 0
        self.sim = rb.Simulation()
        self.sim.exit_max_distance = self.boundary_size
        self.sim.heartbeat = self.heartbeat  # called every timestep
        self.sim.add(m=self.m1, r=self.radius_0, hash=0)  
        self.sim.add(m=self.m2, r=self.radius_1, hash=1, a=1.0, e=self.ecc)
            # hash must be an unsigned-integer, so binary will carry hashes
            # of 0, 1, with test particles hashes 2, 3, 4, ... 
        self.sim.move_to_com()

    def add_test_particles(self, pos, vel):
        """
        Add an array of test particles (mass = 0) to the simulation. 
        Particle hash number will be assigned by their order in the 
        passed pos and vel arrays: the particle with hash n is added
        with position pos[n] and vel[n].

        Currently, all test particles must be added in one call.

        Args:
        pos - ndarraylike shape (N, 3)
            The positions of each particle to add, as a sequence
            of Cartesian coordinates (x, y, z).
        vel - ndarraylike shape (N, 3)
            The velocities of each particle to add, as a sequence
            of Cartesian velocities (vx, vy, vz).
        """
        if self.sim.N > 2:
            raise RuntimeError("test particles' initial "
                               "conditions are already specified")
        for n, [(x, y, z), (vx, vy, vz)] in enumerate(zip(pos, vel)):
            self.sim.add(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=2+n)
                # binary stars use hash 0, 1; test particles 2, 3, ...
                # mass defaults to 0 (test particle)
                # radius defaults to 0

    def allocate_simulation_trackers(self):
        """ 
        Allocate dicts to track particle trajectories, collisions, and
        escapes.  After all particles have been added and the recording
        times set, the maximum number of records is known so arrays
        can be used for storage. The collision and escape arrays will
        likely contain empty slots after the simulation ends. The
        'empty' initializing value is np.nan for float and -1 for int.
        """
        N_test = self.sim.N - 2
        self.colls = {"time":np.full(N_test, np.nan),
                      "bin_pos":np.full((N_test, 3), np.nan), 
                      "bin_hash":np.full(N_test, -1, dtype=int),
                      "test_pos":np.full((N_test, 3), np.nan), 
                      "test_hash":np.full(N_test, -1, dtype=int)}
        self.coll_cntr = 0
        self.escps = {"time":np.full(N_test, np.nan), 
                      "hash":np.full(N_test, -1, dtype=int), 
                      "pos":np.full((N_test, 3), np.nan), 
                      "vel":np.full((N_test, 3), np.nan)}
        self.escp_cntr = 0
        sim_state_shape = (self.sim.N, self.space_dim, self.times.size)
        self.paths = {"pos":np.full(sim_state_shape, np.nan, dtype="float64"), 
                      "vel":np.full(sim_state_shape, np.nan, dtype="float64")}
        self.cur_pos = np.full((self.sim.N, 3), np.nan)

    def run(self, times):
        """
        Integrate the simulation from the current time to the passed
        times, recording the states of all particles at each passed
        time. Results are recorded in the paths attribute.

        Args:
        times - 1D ndarraylike 
            Times at which to record particles' state

        Sets:
        paths - nested dict of arrays
            paths["test"]["pos"][n] is a (3, N) array giving the
            Cartesian position coordinates as a function of time for 
            the test particle with hash of n.  N is the number of time
            samples taken. "pos" -> "vel" gives analogously the
            Cartesian velocities. "test" -> "binary" gives the position
            or velocity of the binary members, with n=0 the more
            massive body and n=1 the less massive.  NaNs indicate that
            the particle has been removed from the simulation. 
        """
        times = np.asarray(times, dtype=float)
        if times.ndim == 0:
            times.reshape(1)
        elif times.ndim != 1:
            raise ValueError("Passed record times must be 1D or scalar")
        if np.isclose(0.0, times[0], atol=10**(-12)):
            self.times = times
        else:  # force first record time to be t=0
            self.times = np.zeros(times.size + 1, dtype=float)
            self.times[1:] = times
        self.allocate_simulation_trackers()
        # run simulation
        for time_index, t in enumerate(self.times):
            try:
                self.sim.integrate(t) # advance simulation to time t
            except rb.Escape:
                self.process_escape()
            # record simulation state
            hashes = np.zeros(self.sim.N, dtype="uint32")
            pos = np.full((self.sim.N, 3), np.nan, dtype="float64")
            vel = np.full((self.sim.N, 3), np.nan, dtype="float64")
            self.sim.serialize_particle_data(hash=hashes, xyz=pos, vxvyvz=vel)
            self.paths["pos"][hashes, :, time_index] = pos
            self.paths["vel"][hashes, :, time_index] = vel

    def get_all_coords(self, target):
        """ 
        Returns array of target particles' xyz coordinates. Target can
        be either 'binary' or 'test' to fetch the binary's coordinates
        or those of the all test particles.

        This is an idiot-proofing wrapper for self.cur_pos, to prevent 
        confusion of binary members and test particles.
        """
        if target == "binary":
            return self.cur_pos[:2, :]
                # binary always occupies first two indexes (hashes 0 and 1) 
                # test particles occupy later indices (hashes >= 2, with gaps)
        elif target == "test":
            return self.cur_pos[2:, :]
        else:
            raise ValueError("Unrecognized target {}".format(target))

    def update_positions(self):
        """ Set self.cur_pos with all current particle positions """
        pos_array_size = (self.sim.N, 3)
        if ((self.cur_pos is None) or                 # first call
            (self.cur_pos.shape != pos_array_size)):  # particles were removed
            self.cur_pos = np.zeros((self.sim.N, 3), dtype="float64")
        self.sim.serialize_particle_data(xyz=self.cur_pos)

    def heartbeat(self, internal_sim_object):
        """ This function runs every simulation timestep """
        self.update_positions()
        self.check_for_collision()

    def process_escape(self):
        """ 
        Rebound has detected an escaped particle - remove it from simulation
        """
        test_coords = self.get_all_coords("test")
        dist = np.sqrt(np.sum(test_coords**2, axis=-1))
        outside = dist >= self.boundary_size
        test_hashes = self.get_active_test_hashes()
        for index, test_hash in enumerate(test_hashes[outside]):
            test_particle = self.sim.get_particle_by_hash(int(test_hash))
            pos, vel = reboundparticle_to_array(test_particle)
            energy =  (0.5*np.sum(vel**2, axis=-1) - 1.0/dist[outside][index])
            bound = energy < 0.0
            if bound:
                msg = ("time {}: particle {} exited the simulation "
                       "on a *bound* orbit".format(self.sim.t, test_hash))
                warnings.warn(msg, RuntimeWarning)
            self.escps["time"][self.escp_cntr] = self.sim.t
            self.escps["hash"][self.escp_cntr] = test_hash
            self.escps["pos"][self.escp_cntr] = pos
            self.escps["vel"][self.escp_cntr] = vel
            print "t={}: removing {} - escape".format(self.sim.t, test_hash)
            self.sim.remove(hash=int(test_hash))
                # selecting from numpy int array does not return python int,
                # but rather numpy.int32, which fails rebound's type checks 
            self.escp_cntr += 1

    def check_for_collision(self):
        """
        Check for test particle + binary collisions, record and remove
        from the simulation.
        """
        test_coords = self.get_all_coords("test")
        binary_coords = self.get_all_coords("binary")
        # check all test particles against binary member 0
        dist0_sq = np.sum((test_coords - binary_coords[0])**2, axis=-1)
        colliding0 = dist0_sq < self.radius_0**2
        # check all test particles against binary member 1
        bin_sep = np.sqrt(np.sum((binary_coords[0] -
                                  binary_coords[1])**2, axis=-1))
        min_dist0_sq = (bin_sep - self.radius_1)**2
        max_dist0_sq = (bin_sep + self.radius_1)**2
            # min and max distances from binary 0 which
            # can give a collision with binary 1
        to_check = (min_dist0_sq <= dist0_sq) & (dist0_sq <= max_dist0_sq)
        dist1_sq = np.sum((test_coords[to_check] - 
                           binary_coords[1])**2, axis=-1)
        colliding1 = np.zeros(colliding0.shape, dtype=bool)
        colliding1[to_check] = dist1_sq < self.radius_1**2
        # process collisions
        if np.any(colliding0 | colliding1) > 0:
            test_hashes = self.get_active_test_hashes()
            for bin_hash, colliding in zip([0, 1], [colliding0, colliding1]):
                bin_pos = binary_coords[bin_hash]
                for index, test_hash in enumerate(test_hashes[colliding]):
                    # record collision
                    self.colls["time"][self.coll_cntr] = self.sim.t
                    self.colls["test_hash"][self.coll_cntr] = test_hash
                    test_pos = test_coords[colliding][index]
                    self.colls["test_pos"][self.coll_cntr] = test_pos
                    self.colls["bin_hash"][self.coll_cntr] = bin_hash
                    self.colls["bin_pos"][self.coll_cntr] = bin_pos
                    # remove colliding particle
                    print ("t={}: removing {} - collision with binary {}"
                           "".format(self.sim.t, test_hash, bin_hash))
                    self.sim.remove(hash=int(test_hash)) # see process_escape
                    self.coll_cntr += 1

    def get_active_test_hashes(self):
        """ """
        hashes = np.zeros(self.sim.N, dtype="uint32")
        self.sim.serialize_particle_data(hash=hashes)
        test_hashes = hashes[2:]
        return test_hashes


def reboundparticle_to_array(p):
    """
    Convert a Rebound Particle object to an array containing its
    Cartesian position and velocity.  Accepts a Rebound Particle 
    and returns two (3,) ndarrays: pos, vel 
    """
    pos = np.array([p.x, p.y, p.z])
    vel = np.array([p.vx, p.vy, p.vz])
    return pos, vel

