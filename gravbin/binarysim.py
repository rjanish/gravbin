""" 
This module computes Newtonian gravitational orbits of massless test
particles about a binary using the Rebound package.
"""


import warnings

import numpy as np
import matplotlib.pyplot as plt
import rebound as rb

import utilities as utl
import gravbin as gb


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
    def __init__(self, mass_ratio=0.5, eccentricity=0.0, radius0=0.0,
                 radius1=0.0, boundary=100.0, label="binsim", verbose=False):
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
        verbose - bool, default False
            If True, print simulation updates to stdout
        """
        # binary specifications
        self.mr = float(mass_ratio)
        self.ecc = float(eccentricity)
        self.period = 2*np.pi*(1.0 + self.ecc)**(-1.5)
        self.bin_sep_max = 1 + self.ecc
        self.bin_sep_min = 1 - self.ecc
        self.radius0 = float(radius0)
        self.radius1 = float(radius1)
        # simulation specifications
        self.boundary = float(boundary)
        self.label = str(label)
        self.verbose = bool(verbose)
        # initialize simulation
        self.N_test_start = 0  # test particles before collisions/escapes
        self.initial = None  # binary and test particle initial states (t=0)
        self.target_time = None
        self.cur_pos = None
            # above are updated when particles added/integration starts
        self.space_dim = 3
        self.m0 = self.mr
        self.m1 = 1 - self.mr
        self.sim = rb.Simulation()
        self.sim.exit_max_distance = self.boundary
        self.sim.heartbeat = self.heartbeat  # called every timestep
        self.sim.add(m=self.m0, r=self.radius0, hash=0)  # binary0
        self.sim.add(m=self.m1, r=self.radius1, hash=1,  # binary1
                     a=1.0, e=self.ecc)
        self.sim.N_active = 2  # rebound will ignore test-particle gravity
        self.sim.move_to_com()

    def add_test_particles(self, pos, vel):
        """
        Add an array of test particles (mass = 0) to the simulation. 
        Particle hash number will be assigned by their order in the 
        passed pos and vel arrays: the particle with hash n is added
        with position pos[n] and vel[n]. 

        Currently, particles can only be added at time t=0.

        Args:
        pos - ndarraylike shape (N, 3)
            The positions of each particle to add, as a sequence
            of Cartesian coordinates (x, y, z).
        vel - ndarraylike shape (N, 3)
            The velocities of each particle to add, as a sequence
            of Cartesian velocities (vx, vy, vz).
        """
        if self.sim.t > 0:
            raise RuntimeError("can only add test particles at time t=0; "
                               "simulation time is t={}".format(self.sim.t))
        starting_hash = 2 + self.N_test_start
            # binary stars use hash 0, 1; test particles use 2, 3, 4, ...
        for index, [(x, y, z), (vx, vy, vz)] in enumerate(zip(pos, vel)):
            self.sim.add(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
                         hash=starting_hash + index)
                # mass defaults to 0 (test particle); radius defaults to 0
        self.N_test_start = self.sim.N - 2
        self.initial = {'pos':self.bucket("coord", "all"),
                        'vel':self.bucket("coord", "all"),
                        'hash':self.bucket("hash", "all")}
        self.sim.serialize_particle_data(xyz=self.initial["pos"],
                                         vxvyvz=self.initial["vel"],
                                         hash=self.initial["hash"])

    def allocate_simulation_trackers(self):
        """ 
        Allocate dicts to track particle trajectories, collisions, and
        escapes.  After all particles have been added and the recording
        times set, the maximum number of records is known so arrays
        can be used for storage. The collision and escape arrays will
        likely contain empty slots after the simulation ends. The
        'empty' initializing value is np.nan for float and -1 for int.
        """
        self.colls = {"time":self.bucket("scalar", "all"), 
                      "bin_pos":self.bucket("coord", "all"), 
                      "bin_hash":self.bucket("hash", "all"),   
                      "test_pos":self.bucket("coord", "all"), 
                      "test_hash":self.bucket("hash", "all"),
                      "number":0}  
        self.colls["number"] = 0 
        self.escps = {"time":self.bucket("scalar", "all"),
                      "hash":self.bucket("hash", "all"), 
                      "pos":self.bucket("coord", "all"),
                      "vel":self.bucket("coord", "all"),
                      "number":0}
        self.escps["number"] = 0
        self.paths = {"pos":[],  "vel":[], "time":[]}

    def record(self):
        """ Save all particle states to memory; updates paths attribute """
        # get temporary containers for all remaining particles
        hashes = self.bucket("hash", "current")
        pos = self.bucket("coord", "current")
        vel = self.bucket("coord", "current")
        self.sim.serialize_particle_data(hash=hashes, xyz=pos, vxvyvz=vel)
        # fill data into containers for all particles, including coll/escp
        all_pos = self.bucket("coord", "all")
        all_pos[hashes] = pos
        all_vel = self.bucket("coord", "all")
        all_vel[hashes] = vel   
            # bucket fills np.nan --> escp/coll particles are recorded as nan     
        self.paths["pos"].append(all_pos)
        self.paths["vel"].append(all_vel)
        self.paths["time"].append(self.sim.t)

    def process_outputs(self):
        """ Reorganize outputs, remove empty data, etc """
        # clean escapes and collision trackers
        for container in [self.colls, self.escps]:
            for key in container:
                if key is "number": 
                    continue # number is only entry without possible empties
                container[key] = container[key][:container["number"], ...]
                    # strip off unused entries
        # process paths trackers
        for key in self.paths:
            self.paths[key] = np.asarray(self.paths[key])
            if key == "time":
                continue
            self.paths[key] = np.swapaxes(self.paths[key], 0, 1)
            self.paths[key] = np.swapaxes(self.paths[key], 1, 2)
                # old axes: (time, particle, coordinate)
                # new axes: (particle, coordinate, time)

    def run(self, target_time, record=True):
        """
        Integrate the simulation from the current time to the passed
        target_time. If record is set, the states of all particles will
        be saved in the paths attribute at each integration timestep.

        Sets:
        paths - dict of arrays
            paths["pos"][n] is a (3, T) array giving the Cartesian
            position as a function of time for the particle with hash
            n. n can be [0, 2 + N], with N the number of test particles
            in the simulation. T is the number of timesteps taken.
            "pos" -> "vel" gives analogously the velocities. NaNs show
            that the particle has been removed from the simulation. 
        """
        self.target_time = float(target_time)
        self.allocate_simulation_trackers()
        while self.sim.t < self.target_time:
            try:
                self.sim.integrate(self.target_time) 
            except rb.Escape:
                self.process_escape()
                if self.sim.N == 2:  # all test particles have been removed
                    break
        self.process_outputs()

    def get_coords(self, target):
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
        if ((self.cur_pos is None) or                 # first call
            (self.cur_pos.shape[0] != self.sim.N)):  # particles were removed
            self.cur_pos = self.bucket("coord", "current")
        self.sim.serialize_particle_data(xyz=self.cur_pos)

    def heartbeat(self, internal_sim_object):
        """ This function runs every simulation timestep """
        self.update_positions()
        self.check_for_collision()
        self.record()

    def process_escape(self):
        """ 
        Rebound has detected an escaped particle - remove it from simulation
        """
        test_coords = self.get_coords("test")
        dist = np.sqrt(np.sum(test_coords**2, axis=-1))
        outside = dist >= self.boundary
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
            self.escps["time"][self.escps["number"]] = self.sim.t
            self.escps["hash"][self.escps["number"]] = test_hash
            self.escps["pos"][self.escps["number"]] = pos
            self.escps["vel"][self.escps["number"]] = vel
            if self.verbose:
                print ("t={}: removing {} - escape"
                       "".format(self.sim.t, test_hash))
            self.sim.remove(hash=int(test_hash))
                # selecting from numpy int array does not return python int,
                # but rather numpy.int32, which fails rebound's type checks 
            self.escps["number"] += 1

    def check_for_collision(self):
        """
        Check for test particle + binary collisions, record and remove
        from the simulation.
        """
        test_coords = self.get_coords("test")
        binary_coords = self.get_coords("binary")
        # check all test particles against binary member 0
        dist0_sq = np.sum((test_coords - binary_coords[0])**2, axis=-1)
        colliding0 = dist0_sq < self.radius0**2
        # check all test particles against binary member 1
        bin_sep = np.sqrt(np.sum((binary_coords[0] -
                                  binary_coords[1])**2, axis=-1))
        min_dist0_sq = (bin_sep - self.radius1)**2
        max_dist0_sq = (bin_sep + self.radius1)**2
            # min and max distances from binary 0 which
            # can give a collision with binary 1
        to_check = (min_dist0_sq <= dist0_sq) & (dist0_sq <= max_dist0_sq)
        dist1_sq = np.sum((test_coords[to_check] - 
                           binary_coords[1])**2, axis=-1)
        colliding1 = np.zeros(colliding0.shape, dtype=bool)
        colliding1[to_check] = dist1_sq < self.radius1**2
        # process collisions
        if np.any(colliding0 | colliding1) > 0:
            test_hashes = self.get_active_test_hashes()
            for bin_hash, colliding in zip([0, 1], [colliding0, colliding1]):
                bin_pos = binary_coords[bin_hash]
                for index, test_hash in enumerate(test_hashes[colliding]):
                    # record collision
                    self.colls["time"][self.colls["number"]] = self.sim.t
                    self.colls["test_hash"][self.colls["number"]] = test_hash
                    test_pos = test_coords[colliding][index]
                    self.colls["test_pos"][self.colls["number"]] = test_pos
                    self.colls["bin_hash"][self.colls["number"]] = bin_hash
                    self.colls["bin_pos"][self.colls["number"]] = bin_pos
                    # remove colliding particle
                    if self.verbose:
                        print ("t={}: removing {} - collision with binary {}"
                               "".format(self.sim.t, test_hash, bin_hash))
                    self.sim.remove(hash=int(test_hash)) # see process_escape
                    self.colls["number"] += 1

    def get_active_test_hashes(self):
        """ """
        hashes = np.zeros(self.sim.N, dtype="uint32")
        self.sim.serialize_particle_data(hash=hashes)
        test_hashes = hashes[2:]
        return test_hashes

    def save_sim(self, filebase=None):
        """
        Output simulation data to pickle and plot current trajectories.
        The passed filename will be used for both pickle and plots,
        and defaults to the simulation's label attribute.
        """
        if filebase is None:
            filebase = self.label
        pickle_filename = "{}.p".format(filebase)
        sim_info = {"mr":self.mr,
                    "ecc":self.ecc,
                    "period":self.period,
                    "bin_sep_max":self.bin_sep_max,
                    "bin_sep_min":self.bin_sep_min,
                    "radius0":self.radius0,
                    "radius1":self.radius1,
                    "boundary":self.boundary,
                    "label":self.label,
                    "colls":self.colls,
                    "escps":self.escps,
                    "paths":self.paths,
                    "cur_pos":self.cur_pos,
                    "target_time":self.target_time,
                    "cur_time":self.sim.t,
                    "initial":self.initial}
        utl.save_pickle(sim_info, pickle_filename)
        fig, ax = gb.plot_sim_verbose(self)
        initial = np.absolute(self.paths["pos"][2:, :, 0]).max()
        boxes = [initial*1.03, 2.5, self.boundary*1.03] # magic
        names = ['starting', 'central', 'boundary']
        for box, name in zip(boxes, names):
            ax.set_xlim(-box, box) 
            ax.set_ylim(-box, box)
            fig.savefig("{}-{}.png".format(filebase, name))
        plt.close("all")

    def bucket(self, desc, num, shape=None, dtype=None, fill=None):
        """
        Generate a blank numpy array to hold data of the passed
        description for a number num of particles. num can be integer,
        or a string flag: 'all' makes enough space for every particle
        in the simulation, including those which have been removed by
        collision or escape, and 'current' stores only particles
        currently being integrated.  The size, dtype, and fill values
        are be determined from desc and num unless explicitly passed.
        Default particle numbers include the two binary members.
        """
        # get number of particles
        if num == 'all':
            N = self.N_test_start + 2
        elif num == 'current':
            N = self.sim.N 
        else:
            N = int(num)
        # make array
        if desc == "coord":
            if shape is None:
                shape = (N, self.space_dim)
            if fill is None:
                fill = np.nan
            if dtype is None:
                dtype = "float64"
        elif desc == "hash":
            if shape is None:
                shape = (N,)
            if fill is None:
                fill = 0
            if dtype is None:
                dtype = "uint32"
        elif desc == "scalar":
            if shape is None:
                shape = (N,)
            if fill is None:
                fill = np.nan
            if dtype is None:
                dtype = "float64"
        else:
            raise ValueError("Unrecognized desc {}".format(desc))
        return np.full(shape, fill, dtype=dtype)


def reboundparticle_to_array(p):
    """
    Convert a Rebound Particle object to an array containing its
    Cartesian position and velocity.  Accepts a Rebound Particle 
    and returns two (3,) ndarrays: pos, vel 
    """
    pos = np.array([p.x, p.y, p.z])
    vel = np.array([p.vx, p.vy, p.vz])
    return pos, vel

