""" 
Test basic functionality gravbin rebound wrapper - adding particles,
running simulation, detecting and recording collision and escape 
events, and plotting. 

Particles are randomly added on two spheres concentric with each binary
member, in sets moving with the same velocity.
"""


import warnings
warnings.simplefilter("always", RuntimeWarning)
import time

import numpy as np
import matplotlib.pyplot as plt

import gravbin as gb


import cProfile
pr = cProfile.Profile()

bin_radius = 0.1
test_start_distance = bin_radius*2.5
test_vels = [0.7, 1.8]
num_tests = 10000  # per direction, per binary
mass_ratio = 0.5
ecc = 0.0
orbits = 10

test_sim = gb.BinarySim(mass_ratio=mass_ratio, radius0=bin_radius,
                        radius1=bin_radius, eccentricity=ecc, label='testsim')
final_time = test_sim.period*orbits
bin0_start = np.array([(mass_ratio - 1)*test_sim.bin_sep_min, 0, 0])
bin1_start = np.array([mass_ratio*test_sim.bin_sep_min, 0, 0])
for binary_start in [bin0_start, bin1_start]:
    for start_vel in test_vels:
        random_vects = np.random.random((num_tests, 3)) - 0.5
        random_dirs = (random_vects.T/np.sum(random_vects**2, axis=1)).T
        pos = binary_start + test_start_distance*random_dirs
        vel = start_vel*random_dirs
        test_sim.add_test_particles(pos, vel)
pr.enable()
test_sim.run(final_time, page=8*2**30, record=False)
pr.disable()
# test_sim.save_sim()
pr.print_stats(sort="time")