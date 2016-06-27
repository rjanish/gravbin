""" 
Test basic functionality gravbin rebound wrapper - adding particles,
running simulation, detecting and recording collision and escape 
events, and plotting. 

Particles are randomly added on two spheres concentric with each binary
member, each with a velocity radial from its respective binary. Half 
of these particles start moving inward (test collision) and the other
half outward (test escape).
"""


import warnings
warnings.simplefilter("always", RuntimeWarning)

import numpy as np
import matplotlib.pyplot as plt

import gravbin as gb


bin_radius = 0.1
test_start_distance = 0.15
test_start_speed_out = 1.0
test_start_speed_in = 2.5
num_tests = 20 # per direction, per binary
mass_ratio = 0.6
ecc = 0.2
test_per_binary = int(num_tests*2)

T = 2*np.pi
orbits = 10
samples_per_orbit = 500
times = np.linspace(0, orbits*T, orbits*samples_per_orbit)

test_sim = gb.BinarySim(mass_ratio=mass_ratio, radius0=bin_radius, 
                        radius1=bin_radius, eccentricity=ecc, label='testsim')
bin0_start = np.array([(mass_ratio - 1)*test_sim.bin_sep_min, 0, 0])
bin1_start = np.array([mass_ratio*test_sim.bin_sep_min, 0, 0])
for binary_start in [bin0_start, bin1_start]:
    for start_vel in [test_start_speed_in, test_start_speed_out]:
        random_vects = np.random.random((num_tests, 3)) - 0.5
        random_dirs = (random_vects.T/np.sum(random_vects**2, axis=1)).T
        pos = binary_start + test_start_distance*random_dirs
        vel = start_vel*random_dirs
        test_sim.add_test_particles(pos, vel)
test_sim.run(times)
test_sim.save_sim()