""" 
Test running time and memory usage. 
"""


import warnings
warnings.simplefilter("always", RuntimeWarning)
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

import gravbin as gb


def timed_radial_test(test_num, orbits, samples_per_orbit, label,
                      mass_ratio=0.5, ecc=0, bin_radius=0.1,
                      start_frac=1.01, start_vels=[1.1]):
    test_sim = gb.BinarySim(mass_ratio=mass_ratio, radius0=bin_radius, 
                            radius1=bin_radius, eccentricity=ecc, label=label)
    bin0_start = np.array([(mass_ratio - 1)*test_sim.bin_sep_min, 0, 0])
    bin1_start = np.array([mass_ratio*test_sim.bin_sep_min, 0, 0])
    for binary_start in [bin0_start, bin1_start]:
        for start_vel in start_vels:
            random_vects = np.random.random((num_tests, 3)) - 0.5
            random_dirs = (random_vects.T/np.sum(random_vects**2, axis=1)).T
            pos = binary_start + bin_radius*start_frac*random_dirs
            vel = start_vel*random_dirs
            test_sim.add_test_particles(pos, vel)
    times = np.linspace(0, orbits*test_sim.period, orbits*samples_per_orbit)
    t0 = time.time()
    test_sim.run(times)
    t1 = time.time()
    timing = t1 - t0
    test_sim.save_sim()
    return timing


num_trials = 10
all_num_tests = [3, 5, 10, 25, 50, 100, 250, 500] # per direction, per binary
mass_ratio = 0.6
ecc = 0.1
orbits = 10
samples_per_orbit = 100

timing = np.zeros((len(all_num_tests)*num_trials, 2))
for num_index, num_tests in enumerate(all_num_tests):
    for trial in range(num_trials):
        index = num_index*num_trials + trial
        print "testing N = {}, trial {}".format(num_tests, trial + 1)
        name = "timing-{}-{}".format(num_tests, trial)
        timing[index, 0] = num_tests
        timing[index, 1] = timed_radial_test(num_tests, orbits, samples_per_orbit,
                                             name, mass_ratio=mass_ratio, ecc=ecc)
        print "ran in {} seconds".format(timing[index, 1])
    np.savetxt("timetest.dat", timing)
    fig, ax = plt.subplots()
    ax.plot(*timing.T, marker='.', linestyle='', color='k')
    ax.set_xlim(timing[:, 0].min() - 1, timing[:, 0].max() + 1)
    fig.savefig("timetest.png")