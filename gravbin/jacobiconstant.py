"""
Jacobi constant calculations
"""

import functools

import numpy as np 
import scipy.optimize as opt
import matplotlib.pyplot as plt


def find_local_min(func, bounds, tol=10**(-6)):
    out = opt.minimize_scalar(func, bounds=bounds, method="bounded")
    min_location = out['x']
    if np.isclose(min_location, bounds, rtol=10**(-4)).any():
        return None
    return [out['x'], out['fun']]


class JacobiBarrier(object):
    """
    Computes the zero-velocity surface associated with a given Jacobi constant.
    """
    def __init__(self, mass_ratio=0.5):
        self.mr = float(mass_ratio)
        if (self.mr < 0.5) or (self.mr > 1):
            raise ValueError("Invalid mass ratio: {}".format(self.mr)) 

    def eff_potential(self, r=None, theta=None, phi=None):
        """ The effective potential in the co-rotating frame """
        chi = np.sin(theta)*np.cos(phi)  # encodes angular position
        delta0 = np.sqrt((r - (self.mr - 1.0)*chi)**2 + 
                         (1.0 - chi**2)*(self.mr - 1.0)**2)
        delta1 = np.sqrt((r - self.mr*chi)**2 + (1.0 - chi**2)*self.mr**2)
            # distance to the first (most massive) and second binary members
        return -r**2 - self.mr/delta0 - (1.0 - self.mr)/delta1

    def find_extrema(self, theta, phi):
        chi = np.sin(theta)*np.cos(phi)  # encodes angular position
        extrema = []
        # find peaks
        minus_v_eff = lambda r: -self.eff_potential(r, theta, phi) # max -> min
        dips = np.asarray([(self.mr - 1)*chi, self.mr*chi]) # planet locations
        dips.sort()
        intervals = [[-1.5, dips[0]], [dips[0], dips[1]], [dips[1], 1.5]]
            # at most three peaks, one to be found between the planets and
            # one outside each planet, with outer peaks always closer than 1.5
        for interval in intervals:
            out = find_local_min(minus_v_eff, interval) # None if no min found
            if out is not None:
                pos, value = out # [radius, min function value]
                extrema.append([pos, -value]) # min -> max
        # find valleys
        v_eff = lambda r: self.eff_potential(r, theta, phi) 
        peaks = np.array(extrema)[:, 0] # peak locations
        intervals = zip(peaks[:-1], peaks[1:]) # valleys are between peaks
        for interval in intervals:
            out = find_local_min(v_eff, interval)
            if out is not None:
                extrema.append(out)
        extrema = np.asarray(extrema)
        extrema = extrema[np.argsort(extrema[:, 0])] # sort by radial position
        return extrema

    def test_extrema(self, theta, phi):
        fig, ax = plt.subplots()
        r = np.linspace(-1.5, 1.5, 1000)
        extrema = self.find_extrema(theta, phi)
        ax.plot(r, self.eff_potential(r, theta, phi), alpha=0.6,
                linestyle='-', marker='', color='b')
        ax.plot(*extrema.T, marker='o', linestyle='', color='k')
        plt.show()
        return

    def find_barriers(self, jacobi, theta, phi):
        root_func = lambda r: jacobi - self.eff_potential(r, theta, phi) 
        intervals = [] # bracket each root
        extrema = self.find_extrema(theta, phi)
        extrema = extrema[np.argsort(extrema[:, 0])] # sort by radial position
        # find edge intervals
        edge_peaks = extrema[[0, -1]]
        for loc, height in edge_peaks:
            if height < jacobi:
                continue
            guess = np.sqrt(-jacobi) # kosher because height is negative
            interval = np.sort([loc, np.sign(loc)*guess])
            intervals.append(interval)
        # find central intervals
        possible_intervals = zip(extrema[:-1, 0], extrema[1:, 0])
        for left, right in possible_intervals:
            if np.sign(root_func(left)) != np.sign(root_func(right)):
                intervals.append([left, right]) # sign change => root
        roots = []
        for interval in intervals:
            root = opt.brentq(root_func, *interval)
            roots.append(root)
        return np.asarray(roots)

    def test_barriers(self, jacobi, theta, phi, res=10**6):
        fig, ax = plt.subplots()
        roots = self.find_barriers(jacobi, theta, phi)
        extrema = self.find_extrema(theta, phi)
        limit = np.absolute(list(roots) + list(extrema[:, 0])).max()
        r = np.arange(-limit*1.05, limit*1.05, 2*limit/res)
        ax.plot(r, self.eff_potential(r, theta, phi), alpha=0.6,
                linestyle='-', marker='', color='b')
        ax.axhline(jacobi, color='r')
        ax.plot(roots, self.eff_potential(roots, theta, phi),
                marker='o', linestyle='', color='k')
        plt.show()
        return
