# gravbin
Python module for calculating 3-body orbits around a Keplerian binary.

The objective here is understanding the capture and evolution of projectiles by a binary. To do this, the module provides solvers for the orbits of the projectile in a restricted case: 
- The projectile mass is assumed negligible, so that the binary executes a Keplerian orbit. 
- Binary orbit is assumed to be circular. 

The projectile is not necessarily assumed to orbit in the plane of the binary. In addition to the orbit solvers, code is included to produce and analyze suites of orbits to address several questions of interest: 
- What is the capture cross section for incoming light projectiles?
- What is the timescale for escape or collision with a binary member for a captured projectile? What fraction of captures result in an escape vs collision?

For theoretical discussion of the results used in the code, see theory.md
