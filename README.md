Bingham Statistics Library (BSL)
================================

The Bingham statistics library contains implementations of the Bingham distribution for directional (axial) statistics on the unit spheres S1, S2, and S3. In addition, finite element approximations are available via tessellations of S2 and S3.

This repository was exported from the original and **now to be abandoned** [SVN repository](https://code.google.com/p/bingham/) on google code with consent of the original author Jared Glover.

News
====

### [2013-12-14] Created libbingham Tutorials ###
  * See the Tutorials section below for Matlab and C tutorials.

### [2013-05-20] Released libbingham v0.3 ###
  * Added matlab functions bingham\_fit() and bingham\_fit\_scatter().
  * Added several utility functions.

### [2012-04-18] Released libbingham v0.2.1 ###
  * Fixed a bug in bingham\_mult() caused by eigen\_symm() returning eigenvalues in the wrong order (compared to the GSL version).
  * Also added more matlab functions.

### [2012-04-12] Released libbingham v0.2 ###
  * Enlarged Bingham lookup tables to handle concentration params up to -900.
  * More MATLAB support.
  * Removed dependency on GSL.
  * Switched from GPL to BSD license.

Documentation
=============

See the [wiki](https://github.com/SebastianRiedel/bingham/wiki) for more information.

Quick Usage / Installation
==========================

The preferred method for download is checking out the latest code in this repository.

The library contains both C and Matlab code, but only the C code is currently supported. (The Matlab functions are mostly used for plotting figures, and do not contain a complete implementation of the Bingham distribution. Use with caution!)

In the folder bingham/c, run:

```
make
make install
```

This will install the C library, libbingham.a, on your machine. To use the library in your own project, simply add "-lbingham" to your linker flags and include the appropriate headers as you would with any other C library.

Documentation is still sparse, but there are many examples in the bingham/c directory in test_bingham.c, test_util.c, fit_bingham.c, bingham_lookup.c, bingham_sample.c, and cluster_bingham.c.
