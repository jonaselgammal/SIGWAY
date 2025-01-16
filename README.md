# SIGWAY

### A package for computing second-order, scalar induced gravitational wave signals. 

SIGWAY is a collaborative effort of the LISA Cosmology Working Group in an effort to build the foundation of a data analysis pipeline for stochastic gravitational wave signals emitted from inflation. Currently the package contains modules for

- solving the Mukhanov-Sasaki equation for single field ultra-slow roll inflationary models and computing the primordial scalar power spectrum $\mathcal{P}_\zeta$
- computing the second order gravitational wave power spectrum $\Omega_{\mathrm{GW}}$ from $\mathcal{P}_\zeta$ for reentry during radiation domination or a phase of early matter domination.

This code is still under heavy development. There is no stable release yet but we are working on it. The documentation is also still under construction. However there are docstrings in the modules.

### Using this code
If you use this code please cite our paper (ArXiv link will be added shortly) and feel free to drop me an email if you encounter any problems. Also, if there are bugs please report them!

### Installation
Currently, to install this code, you will have to clone this repository. To do that you want to clone either with SSH or HTTPS by navigating to the folder you want the package to live in and running

```
$ git clone git@github.com:jonaselgammal/SIGWAY.git
```

or

```
$ git clone https://github.com/jonaselgammal/SIGWAY.git
```

Then, cd to the folder that's created and run

```
$ pip install -e .
```

The -e is to install in "editable" mode, meaning that if you change something in the code you don't have to re-install with pip. 

### Dependencies

The current, minimal public version that contains the core functionality of the package needs jax, diffrax, numpy, scipy and matplotlib. The dependencies should be installed automatically but if not pip is your friend.