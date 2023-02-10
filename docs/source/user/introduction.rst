============
Introduction
============

Brief theory
============

Digital Particle Image Velocimetry (DPIV) has become a well-established non-intrusive measuring technique to study fluid flows. The standard DPIV setup allows for instantaneous two-dimensional (2D), two-component (2C) velocity fields consisting of four main components: a camera, seeding particles, analysis software, and a laser with optics. To obtain the flow field, the flow of interest must be seeded with particles. The particles should be chosen carefully so that their motion faithfully follows that of the flow. The particles are made visible by illuminating them with a light source combined with appropriate optics. The illuminated particles can be filmed and analyzed with a synchronizer or DPIV software. To estimate the flow field of the captured images, they are divided into smaller areas, called interrogation windows, where the flow is expected to be mostly uniform. By cross-correlating the interrogation windows of two images that are obtained within a short time interval Δ𝒕, the displacement can be obtained in terms of pixel/Δ𝒕. The displacements, in terms of pixel/Δ𝒕, can be transformed into velocity in meters/second by using the time between the two images and the relation between the camera sensor and the measurement plane.

OpenPIV-Python-cxx
==================
The OpenPIV-Python-cxx library, or OpenPIV-cxx, is a Python library for analyzing DPIV images. The package is designed to use 2-dimensional (soon 3-dimensional) NumPy arrays, making it fast and flexible for a wide range of use-cases. Additionally, this package provides the following:
 - A low memory usage framework
 - Generaly faster than its Python counterpart
 - Better interpolation routines
 - An additional API for backwards compatability

Additionally, a FFTW3 backend is in the works to allow DPIV processing at the speed of commercial softwares.
