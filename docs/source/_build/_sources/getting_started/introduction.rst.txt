Introduction
============

Brief Theory
------------

Digital Particle Image Velocimetry (DPIV) has become a well-established non-intrusive measuring technique to study fluid flows. The standard DPIV setup allows for instantaneous two-dimensional (2D), two-component (2C) velocity fields consisting of four main components: a camera, seeding particles, analysis software, and a laser with optics. To obtain the flow field, the flow of interest must be seeded with particles. The particles should be chosen carefully so that their motion faithfully follows that of the flow. The particles are made visible by illuminating them with a light source combined with appropriate optics. The illuminated particles can be filmed and analyzed with a synchronizer or DPIV software. To estimate the flow field of the captured images, they are divided into smaller areas, called interrogation windows, where the flow is expected to be mostly uniform. By cross-correlating the interrogation windows of two images that are obtained within a short time interval Δ𝒕, the displacement can be obtained in terms of pixel/Δ𝒕. The displacements, in terms of pixel/Δ𝒕, can be transformed into velocity in meters/second by using the time between the two images and the relation between the camera sensor and the measurement plane.

OpenPIV-Python-cxx
------------------
Using OpenPIV and its associated packages, users can easily perform digital analysis of DPIV image pairs. The core of these packages are OpenPIV-Python, an open source general use DPIV package written in Python 3. Due to memory constraints caused by performing 3D ffts on a stack of 2D interrogation windows on low- to mid-end consumer-grade laptops, a branch of OpenPIV-Python was made with a c++ backend. This greatly lowers memory consuption during DPIV processing and is generally faster.