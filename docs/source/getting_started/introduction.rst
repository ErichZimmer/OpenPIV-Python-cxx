Introduction
============

Brief Theory
------------

Digital Particle Image Velocimetry (DPIV) has become a well-established non-intrusive measuring technique to study fluid flows. The standard DPIV setup allows for instantaneous two-dimensional (2D), two-component (2C) velocity fields consisting of four main components: a camera, seeding particles, analysis software, and a laser with optics. To obtain the flow field, the flow of interest must be seeded with particles. The particles should be chosen carefully so that their motion faithfully follows that of the flow. The particles are made visible by illuminating them with a light source combined with appropriate optics. The illuminated particles can be filmed and analyzed with a synchronizer or DPIV software. To estimate the flow field of the captured images, they are divided into smaller areas, called interrogation windows, where the flow is expected to be mostly uniform. By cross-correlating the interrogation windows of two images that are obtained within a short time interval Δ𝒕, the displacement can be obtained in terms of pixel/Δ𝒕. The displacements, in terms of pixel/Δ𝒕, can be transformed into velocity in meters/second by using the time between the two images and the relation between the camera sensor and the measurement plane.

OpenPIV-Python-cxx
------------------
OpenPIV-Python-cxx is a modified version of OpenPIV-Python that uses less memory while retaining a similar performance. Most computationally intensive functions that are considered "slow" and uses Numpy, SciPy, or Scikit-Image were implemented in c++ and wrapped with pybind11. This enables the processing of large images on hardware with limited RAM and disk space (e.g. some consumer-grade laptops). To keep compatability with OpenPIV-Python, the submodule openpiv was created with an identicle API to the original package (still a work in progress).

