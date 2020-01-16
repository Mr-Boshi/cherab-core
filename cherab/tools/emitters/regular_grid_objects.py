# -*- coding: utf-8 -*-
#
# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

"""
Ray transfer objects accelerate the calculation of geometry matrices (or Ray Transfer Matrices as
they were called in `S. Kajita, et al. Contrib. Plasma Phys., 2016, 1-9
<https://onlinelibrary.wiley.com/doi/abs/10.1002/ctpp.201500124>`_)
in the case of regular spatial grids. As in the case of Voxels, the spectral array is used to store
the data for individual light sources (in this case the grid cells or their unions), however
no voxels are created at all. Instead, a custom integration along the ray is implemented.
Ray transfer objects allow to calculate geometry matrices for a single value of wavelength.

Use `RayTransferBox` class for Cartesian grids and `RayTransferCylinder` class for cylindrical grids
(3D or axisymmetrical).

Performance tips:

The best performance is achieved when Ray Transfer Objects are used with special pipelines and
optimised materials (currently only rough metals are optimised, see the demos).

When the number of individual light sources and respective bins in the spectral array is higher
than ~50-70 thousands, the lack of CPU cache memory becomes a serious factor affecting performance.
Therefore, it is not recommended to use hyper-threading when calculating geometry matrices for
a large number of light sources. It is also recommended to divide the calculation into several
parts and to calculate partial geometry matrices for not more than ~50-70 thousands of light
sources in a single run. Partial geometry matrices can easily be combined into one when all
computations are complete.
"""

from raysect.primitive import Cylinder, Subtract, Box
from raysect.optical import Point3D
from .regular_grid_emitters import CylindricalRegularIntegrator, CartesianRegularIntegrator
from .regular_grid_emitters import CylindricalRegularEmitter, CartesianRegularEmitter


class RegularGridObject:
    """
    Basic class for regular grid objects.

    :ivar np.ndarray emission: Spectral emission (in :math:`W/(sr m^3 nm)`) defined
        on a regular 3D grid.
    :ivar np.ndarray spectral_index: The 1D array, which maps the spectral emission
        array to the respective spectral bins.
    :ivar int min_wavelength: The minimal wavelength equal to `camera.min_wavelength`.
    :ivar Node parent: Scene-graph parent node.
    :ivar AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system
        relative to the scene-graph parent.
    :ivar RegularGridEmitter material: Regular grid emitter.
    :ivar float step: Integration step of volume integrator.
    """

    def __init__(self, primitive):
        self._primitive = primitive

    @property
    def parent(self):
        return self._primitive.parent

    @parent.setter
    def parent(self, value):
        self._primitive.parent = value

    @property
    def transform(self):
        return self._primitive.transform

    @transform.setter
    def transform(self, value):
        self._primitive.transform = value

    @property
    def step(self):
        return self._primitive.material.integrator.step

    @step.setter
    def step(self, value):
        self._primitive.material.integrator.step = value

    @property
    def material(self):
        return self._primitive.material

    @property
    def min_wavelength(self):
        return self._primitive.material.min_wavelength

    @property
    def spectral_index(self):
        return self._primitive.material.spectral_index

    @property
    def emission(self):
        return self._primitive.material.emission


class RegularGridCylinder(RegularGridObject):
    """
    Regular Grid Object for cylindrical emitter defined on a regular 3D :math:`(R, \phi, Z)` grid.
    The emitter is periodic in :math:`\phi` direction.
    The base of the cylinder is located at `Z = 0` plane. Use `transform`
    parameter to move it.

    :param np.ndarray emission: The 4D array containing the spectral emission
        (in :math:`W/(sr m^3 nm)`) defined on a regular 3D grid in cylindrical coordinates:
        :math:`(R, \phi, Z)` (in axisymmetric case `emission.shape[1] == 1`).
        The last dimension of this array is the spectral one.
        The spectral resolution of the emission profile must be equal to
        `(camera.max_wavelength - camera.min_wavelength) / camera.spectral_bins`.
        Some of the spectral bins can be skipped (e.g. if the material does not emit on
        certain wavelengths of the specified wavelength range) with the help of the
        `spectral_index` parameter  This allows to save memory. which is
        especially useful in a non-axisymmetric case.
    :param np.ndarray spectral_index: The 1D array with the size equal to
        `emission.shape[3]`, which maps the spectral emission array to the respective
        spectral bins.
    :param double min_wavelength: The minimal wavelength which must be equal to
        `camera.min_wavelength`. This parameter is required to correctly process
        dispersive rendering.
    :param float radius_outer: Radius of the outer cylinder and the upper bound of grid in
        `R` direction (in meters).
    :param float height: Height of the cylinder and the length of grid in `Z` direction
        (in meters).
    :param float radius_inner: Radius of the inner cylinder and the lower bound of grid in
        `R` direction (in meters), defaults to `radius_inner=0`.
    :param float period: A period in :math:`\phi` direction (in degree), defaults to `period=360`.
    :param float step: The step of integration along the ray (in meters), defaults to
        `0.25*min((radius_outer - radius_inner) / emission.shape[0], height / emission.shape[2])`.
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system
        relative to the scene-graph parent (default = identity matrix).

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import World, translate
        >>> from raysect.optical.observer import SpectralRadiancePipeline2D
        >>> from raysect.primitive import Cylinder, Subtract
        >>> from cherab.tools.emitters import RegularGridCylinder
        >>> # Assume that the files 'Be_4574A.npy' and 'Be_527A.npy' contain the emissions
        >>> # (in W / m^3) of Be I (3d1 1D2 -> 2p1 1P1) and Be II (4s1 2S0.5 -> 3p1 2P2.5)
        >>> # defined on a regular cylindrical grid: 3.5 m < R < 9 m,
        >>> # 0 < phi < 20 deg, -5 m < Z < 5 m, and periodic in phi direction.
        >>> emission_4574 = np.load('Be_4574A.npy')
        >>> emission_5272 = np.load('Be_4574A.npy')
        >>> # Grid properties
        >>> rmin = 3.5
        >>> rmax = 9.
        >>> phi_period = 20.
        >>> zmin = -5.
        >>> zmax = 5.
        >>> grid_shape = emission_4574.shape
        >>> # Defining wavelength step and converting to W/(m^3 sr nm)
        >>> delta_wavelength = 5.  # 5 nm wavelength step
        >>> emission = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], 2))
        >>> emission[:, :, :, 0] = emission_4574 / (4. * np.pi * delta_wavelength)  # W/(m^3 sr nm)
        >>> emission[:, :, :, 1] = emission_5272 / (4. * np.pi * delta_wavelength)
        >>> # Defining wavelength range and creating spectral_index array
        >>> min_wavelength = 457.4 - 0.5 * delta_wavelength
        >>> spectral_index = np.zeros(2, dtype=np.int32)
        >>> spectral_index[1] = int((527.2 - min_wavelength) / delta_wavelength)
        >>> spectral_bins = spectral_index[1] + 1
        >>> max_wavelength = min_wavelength + spectral_bins * delta_wavelength
        >>> # Creating the scene
        >>> world = World()
        >>> pipeline = SpectralRadiancePipeline2D()
        >>> rgc = RegularGridCylinder(emission, spectral_index, min_wavelength,
                                      radius_outer=rmax, height=zmax - zmin,
                                      radius_inner=rmin, period=phi_period,
                                      parent=world, transform=translate(0, 0, zmin))
        ...
        >>> camera.spectral_bins = spectral_bins
        >>> camera.min_wavelength = min_wavelength
        >>> camera.max_wavelength = max_wavelength
        ...
        >>> # If reflections do not change the wavelength, the results for each spectral line
        >>> # can be obtained in W/(m^2 sr) in the following way.
        >>> radiance_4574 = pipeline.frame.mean[:, :, spectral_index[0]] * delta_wavelength
        >>> radiance_5272 = pipeline.frame.mean[:, :, spectral_index[1]] * delta_wavelength
    """

    def __init__(self, emission, spectral_index, min_wavelength, radius_outer, height, radius_inner=0, period=360., step=None,
                 parent=None, transform=None):
        if 360. % period > 1.e-3:
            raise ValueError("The period %.3f is not a multiple of 360." % period)
        if emission.ndim != 4:
            raise ValueError("Argument 'emission' must be a 4D array.")
        dr = (radius_outer - radius_inner) / emission.shape[0]
        dphi = period / emission.shape[1]
        dz = height / emission.shape[2]
        grid_steps = (dr, dphi, dz)
        eps_r = 1.e-5 * dr
        eps_z = 1.e-5 * dz
        step = step or 0.25 * min(dr, dz)
        material = CylindricalRegularEmitter(emission, spectral_index, grid_steps, min_wavelength,
                                             integrator=CylindricalRegularIntegrator(step), rmin=radius_inner)
        primitive = Subtract(Cylinder(radius_outer - eps_r, height - eps_z), Cylinder(radius_inner + eps_r, height - eps_z),
                             material=material, parent=parent, transform=transform)
        super().__init__(primitive)


class RegularGridBox(RegularGridObject):
    """
    Regular Grid Object for rectangular emitter defined on a regular 3D :math:`(X, Y, Z)` grid.
    The grid starts at (0, 0, 0). Use `transform` parameter to move it.

    :param np.ndarray emission: The 4D array containing the spectral emission
        (in :math:`W/(sr m^3 nm)`) defined on a regular 3D grid in Cartesian coordinates.
        The last dimension of this array is the spectral one.
        The spectral resolution of the emission profile must be equal to
        `(camera.max_wavelength - camera.min_wavelength) / camera.spectral_bins`.
        Some of the spectral bins can be skipped (e.g. if the material does not emit on
        certain wavelengths of the specified wavelength range) with the help of the
        `spectral_index` parameter  This allows to save memory.
    :param np.ndarray spectral_index: The 1D array with the size equal to
        `emission.shape[3]`, which maps the spectral emission array to the respective
        spectral bins.
    :param float xmax: Upper bound of grid in `X` direction (in meters).
    :param float ymax: Upper bound of grid in `Y` direction (in meters).
    :param float zmax: Upper bound of grid in `Z` direction (in meters).
    :param float step: The step of integration along the ray (in meters), defaults to
        `step = 0.25 * min(xmax / emission.shape[0],
                           ymax / emission.shape[1],
                           zmax / emission.shape[2])`.
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system
        relative to the scene-graph parent (default = identity matrix).

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import World, translate, Point3D
        >>> from raysect.primitive import Box
        >>> from raysect.optical.observer import SpectralRadiancePipeline2D
        >>> from cherab.tools.emitters import RegularGridBox
        >>> # Assume that the files 'Be_4574A.npy' and 'Be_527A.npy' contain the emissions
        >>> # (in W / m^3) of Be I (3d1 1D2 -> 2p1 1P1) and Be II (4s1 2S0.5 -> 3p1 2P2.5)
        >>> # defined on a regular Cartesian grid: -3 m < X < 3 m,
        >>> # -3 m < Y < 3 m and -6 m < Z < 6 m.
        >>> emission_4574 = np.load('Be_4574A.npy')
        >>> emission_5272 = np.load('Be_4574A.npy')
        >>> # Grid properties
        >>> xmin = ymin = -3.
        >>> xmax = ymax = 3.
        >>> zmin = -6.
        >>> zmax = 6.
        >>> grid_shape = emission_4574.shape
        >>> # Defining wavelength step and converting to W/(m^3 sr nm)
        >>> delta_wavelength = 5.  # 5 nm wavelength step
        >>> emission = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], 2))
        >>> emission[:, :, :, 0] = emission_4574 / (4. * np.pi * delta_wavelength)  # W/(m^3 sr nm)
        >>> emission[:, :, :, 1] = emission_5272 / (4. * np.pi * delta_wavelength)
        >>> # Defining wavelength range and creating spectral_index array
        >>> min_wavelength = 457.4 - 0.5 * delta_wavelength
        >>> spectral_index = np.zeros(2, dtype=np.int32)
        >>> spectral_index[1] = int((527.2 - min_wavelength) / delta_wavelength)
        >>> spectral_bins = spectral_index[1] + 1
        >>> max_wavelength = min_wavelength + spectral_bins * delta_wavelength
        >>> # Creating the scene
        >>> world = World()
        >>> pipeline = SpectralRadiancePipeline2D()
        >>> box = RegularGridBox(emission, spectral_index, min_wavelength,
                                 xmax=xmax - xmin, ymax=ymax - ymin, zmax=zmax - zmin,
                                 parent=world, transform=translate(xmin, ymin, zmin))
        ...
        >>> camera.spectral_bins = spectral_bins
        >>> camera.min_wavelength = min_wavelength
        >>> camera.max_wavelength = max_wavelength
        ...
        >>> # If reflections do not change the wavelength, the results for each spectral line
        >>> # can be obtained in W/(m^2 sr) in the following way.
        >>> radiance_4574 = pipeline.frame.mean[:, :, spectral_index[0]] * delta_wavelength
        >>> radiance_5272 = pipeline.frame.mean[:, :, spectral_index[1]] * delta_wavelength
    """

    def __init__(self, emission, spectral_index, min_wavelength, xmax, ymax, zmax,
                 step=None, parent=None, transform=None):
        if emission.ndim != 4:
            raise ValueError("Argument 'emission' must be a 4D array.")
        dx = xmax / emission.shape[0]
        dy = ymax / emission.shape[1]
        dz = zmax / emission.shape[2]
        grid_steps = (dx, dy, dz)
        eps_x = 1.e-5 * dx
        eps_y = 1.e-5 * dy
        eps_z = 1.e-5 * dz
        step = step or 0.25 * min(dx, dy, dz)
        material = CartesianRegularEmitter(emission, spectral_index, grid_steps, min_wavelength,
                                           integrator=CartesianRegularIntegrator(step))
        primitive = Box(lower=Point3D(0, 0, 0), upper=Point3D(xmax - eps_x, ymax - eps_y, zmax - eps_z),
                        material=material, parent=parent, transform=transform)
        super().__init__(primitive)
