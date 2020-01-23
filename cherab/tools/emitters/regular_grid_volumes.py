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
Regular Grid Volumes accelerate integration through inhomogeneous emitting volume
as they use pre-calculated values of spectral emissivity on a regular grid.

Use `RegularGridBox` class for Cartesian grids and `RegularGridCylinder` class for cylindrical grids
(3D or axisymmetrical).
"""

from raysect.primitive import Cylinder, Subtract, Box
from raysect.optical import Point3D
from .regular_grid_emitters import CylindricalRegularIntegrator, CartesianRegularIntegrator
from .regular_grid_emitters import CylindricalRegularEmitter, CartesianRegularEmitter


class RegularGridVolume:
    """
    Basic class for regular grid volumes.

    :ivar np.ndarray ~.emission: 2D array of spectral emission (in :math:`W/(sr\,m^3\,nm)`)
        defined on the cells of a regular 3D grid.
    :ivar np.ndarray spectral_map: The 1D array, which maps the spectral emission
        array to the respective spectral bins of spectral array specified in the camera
        settings.
    :ivar np.ndarray voxel_map: The 3D array containing for each grid cell the row index of
        `emission` array (or -1 for the grid cells with zero emission or no data). This array
        maps 3D spatial grid to the `emission` array.
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
    def material(self):
        return self._primitive.material


class RegularGridCylinder(RegularGridVolume):
    """
    Regular Grid Volume for cylindrical emitter defined on a regular 3D :math:`(R, \phi, Z)` grid.
    The emitter is periodic in :math:`\phi` direction.
    The base of the cylinder is located at `Z = 0` plane. Use `transform`
    parameter to move it.
    
    :param np.ndarray ~.emission: The 2D (row-major) or 4D array containing the spectral emission
        (in :math:`W/(sr\,m^3\,nm)`) defined on a regular 3D grid in cylindrical coordinates:
        :math:`(R, \phi, Z)` (if provided as a 4D array, in axisymmetric case
        `emission.shape[1]` must be equal to 1).
        Spectral emission can be provided either for selected cells of the regular
        grid (2D array) or for all grid cells (4D array).
        If provided for selected cells, the 3D `voxel_map` array must be specified, which
        maps 3D spatial grid to the `emission` array. Providing spectral emission
        only for selected cells is less memory consuming if many grid cells have zero emission.
        The last dimension of `emission` array is the spectral one.
        Spectral resolution of the emission must be equal to
        `(camera.max_wavelength - camera.min_wavelength) / camera.spectral_bins`.
        For memory saving, the data can be provided for selected
        spectral bins only (e.g. if the material does not emit on certain wavelengths of the
        specified wavelength range). In this case, the 1D `spectral_map` array must be provided,
        which maps each spectral slice of `emission` array to the respective spectral bin.
        `RegularGridEmitter` stores spectral emission as a 2D array even if it was provided
        in 4D. If `voxel_map` is not specified, all grid cells containing all-zero
        spectra are deleted automatically. Similar to that, if `spectral_map` is not specified,
        all spectral slices with zero emission anywhere on the spatial grid
        are deleted.
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
    :param np.ndarray spectral_map: The 1D array with
        `spectral_map.size == emission.shape[-1]`, which maps the emission
        array to the respective bins of spectral array specified in the camera
        settings. If not provided, it is assumed that `emission` array contains the data
        for all spectral bins of the spectral range. Defaults to `spectral_map=None`.
    :param np.ndarray voxel_map: The 3D array containing for each grid cell the row index of
        `emission` array (or -1 for the grid cells with zero emission or no data). This array maps
        3D spatial grid to the `emission` array. In axisymmetric case `voxel_map.shape[1]` must be
        equal to 1. This parameter is ignored if spectral emission is
        provided as a 4D array. Defaults to `voxel_map=None`.
    :param float step: The step of integration along the ray (in meters), defaults to
        `0.25*min((radius_outer - radius_inner) / n_r, height / n_z)`, where n_r and n_z are
        the grid resolutions in `R` and `Z` directions.
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
        >>> # Defining wavelength range and creating spectral_map array
        >>> min_wavelength = 457.4 - 0.5 * delta_wavelength
        >>> spectral_map = np.zeros(2, dtype=np.int32)
        >>> spectral_map[1] = int((527.2 - min_wavelength) / delta_wavelength)
        >>> spectral_bins = spectral_map[1] + 1
        >>> max_wavelength = min_wavelength + spectral_bins * delta_wavelength
        >>> # Creating the scene
        >>> world = World()
        >>> pipeline = SpectralRadiancePipeline2D()
        >>> rgc = RegularGridCylinder(emission, min_wavelength, radius_outer=rmax,
                                      height=zmax - zmin, radius_inner=rmin, period=phi_period,
                                      spectral_map=spectral_map,
                                      parent=world, transform=translate(0, 0, zmin))
        ...
        >>> camera.spectral_bins = spectral_bins
        >>> camera.min_wavelength = min_wavelength
        >>> camera.max_wavelength = max_wavelength
        ...
        >>> # If reflections do not change the wavelength, the results for each spectral line
        >>> # can be obtained in W/(m^2 sr) in the following way.
        >>> radiance_4574 = pipeline.frame.mean[:, :, spectral_map[0]] * delta_wavelength
        >>> radiance_5272 = pipeline.frame.mean[:, :, spectral_map[1]] * delta_wavelength
    """

    def __init__(self, emission, wavelengths, radius_outer, height, radius_inner=0, period=360., grid_shape=None,
                 step=None, interpolate=True, extrapolate=True, parent=None, transform=None):
        if 360. % period > 1.e-3:
            raise ValueError("The period %.3f is not a multiple of 360." % period)
        if emission.ndim == 2:
            if grid_shape is None:
                raise ValueError("If 'emission' is a 2D array, 'grid_shape' parameter must be specified.")
            if len(grid_shape) != 3:
                raise ValueError("Argument 'grid_shape' must contain 3 elements.")
            if grid_shape[0] <= 0 or grid_shape[1] <= 0 or grid_shape[2] <= 0:
                raise ValueError('Grid sizes must be > 0.')

        elif emission.ndim == 4:
            grid_shape = (emission.shape[0], emission.shape[1], emission.shape[2])

        else:
            raise ValueError("Argument 'emission' must be a 4D or 2D array.")

        dr = (radius_outer - radius_inner) / grid_shape[0]
        dphi = period / grid_shape[1]
        dz = height / grid_shape[2]
        grid_steps = (dr, dphi, dz)
        eps_r = 1.e-5 * dr
        eps_z = 1.e-5 * dz
        step = step or 0.25 * min(dr, dz)
        material = CylindricalRegularEmitter(grid_shape, grid_steps, emission, wavelengths, interpolate=interpolate, extrapolate=extrapolate,
                                             integrator=CylindricalRegularIntegrator(step), rmin=radius_inner)
        primitive = Subtract(Cylinder(radius_outer - eps_r, height - eps_z), Cylinder(radius_inner + eps_r, height - eps_z),
                             material=material, parent=parent, transform=transform)
        super().__init__(primitive)


class RegularGridBox(RegularGridVolume):
    """
    Regular Grid Volume for rectangular emitter defined on a regular 3D :math:`(X, Y, Z)` grid.
    The grid starts at (0, 0, 0). Use `transform` parameter to move it.

    :param np.ndarray ~.emission: The 2D (row-major) or 4D array containing the spectral emission
        (in :math:`W/(sr\,m^3\,nm)`) defined on a regular 3D grid in Cartesian coordinates.
        Spectral emission can be provided either for selected cells of the regular
        grid (2D array) or for all grid cells (4D array).
        If provided for selected cells, the 3D `voxel_map` array must be specified, which
        maps 3D spatial grid to the `emission` array. Providing spectral emission
        only for selected cells is less memory consuming if many grid cells have zero emission.
        The last dimension of `emission` array is the spectral one.
        Spectral resolution of the emission must be equal to
        `(camera.max_wavelength - camera.min_wavelength) / camera.spectral_bins`.
        For memory saving, the data can be provided for selected
        spectral bins only (e.g. if the material does not emit on certain wavelengths of the
        specified wavelength range). In this case, the 1D `spectral_map` array must be provided,
        which maps each spectral slice of `emission` array to the respective spectral bin.
        `RegularGridEmitter` stores spectral emission as a 2D array even if it was provided
        in 4D. If `voxel_map` is not specified, all grid cells containing all-zero
        spectra are deleted automatically. Similar to that, if `spectral_map` is not specified,
        all spectral slices with zero emission anywhere on the spatial grid
        are deleted.
    :param double min_wavelength: The minimal wavelength which must be equal to
        `camera.min_wavelength`. This parameter is required to correctly process
        dispersive rendering.
    :param float xmax: Upper bound of grid in `X` direction (in meters).
    :param float ymax: Upper bound of grid in `Y` direction (in meters).
    :param float zmax: Upper bound of grid in `Z` direction (in meters).
    :param np.ndarray spectral_map: The 1D array with
        `spectral_map.size == emission.shape[-1]`, which maps the emission
        array to the respective bins of spectral array specified in the camera
        settings. If not provided, it is assumed that `emission` array contains the data
        for all spectral bins of the spectral range. Defaults to `spectral_map=None`.
    :param np.ndarray voxel_map: The 3D array containing for each grid cell the row index of
        `emission` array (or -1 for the grid cells with zero emission or no data). This array maps
        3D spatial grid to the `emission` array. This parameter is ignored if spectral emission is
        provided as a 4D array. Defaults to `voxel_map=None`.
    :param float step: The step of integration along the ray (in meters), defaults to
        `step = 0.25 * min(xmax / n_x, ymax / n_y, zmax / n_z)`, where (n_x, n_y, n_z) is
        the grid resolution.
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
        >>> # Defining wavelength range and creating spectral_map array
        >>> min_wavelength = 457.4 - 0.5 * delta_wavelength
        >>> spectral_map = np.zeros(2, dtype=np.int32)
        >>> spectral_map[1] = int((527.2 - min_wavelength) / delta_wavelength)
        >>> spectral_bins = spectral_map[1] + 1
        >>> max_wavelength = min_wavelength + spectral_bins * delta_wavelength
        >>> # Creating the scene
        >>> world = World()
        >>> pipeline = SpectralRadiancePipeline2D()
        >>> box = RegularGridBox(emission, min_wavelength,
                                 xmax=xmax - xmin, ymax=ymax - ymin, zmax=zmax - zmin,
                                 spectral_map=spectral_map,
                                 parent=world, transform=translate(xmin, ymin, zmin))
        ...
        >>> camera.spectral_bins = spectral_bins
        >>> camera.min_wavelength = min_wavelength
        >>> camera.max_wavelength = max_wavelength
        ...
        >>> # If reflections do not change the wavelength, the results for each spectral line
        >>> # can be obtained in W/(m^2 sr) in the following way.
        >>> radiance_4574 = pipeline.frame.mean[:, :, spectral_map[0]] * delta_wavelength
        >>> radiance_5272 = pipeline.frame.mean[:, :, spectral_map[1]] * delta_wavelength
    """

    def __init__(self, emission, wavelengths, xmax, ymax, zmax, grid_shape=None, step=None,
                 interpolate=True, extrapolate=True, parent=None, transform=None):

        if emission.ndim == 2:
            if grid_shape is None:
                raise ValueError("If 'emission' is a 2D array, 'grid_shape' parameter must be specified.")
            if len(grid_shape) != 3:
                raise ValueError("Argument 'grid_shape' must contain 3 elements.")
            if grid_shape[0] <= 0 or grid_shape[1] <= 0 or grid_shape[2] <= 0:
                raise ValueError('Grid sizes must be > 0.')

        elif emission.ndim == 4:
            grid_shape = (emission.shape[0], emission.shape[1], emission.shape[2])

        else:
            raise ValueError("Argument 'emission' must be a 4D or 2D array.")

        dx = xmax / grid_shape[0]
        dy = ymax / grid_shape[1]
        dz = zmax / grid_shape[2]
        grid_steps = (dx, dy, dz)
        eps_x = 1.e-5 * dx
        eps_y = 1.e-5 * dy
        eps_z = 1.e-5 * dz
        step = step or 0.25 * min(dx, dy, dz)
        material = CartesianRegularEmitter(grid_shape, grid_steps, emission, wavelengths, interpolate=interpolate, extrapolate=extrapolate,
                                           integrator=CartesianRegularIntegrator(step))
        primitive = Box(lower=Point3D(0, 0, 0), upper=Point3D(xmax - eps_x, ymax - eps_y, zmax - eps_z),
                        material=material, parent=parent, transform=transform)
        super().__init__(primitive)
