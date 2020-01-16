# cython: language_level=3

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
The following emitters and integrators are used in regular grid objects.
They allow fast integration along the ray's trajectory as they use pre-calculated
values of spectral emissivity on a regular grid.
Note that these emitters support other integrators as well, however high performance
with other integrators is not guaranteed.
"""

import numpy as np
from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material cimport VolumeIntegrator, InhomogeneousVolumeEmitter
from libc.math cimport round, sqrt, atan2, M_PI as pi
from libc.stdlib cimport malloc, free
cimport numpy as np
cimport cython
ctypedef np.uint8_t uint8


cdef class RegularGridIntegrator(VolumeIntegrator):
    """
    Basic class for regular grid integrators.

    :param float step: Integration step (in meters), defaults to `step=0.001`.
    :param int min_samples: The minimum number of samples to use over integration range,
        defaults to `min_samples=2`.

    :ivar float step: Integration step.
    :ivar int min_samples: The minimum number of samples to use over integration range.
    """

    cdef:
        double _step
        int _min_samples

    def __init__(self, double step=0.001, int min_samples=2):
        self.step = step
        self.min_samples = min_samples

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        if value <= 0:
            raise ValueError("Numerical integration step size can not be less than or equal to zero.")
        self._step = value

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, value):
        if value < 2:
            raise ValueError("At least two samples are required to perform the numerical integration.")
        self._min_samples = value


cdef class CylindricalRegularIntegrator(RegularGridIntegrator):
    """
    Integrates the spectral emissivity defined on a regular grid
    in cylindrical coordinate system: :math:`(R, \phi, Z)` along the ray's trajectory.
    This integrator must be used with the `CylindricalRegularEmitter` material class. 
    It is assumed that the emitter is periodic in :math:`\phi` direction with a period
    equal to `material.period`.
    This integrator does not perform interpolation, so the spectral emissivity at
    any spatial point along the ray's trajectory is equal to that of the grid cell
    where this point is located.
    """

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D direction
            int i, bins, ibin_start, ray_bins, it, ir, iphi, iz, ir_current, iphi_current, iz_current, n, nphi
            double delta_wavelength, length, t, dt, x, y, z, r, phi, dr, dz, dphi, rmin, period, ray_path
            double[:, :, :, ::1] emission_mv
            uint8[:, :, ::1] notempty_mv
            int[:] spectral_index_mv
            int *ibin
            int *ispec

        if not isinstance(material, CylindricalRegularEmitter):
            raise TypeError('Only CylindricalRegularLineEmitter material is supported by CylindricalRegularIntegrator.')
        # In dispersive rendering, ray samples a small portion of the spectrum.
        # Determining the first spectral bin in this portion.
        ray_bins = ray.get_bins()
        delta_wavelength = (ray.get_max_wavelength() - ray.get_min_wavelength()) / ray_bins
        ibin_start = <int>round((ray.get_min_wavelength() - material.min_wavelength) / delta_wavelength)

        # Determining the spectral emission that fall into the portion of the spectrum sampled by this ray.
        ibin = <int *> malloc(ray_bins * sizeof(int))
        ispec = <int *> malloc(ray_bins * sizeof(int))
        spectral_index_mv = material.spectral_index_mv
        bins = 0
        for i in range(material.emission.shape[3]):
            ibin[bins] = spectral_index_mv[i] - ibin_start
            if -1 < ibin[bins] < ray_bins:
                ispec[bins] = i
                bins += 1
        if not bins:  # return if the material does not emit at this portion of the spectrum
            free(ibin)
            free(ispec)
            return spectrum

        # Cython performs checks on attributes of external class, so it's better to do the checks before the loop
        emission_mv = material.emission_mv
        notempty_mv = material.notempty_mv
        nphi = material.emission.shape[1]
        dz = material.dz
        dr = material.dr
        dphi = material.dphi
        period = material.period
        rmin = material.rmin

        # Determining direction of integration and effective integration step
        start = start_point.transform(world_to_primitive)  # start point in local coordinates
        end = end_point.transform(world_to_primitive)  # end point in local coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.get_length()  # integration length
        if length < 0.1 * self._step:  # return if ray's path is too short
            free(ibin)
            free(ispec)
            return spectrum
        direction = direction.normalise()  # normalized direction
        n = max(self._min_samples, <int>(length / self._step))  # number of points along ray's trajectory
        dt = length / n  # integration step

        # Starting integration
        ir_current = 0
        iphi_current = 0
        iz_current = 0
        ray_path = 0
        for it in range(n):
            t = (it + 0.5) * dt
            x = start.x + direction.x * t  # x coordinates of the points
            y = start.y + direction.y * t  # y coordinates of the points
            z = start.z + direction.z * t  # z coordinates of the points
            iz = <int>(z / dz)  # Z-indices of grid cells, in which the points are located
            r = sqrt(x * x + y * y)  # R coordinates of the points
            ir = <int>((r - rmin) / dr)  # R-indices of grid cells, in which the points are located
            if nphi == 1:  # axisymmetric case
                iphi = 0
            else:
                phi = (180. / pi) * atan2(y, x)  # phi coordinates of the points (in the range [-180, 180))
                phi = (phi + 360) % period  # moving into the [0, period) sector (periodic emitter)
                iphi = <int>(phi / dphi)  # phi-indices of grid cells, in which the points are located
            if ir != ir_current or iphi != iphi_current or iz != iz_current:  # we moved to the next cell
                if notempty_mv[ir_current, iphi_current, iz_current]:  # checking if the cell contains non-zeros data
                    for i in range(bins):
                        spectrum.samples_mv[ibin[i]] += ray_path * emission_mv[ir_current, iphi_current, iz_current, ispec[i]]
                ir_current = ir
                iphi_current = iphi
                iz_current = iz
                ray_path = 0  # zeroing ray's path along the cell
            ray_path += dt
        if notempty_mv[ir_current, iphi_current, iz_current]:
            for i in range(bins):
                spectrum.samples_mv[ibin[i]] += ray_path * emission_mv[ir_current, iphi_current, iz_current, ispec[i]]

        free(ibin)
        free(ispec)
        return spectrum


cdef class CartesianRegularIntegrator(RegularGridIntegrator):
    """
    Integrates the spectral emissivity defined on a regular grid
    in Cartesian coordinate system: :math:`(X, Y, Z)` along the ray's trajectory.
    This integrator must be used with the `CartesianRegularEmitter` material class. 
    This integrator does not perform interpolation, so the spectral emissivity at
    any spatial point along the ray's trajectory is equal to that of the grid cell
    where this point is located.
    """

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D direction
            int i, bins, ibin_start, ray_bins, it, ix, iy, iz, ix_current, iy_current, iz_current, n
            double delta_wavelength, length, t, dt, x, y, z, dx, dy, dz, ray_path
            double[:, :, :, ::1] emission_mv
            uint8[:, :, ::1] notempty_mv
            int[:] spectral_index_mv
            int *ibin
            int *ispec

        if not isinstance(material, CartesianRegularEmitter):
            raise TypeError('Only CartesianRegularLineEmitter material is supported by CartesianRegularIntegrator')
        # In dispersive rendering, ray samples a small portion of the spectrum.
        # Determining the first spectral bin in this portion.
        ray_bins = ray.get_bins()
        delta_wavelength = (ray.get_max_wavelength() - ray.get_min_wavelength()) / ray_bins
        ibin_start = <int>round((ray.get_min_wavelength() - material.min_wavelength) / delta_wavelength)

        # Determining the spectral emission that fall into the portion of the spectrum sampled by this ray.
        ibin = <int *> malloc(ray_bins * sizeof(int))
        ispec = <int *> malloc(ray_bins * sizeof(int))
        spectral_index_mv = material.spectral_index_mv
        bins = 0
        for i in range(material.emission.shape[3]):
            ibin[bins] = spectral_index_mv[i] - ibin_start
            if -1 < ibin[bins] < ray_bins:
                ispec[bins] = i
                bins += 1
        if not bins:  # return if the material does not emit at this portion of the spectrum
            free(ibin)
            free(ispec)
            return spectrum

        # Cython performs checks on attributes of external class, so it's better to do the checks before the loop
        emission_mv = material.emission_mv
        notempty_mv = material.notempty_mv
        spectral_index_mv = material.spectral_index_mv
        nspec = material.emission.shape[3]
        dx = material.dx
        dy = material.dy
        dz = material.dz

        # Determining direction of integration and effective integration step
        start = start_point.transform(world_to_primitive)  # start point in local coordinates
        end = end_point.transform(world_to_primitive)  # end point in local coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.get_length()  # integration length
        if length < 0.1 * self._step:  # return if ray's path is too short
            return spectrum
        direction = direction.normalise()  # normalized direction
        n = max(self._min_samples, <int>(length / self._step))  # number of points along ray's trajectory
        dt = length / n  # integration step

        # Starting integrations
        ix_current = 0
        iy_current = 0
        iz_current = 0
        ray_path = 0
        for it in range(n):
            t = (it + 0.5) * dt
            x = start.x + direction.x * t  # x coordinates of the points
            y = start.y + direction.y * t  # y coordinates of the points
            z = start.z + direction.z * t  # z coordinates of the points
            ix = <int>(x / dx)  # X-indices of grid cells, in which the points are located
            iy = <int>(y / dy)  # Y-indices of grid cells, in which the points are located
            iz = <int>(z / dz)  # Z-indices of grid cells, in which the points are located
            if ix != ix_current or iy != iy_current or iz != iz_current:  # we moved to the next cell
                if notempty_mv[ix_current, iy_current, iz_current]:  # checking if the cell contains non-zeros data
                    for i in range(bins):
                        spectrum.samples_mv[ibin[i]] += ray_path * emission_mv[ix_current, iy_current, iz_current, ispec[i]]
                ix_current = ix
                iy_current = iy
                iz_current = iz
                ray_path = 0  # zeroing ray's path along the cell
            ray_path += dt
        if notempty_mv[ix_current, iy_current, iz_current]:
            for i in range(bins):
                spectrum.samples_mv[ibin[i]] += ray_path * emission_mv[ix_current, iy_current, iz_current, ispec[i]]

        free(ibin)
        free(ispec)
        return spectrum


cdef class RegularGridEmitter(InhomogeneousVolumeEmitter):
    """
    Basic class for the emitters defined on a regular grid.

    :param np.ndarray emission: The 4D array containing the spectral emission
        (in :math:`W/(sr m^3 nm)`) defined on a regular 3D grid.
        The last dimension of this array is the spectral one.
        The spectral resolution of the emission profile must be equal to
        `(camera.max_wavelength - camera.min_wavelength) / camera.spectral_bins`.
        Some of the spectral bins can be skipped (e.g. if the material does not emit on
        certain wavelengths of the specified wavelength range) with the help of the
        `spectral_index` parameter  This allows to save memory.
    :param np.ndarray spectral_index: The 1D array with
        `spectral_index.size == emission.shape[3]`, which maps the spectral emission
        array to the respective spectral bins.
    :param tuple grid_steps: The sizes of grid cells along each direction.
    :param double min_wavelength: The minimal wavelength which must be equal to
        `camera.min_wavelength`. This parameter is required to correctly process
        dispersive rendering.
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator,
        defaults to `integrator=NumericalIntegrator()`.

    :ivar tuple grid_shape: The shape of regular grid.
    :ivar tuple grid_steps: The sizes of grid cells along each direction.
    :ivar np.ndarray emission: Spectral emission (in :math:`W/(sr m^3 nm)`) defined
        on a regular 3D grid.
    :ivar np.ndarray spectral_index: The 1D array, which maps the spectral emission
        array to the respective spectral bins.
    :ivar np.ndarray notempty: The 3D boolean array, which is True where `emission.sum(3)`
        is non-zero.
    :ivar int min_wavelength: The minimal wavelength equal to `camera.min_wavelength`.
    """

    cdef:
        int[3] _grid_shape
        double[3] _grid_steps
        double _min_wavelength
        int _n_spec
        np.ndarray _emission, _spectral_index, _notempty
        public:
            double[:, :, :, ::1] emission_mv
            int[:] spectral_index_mv
            uint8[:, :, ::1] notempty_mv

    @cython.wraparound(False)
    def __init__(self, np.ndarray emission, np.ndarray spectral_index, tuple grid_steps, double min_wavelength, VolumeIntegrator integrator=None):

        cdef:
            double step

        for step in grid_steps:
            if step <= 0:
                raise ValueError('Grid steps must be > 0.')
        self._grid_steps = grid_steps

        if emission.ndim != 4:
            raise ValueError("Argument 'emission' must be a 4D array.")
        if emission.shape[3] != spectral_index.size:
            raise ValueError("The size of 'spectral_index' array must be equal to emission.shape[3].")

        self._grid_shape = (emission.shape[0], emission.shape[1], emission.shape[2])
        self._n_spec = emission.shape[3]
        self._emission = emission
        self.emission_mv = self._emission
        self._spectral_index = spectral_index.astype(np.int32)
        self.spectral_index_mv = self._spectral_index
        self._notempty = self._emission.sum(3).astype(np.bool)
        self.notempty_mv = np.frombuffer(self._notempty, dtype=np.uint8).reshape(self.grid_shape)

        if min_wavelength <= 0:
            raise ValueError("Argument 'min_wavelength' must be > 0.")
        self._min_wavelength = min_wavelength

        super().__init__(integrator)

    @property
    def grid_shape(self):
        return <tuple>self._grid_shape

    @property
    def grid_steps(self):
         return <tuple>self._grid_steps

    @property
    def min_wavelength(self):
        return self._min_wavelength

    @property
    def spectral_index(self):
        return self._spectral_index

    @property
    def notempty(self):
        return self._notempty

    @property
    def emission(self):
        return self._emission


cdef class CylindricalRegularEmitter(RegularGridEmitter):
    """
    Spectral emitter defined on a regular 3D grid in cylindrical coordinates:
    :math:`(R, \phi, Z)`. This emitter is periodic in :math:`\phi` direction.
    Note that for performance reason there are no boundary checks in `emission_function()`,
    or in `CylindricalRegularIntegrator`, so this emitter must be placed between a couple
    of coaxial cylinders that act like a bounding box.

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
    :param tuple grid_steps: The sizes of grid cells in `R`, :math:`\phi` and `Z`
        directions. The size in :math:`\phi` must be provided in degrees (sizes in `R` and `Z`
        are provided in meters). The period in :math:`\phi` direction is defined as
        `emission.shape[1] * grid_steps[1]`. Note that the period must be a multiple of 360.
    :param double min_wavelength: The minimal wavelength which must be equal to
        `camera.min_wavelength`. This parameter is required to correctly process
        dispersive rendering.
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator, defaults to
        `CylindricalRegularIntegrator(step = 0.25 * min(grid_steps[0], grid_steps[2]))`.
    :param float rmin: Lower bound of grid in `R` direction (in meters), defaults to `rmin=0`.

    :ivar float period: The period in :math:`\phi` direction (equals to
        `emission.shape[1] * grid_steps[1]`).
    :ivar float rmin: Lower bound of grid in `R` direction.
    :ivar float dr: The size of grid cell in `R` direction (equals to `grid_steps[0]`).
    :ivar float dphi: The size of grid cell in :math:`\phi` direction (equals to `grid_steps[1]`).
    :ivar float dz: The size of grid cell in `Z` direction (equals to `grid_steps[2]`).

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import World, translate
        >>> from raysect.optical.observer import SpectralRadiancePipeline2D
        >>> from raysect.primitive import Cylinder, Subtract
        >>> from cherab.tools.emitters import CylindricalRegularEmitter
        >>> from cherab.tools.emitters import CylindricalRegularIntegrator 
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
        >>> grid_steps = ((rmax - rmin) / grid_shape[0],
                          phi_period / grid_shape[1],
                          (zmax - zmin) / grid_shape[2])
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
        >>> material = CylindricalRegularEmitter(emission, spectral_index, grid_steps,
                                                 min_wavelength, rmin=rmin)
        >>> eps = 1.e-6  # ray must never leave the grid when passing through the volume
        >>> bounding_box = Subtract(Cylinder(rmax - eps, zmax - zmin - eps),
                                    Cylinder(rmin, zmax - zmin - eps),
                                    material=material, parent=world)  # bounding primitive
        >>> bounding_box.transform = translate(0, 0, zmin)
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
    cdef:
        double _dr, _dphi, _dz, _period, _rmin

    def __init__(self, np.ndarray emission, np.ndarray spectral_index, tuple grid_steps, double min_wavelength,
                 VolumeIntegrator integrator=None, double rmin=0):

        cdef:
            double def_integration_step, period

        def_integration_step = 0.25 * min(grid_steps[0], grid_steps[-1])
        integrator = integrator or CylindricalRegularIntegrator(def_integration_step)
        super().__init__(emission, spectral_index, grid_steps, min_wavelength, integrator=integrator)
        self.rmin = rmin
        self._dr = self._grid_steps[0]
        self._dphi = self._grid_steps[1]
        self._dz = self._grid_steps[2]
        period = self._grid_shape[1] * self._grid_steps[1]
        if 360. % period > 1.e-3:
            raise ValueError("The period %.3f (grid_shape[1] * grid_steps[1]) is not a multiple of 360." % period)
        self._period = period

    @property
    def rmin(self):
        return self._rmin

    @rmin.setter
    def rmin(self, value):
        if value < 0:
            raise ValueError("Attribute 'rmin' must be >= 0.")
        self._rmin = value

    @property
    def period(self):
        return self._period

    @property
    def dr(self):
        return self._dr

    @property
    def dphi(self):
        return self._dphi

    @property
    def dz(self):
        return self._dz

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            int ir, iphi, iz, ispec, ibin, ibin_start, ray_bins
            double r, phi, delta_wavelength

        # In dispersive rendering, ray samples a small portion of the spectrum.
        # Determining the first spectral bin in this portion.
        ray_bins = ray.get_bins()
        delta_wavelength = (ray.get_max_wavelength() - ray.get_min_wavelength()) / ray_bins
        ibin_start = <int>round((ray.get_min_wavelength() - self._min_wavelength) / delta_wavelength)
        # Obtaining the index of the grid cell, where the point is located
        iz = <int>(point.z / self._dz)  # Z-index of grid cell, in which the point is located
        r = sqrt(point.x * point.x + point.y * point.y)  # R coordinates of the points
        ir = <int>((r - self._rmin) / self._dr)  # R-index of grid cell, in which the points is located
        if self._grid_shape[1] == 1:  # axisymmetric case
            iphi = 0
        else:
            phi = (180. / pi) * atan2(point.y, point.x)  # phi coordinates of the points (in the range [-180, 180))
            phi = (phi + 360) % self._period  # moving into the [0, period) sector (periodic emitter)
            iphi = <int>(phi / self._dphi)  # phi-index of grid cell, in which the point is located
        if self.notempty_mv[ir, iphi, iz]:
            for ispec in range(self._n_spec):
                ibin = self.spectral_index_mv[ispec] - ibin_start
                if -1 < ibin < ray_bins:
                    spectrum.samples_mv[ibin] += self.emission_mv[ir, iphi, iz, ispec]

        return spectrum


cdef class CartesianRegularEmitter(RegularGridEmitter):
    """
    Spectral emitter defined on a regular 3D grid in Cartesian coordinates.
    Note that for performance reason there are no boundary checks in `emission_function()`,
    or in `CartesianRegularIntegrator`, so this emitter must be placed inside a bounding box.

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
    :param tuple grid_steps: The sizes of grid cells in `X`, `Y` and `Z`
        directions in meters.
    :param double min_wavelength: The minimal wavelength which must be equal to
        `camera.min_wavelength`. This parameter is required to correctly process
        dispersive rendering.
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator, defaults to
        `CartesianRegularIntegrator(step = 0.25 * min(grid_steps))`.

    :ivar float dx: The size of grid cell in `X` direction (equals to `grid_steps[0]`).
    :ivar float dy: The size of grid cell in `Y` direction (equals to `grid_steps[1]`).
    :ivar float dz: The size of grid cell in `Z` direction (equals to `grid_steps[2]`).

     .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import World, translate, Point3D
        >>> from raysect.primitive import Box
        >>> from raysect.optical.observer import SpectralRadiancePipeline2D
        >>> from cherab.tools.emitters import CartesianRegularEmitter, CartesianRegularIntegrator
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
        >>> grid_steps = ((xmax - xmin) / grid_shape[0],
                          (ymax - ymin) / grid_shape[1],
                          (zmax - zmin) / grid_shape[2])
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
        >>> material = CartesianRegularEmitter(emission, spectral_index, grid_steps, min_wavelength)
        >>> eps = 1.e-6  # ray must never leave the grid when passing through the volume
        >>> bounding_box = Box(lower=Point3D(0, 0, 0),
                               upper=Point3D(xmax-xmin-eps, ymax-ymin-eps, zmax-zmin-eps),
                               material=material,
                               parent=world)
        >>> bounding_box.transform = translate(xmin, ymin, zmin)
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

    cdef:
        double _dx, _dy, _dz

    def __init__(self, np.ndarray emission, np.ndarray spectral_index, tuple grid_steps, double min_wavelength, VolumeIntegrator integrator=None):

        cdef:
            double def_integration_step

        def_integration_step = 0.25 * min(grid_steps)
        integrator = integrator or CartesianRegularIntegrator(def_integration_step)
        super().__init__(emission, spectral_index, grid_steps, min_wavelength, integrator=integrator)
        self._dx = self._grid_steps[0]
        self._dy = self._grid_steps[1]
        self._dz = self._grid_steps[2]

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def dz(self):
        return self._dz

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            int isource, ix, iy, iz, ispec, ibin, ibin_start, ray_bins
            double delta_wavelength

        # In dispersive rendering, ray samples a small portion of the spectrum.
        # Determining the first spectral bin in this portion.
        ray_bins = ray.get_bins()
        delta_wavelength = (ray.get_max_wavelength() - ray.get_min_wavelength()) / ray_bins
        ibin_start = <int>round((ray.get_min_wavelength() - self._min_wavelength) / delta_wavelength)
        ix = <int>(point.x / self._dx)  # X-index of grid cell, in which the point is located
        iy = <int>(point.y / self._dy)  # Y-index of grid cell, in which the point is located
        iz = <int>(point.z / self._dz)  # Z-index of grid cell, in which the point is located
        if self.notempty_mv[ix, iy, iz]:
            for ispec in range(self._n_spec):
                ibin = self.spectral_index_mv[ispec] - ibin_start
                if -1 < ibin < ray_bins:
                    spectrum.samples_mv[ibin] += self.emission_mv[ix, iy, iz, ispec]

        return spectrum
