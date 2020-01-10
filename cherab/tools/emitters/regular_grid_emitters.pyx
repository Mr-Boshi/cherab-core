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
The following emitters and integrators are used in ray transfer objects.
Note that these emitters support other integrators as well, however high performance
with other integrators is not guaranteed.
"""

import numpy as np
from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material cimport VolumeIntegrator, InhomogeneousVolumeEmitter
from libc.math cimport sqrt, atan2, M_PI as pi
cimport numpy as np
cimport cython


cdef class RegularGridIntegrator(VolumeIntegrator):
    """
    Basic class for ray transfer integrators that calculate distances traveled by the ray
    through the voxels defined on a regular grid.

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
    Calculates the distances traveled by the ray through the voxels defined on a regular grid
    in cylindrical coordinate system: :math:`(R, \phi, Z)`. This integrator is used
    with the `CylindricalRayTransferEmitter` material class to calculate ray transfer matrices
    (geometry matrices). The value for each voxel is stored in respective bin of the spectral
    array. It is assumed that the emitter is periodic in :math:`\phi` direction with a period
    equal to `material.period`. The distances traveled by the ray through the voxel is calculated
    approximately and the accuracy depends on the integration step.
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
            int ibin, ispec, it, ir, iphi, iz, ir_current, iphi_current, iz_current, n, nphi, nspec
            double length, t, dt, x, y, z, r, phi, dr, dz, dphi, rmin, period, res
            double[:, :, :, ::1] emission_mv
            int[:] spectral_index_mv

        if not isinstance(material, CylindricalRegularEmitter):
            raise TypeError('Only CylindricalRegularEmitter material is supported by CylindricalRegularIntegrator.')
        if material.min_wavelength != spectrum.min_wavelength:
            raise ValueError("Attributes 'min_wavelength' of the objects 'material' and 'spectrum' must be equal.")
        if material.max_wavelength != spectrum.max_wavelength:
            raise ValueError("Attributes 'max_wavelength' of the objects 'material' and 'spectrum' must be equal.")
        if material.bins != spectrum.bins:
            raise ValueError("Attributes 'bins' of the objects 'material' and 'spectrum' must be equal.")
        start = start_point.transform(world_to_primitive)  # start point in local coordinates
        end = end_point.transform(world_to_primitive)  # end point in local coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.get_length()  # integration length
        if length < 0.1 * self._step:  # return if ray's path is too short
            return spectrum
        direction = direction.normalise()  # normalized direction
        n = max(self._min_samples, <int>(length / self._step))  # number of points along ray's trajectory
        dt = length / n  # integration step
        # cython performs checks on attributes of external class, so it's better to do the checks before the loop
        emission_mv = material.emission_mv
        spectral_index_mv = material.spectral_index_mv
        nspec = material.emission.shape[3]
        nphi = material.emission.shape[1]
        dz = material.dz
        dr = material.dr
        dphi = material.dphi
        period = material.period
        rmin = material.rmin
        ir_current = 0
        iphi_current = 0
        iz_current = 0
        res = 0
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
                phi = (180. / pi) * atan2(y, x)  # phi coordinates of the points (in degrees)
                phi = (phi + 360) % period  # moving into the [0, period) sector (periodic emitter)
                iphi = <int>(phi / dphi)  # phi-indices of grid cells, in which the points are located
            if ir != ir_current or iphi != iphi_current or iz != iz_current:  # we moved to the next cell
                if res:
                    for ispec in range(nspec):
                        ibin = spectral_index_mv[ispec]
                        if ibin > -1:
                            spectrum.samples_mv[ibin] += res * emission_mv[ir_current, iphi_current, iz_current, ispec]
                ir_current = ir
                iphi_current = iphi
                iz_current = iz
                res = 0
            res += dt
        for ispec in range(nspec):
            ibin = spectral_index_mv[ispec]
            if ibin > -1:
                spectrum.samples_mv[ibin] += res * emission_mv[ir_current, iphi_current, iz_current, ispec]

        return spectrum


cdef class CartesianRegularIntegrator(RegularGridIntegrator):
    """
    Calculates the distances traveled by the ray through the voxels defined on a regular grid
    in Cartesian coordinate system: :math:`(X, Y, Z)`. This integrator is used with
    the `CartesianRayTransferEmitter` material to calculate ray transfer matrices (geometry
    matrices). The value for each voxel is stored in respective bin of the spectral array.
    The distances traveled by the ray through the voxel is calculated approximately and
    the accuracy depends on the integration step.
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
            int ibin, ispec, it, ix, iy, iz, ix_current, iy_current, iz_current, n, nspec
            double length, t, dt, x, y, z, dx, dy, dz, res
            double[:, :, :, ::1] emission_mv
            int[:] spectral_index_mv

        if not isinstance(material, CartesianRegularEmitter):
            raise TypeError('Only CartesianRegularEmitter material is supported by CartesianRegularIntegrator')
        if material.min_wavelength != spectrum.min_wavelength:
            raise ValueError("Attributes 'min_wavelength' of the objects 'material' and 'spectrum' must be equal.")
        if material.max_wavelength != spectrum.max_wavelength:
            raise ValueError("Attributes 'max_wavelength' of the objects 'material' and 'spectrum' must be equal.")
        if material.bins != spectrum.bins:
            raise ValueError("Attributes 'bins' of the objects 'material' and 'spectrum' must be equal.")
        start = start_point.transform(world_to_primitive)  # start point in local coordinates
        end = end_point.transform(world_to_primitive)  # end point in local coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.get_length()  # integration length
        if length < 0.1 * self._step:  # return if ray's path is too short
            return spectrum
        direction = direction.normalise()  # normalized direction
        n = max(self._min_samples, <int>(length / self._step))  # number of points along ray's trajectory
        dt = length / n  # integration step
        # cython performs checks on attributes of external class, so it's better to do the checks before the loop
        emission_mv = material.emission_mv
        spectral_index_mv = material.spectral_index_mv
        nspec = material.emission.shape[3]
        dx = material.dx
        dy = material.dy
        dz = material.dz
        ix_current = 0
        iy_current = 0
        iz_current = 0
        res = 0
        for it in range(n):
            t = (it + 0.5) * dt
            x = start.x + direction.x * t  # x coordinates of the points
            y = start.y + direction.y * t  # y coordinates of the points
            z = start.z + direction.z * t  # z coordinates of the points
            ix = <int>(x / dx)  # X-indices of grid cells, in which the points are located
            iy = <int>(y / dy)  # Y-indices of grid cells, in which the points are located
            iz = <int>(z / dz)  # Z-indices of grid cells, in which the points are located
            if ix != ix_current or iy != iy_current or iz != iz_current:  # we moved to the next cell
                if res:
                    for ispec in range(nspec):
                        ibin = spectral_index_mv[ispec]
                        if ibin > -1:
                            spectrum.samples_mv[ibin] += res * emission_mv[ix_current, iy_current, iz_current, ispec]
                ix_current = ix
                iy_current = iy
                iz_current = iz
                res = 0
            res += dt
        for ispec in range(nspec):
            ibin = spectral_index_mv[ispec]
            if ibin > -1:
                spectrum.samples_mv[ibin] += res * emission_mv[ix_current, iy_current, iz_current, ispec]

        return spectrum


cdef class RegularGridLineEmitter(InhomogeneousVolumeEmitter):
    """
    Basic class for ray transfer emitters defined on a regular grid. Ray transfer emitters
    are used to calculate ray transfer matrices (geometry matrices) for a single value
    of wavelength.

    :param tuple grid_shape: The shape of regular grid (the number of grid cells
        along each direction).
    :param tuple grid_steps: The sizes of grid cells along each direction.
    :param np.ndarray voxel_map: An array with shape `grid_shape` containing the indices of
        the light sources. This array maps the cells of regular grid to the respective voxels
        (light sources). The cells with identical indices in `voxel_map` array form a single
        voxel (light source). If `voxel_map[i1, i2, ...] == -1`, the cell with indices
        `(i1, i2, ...)` will not be mapped to any light source. This parameters allows to
        apply a custom geometry (pixelated though) to the light sources.
        Default value: `voxel_map=None`.
    :param np.ndarray mask: A boolean mask array with shape `grid_shape`.
        Allows to include (`mask[...] == True`) or exclude (`mask[...] == False`) the cells
        from the calculation. The ray tranfer matrix will be calculated only for those cells
        for which mask is True. This parameter is ignored if `voxel_map` is provided,
        defaults to `mask=None` (all cells are included).
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator,
        defaults to `integrator=NumericalVolumeIntegrator()`

    :ivar tuple grid_shape: The shape of regular grid.
    :ivar tuple grid_steps: The sizes of grid cells along each direction.
    :ivar np.ndarray voxel_map: An array containing the indices of the light sources.
    :ivar np.ndarray ~.mask: A boolean mask array showing active (True) and inactive
        (False) gird cells.
    :ivar int bins: Number of light sources (the size of spectral array must be equal to this value).
    """

    cdef:
        int[3] _grid_shape
        double[3] _grid_steps
        double _min_wavelength, _max_wavelength
        int _bins, _n_spec
        np.ndarray _emission, _spectral_index
        dict _spectral_line_bin
        public:
            double[:, :, :, ::1] emission_mv
            int[:] spectral_index_mv

    @cython.wraparound(False)
    def __init__(self, dict line_emission, tuple grid_steps, max_wavelength_step=1., VolumeIntegrator integrator=None):

        cdef:
            int i, n_wave
            double step, delta_wavelength
            touple shape
            np.array wavelength
            double[:] wavelength_mv

        # checking and assigning grid_steps
        for step in grid_steps:
            if step <= 0:
                raise ValueError('Grid steps must be > 0.')
        self._grid_steps = grid_steps

        # checking line_emission and assigning grid_shape
        if not line_emission:
            raise ValueError("Argument 'line_emission' is an empty dictionary.")
        shape = next(iter(line_emission.values())).shape
        for value in line_emission.values():
            if value.shape != shape:
                raise ValueError("All arrays in 'line_emission' dictionary must be of the same shape.")
        self._grid_shape = shape

        if max_wavelength_step <= 0:
            raise ValueError("Argument 'max_wavelength_step' must be > 0.")

        # determining minimal distance between spectral lines in line_emission
        wavelength = np.array(line_emission.keys())
        wavelength_mv = wavelength
        self._n_spec = wavelength.size
        delta_wavelength = max_wavelength_step
        for i in range(self._n_spec):
            for j in range(i + 1, self._n_spec):
                delta_wavelength = min(delta_wavelength, abs(wavelength_mv[j] - wavelength_mv[i]))

        # setting wavelength
        self._min_wavelength = wavelength.min() - 0.5 * delta_wavelength
        self._max_wavelength = wavelength.max() + 0.5 * delta_wavelength
        self._bins = <int>((self._max_wavelength - self._min_wavelength) / delta_wavelength) + 1
        delta_wavelength = (self._max_wavelength - self._min_wavelength) / self._bins

        # creating and filling the emission array, spectral_index array and spectral_line_bin dictionary
        self._emission = np.zeros((self._grid_shape[0], self._grid_shape[1], self._grid_shape[2], self._n_spec))
        self.emission_mv = self._emission
        self._spectral_index = np.zeros(self._n_spec, dtype=np.int32)
        self.spectral_index_mv = self._spectral_index

        self._spectral_line_bin = {}

        for i in range(self._n_spec):
            self.spectral_index_mv[i] = <int>((wavelength_mv[i] - self._min_wavelength) / delta_wavelength)
            self._emission[:, :, :, i] = line_emission[wavelength_mv[i]] / delta_wavelength
            self._spectral_line_bin[wavelength_mv[i]] = self._spectral_index_mv[i]

        super().__init__(integrator)

    @cython.cdivision(True)
    @cython.nonecheck(False)
    cpdef set_wavelength(self, double min_wavelength, double max_wavelength, int bins):

        cdef:
            int i, index_new
            double delta_wavelength_old, delta_wavelength
            double wavelength

        if min_wavelength >= max_wavelength:
            raise ValueError("Argument 'max_wavelength' must be > then 'min_wavelength'.")

        if bins < 1:
            raise ValueError("Argument 'bins' must be >= 1.")

        delta_wavelength_old = (self._max_wavelength - self._min_wavelength) / self._bins
        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength
        self._bins = bins
        delta_wavelength = (self._max_wavelength - self._min_wavelength) / self._bins

        self._emission *= delta_wavelength_old / delta_wavelength

        for i, wavelength in enumerate(self._spectral_line_bin.keys()):
            index_new = <int>((wavelength - self._min_wavelength) / delta_wavelength)
            if index_new < 0 or index_new >= self._bins:
                index_new = -1
            self.spectral_index_mv[i] = index_new
            self._spectral_line_bin[wavelength] = index_new

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
    def max_wavelength(self):
        return self._max_wavelength

    @property
    def bins(self):
        return self._bins

    @property
    def spectral_line_bin(self):
        return self._spectral_line_bin

    @property
    def spectral_index(self):
        return self._spectral_index

    @property
    def emission(self):
        return self._emission


cdef class CylindricalRegularLineEmitter(RegularGridLineEmitter):
    """
    A unit emitter defined on a regular 2D (RZ plane) or 3D :math:`(R, \phi, Z)` grid, which
    can be used to calculate ray transfer matrices (geometry matrices) for a single value
    of wavelength.
    In case of 3D grid this emitter is periodic in :math:`\phi` direction.
    Note that for performance reason there are no boundary checks in `emission_function()`,
    or in `CylindricalRayTranferIntegrator`, so this emitter must be placed between a couple
    of coaxial cylinders that act like a bounding box.

    :param tuple grid_shape: The shape of regular :math:`(R, \phi, Z)` (3D case)
        or :math:`(R, Z)` (axisymmetric case) grid.
    :param tuple grid_steps: The sizes of grid cells in `R`, :math:`\phi` and `Z`
        (3D case) or `R` and `Z` (axisymmetric case) directions. The size in :math:`\phi`
        must be provided in degrees (sizes in `R` and `Z` are provided in meters).
    :param np.ndarray voxel_map: An array with shape `grid_shape` containing the indices of
        the light sources. This array maps the cells in :math:`(R, \phi, Z)` space to
        the respective voxels (light sources). The cells with identical indices in `voxel_map`
        array form a single voxel (light source). If `voxel_map[ir, iphi, iz] == -1`, the
        cell with indices `(ir, iphi, iz)` will not be mapped to any light source.
        This parameters allows to apply a custom geometry (pixelated though) to the light
        sources. Default value: `voxel_map=None`.
    :param np.ndarray mask: A boolean mask array with shape `grid_shape`.
        Allows to include (mask[ir, iphi, iz] == True) or exclude (mask[ir, iphi, iz] == False)
        the cells from the calculation. The ray tranfer matrix will be calculated only for
        those cells for which mask is True. This parameter is ignored if `voxel_map` is provided,
        defaults to `mask=None` (all cells are included).
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator, defaults to
        `integrator=CylindricalRayTransferIntegrator(step=0.1*min(grid_shape[0], grid_shape[-1]))`.
    :param float rmin: Lower bound of grid in `R` direction (in meters), defaults to `rmin=0`.
    :param float period: A period in :math:`\phi` direction (in degree).
        Used only in 3D case, defaults to `period=360`.

    :ivar float period: The period in :math:`\phi` direction in 3D case or `None` in
        axisymmetric case.
    :ivar float rmin: Lower bound of grid in `R` direction.
    :ivar float dr: The size of grid cell in `R` direction (equals to `grid_shape[0]`).
    :ivar float dz: The size of grid cell in `Z` direction (equals to `grid_shape[-1]`).
    :ivar float dphi: The size of grid cell in :math:`\phi` direction
        (equals to None in axisymmetric case or to `grid_shape[1]` in 3D case).

    .. code-block:: pycon

        >>> from raysect.optical import World, translate
        >>> from raysect.primitive import Cylinder, Subtract
        >>> from cherab.tools.raytransfer import CylindricalRayTransferEmitter
        >>> world = World()
        >>> grid_shape = (10, 10)
        >>> grid_steps = (0.5, 0.5)
        >>> rmin = 2.5
        >>> material = CylindricalRayTransferEmitter(grid_shape, grid_steps, rmin=rmin)
        >>> eps = 1.e-6  # ray must never leave the grid when passing through the volume
        >>> radius_outer = grid_shape[0] * grid_steps[0] - eps
        >>> height = grid_shape[1] * grid_steps[1] - eps
        >>> radius_inner = rmin + eps
        >>> bounding_box = Subtract(Cylinder(radius_outer, height), Cylinder(radius_inner, height),
                                    material=material, parent=world)  # bounding primitive
        >>> bounding_box.transform = translate(0, 0, -2.5)
        ...
        >>> camera.spectral_bins = material.bins
        >>> # ray transfer matrix will be calculated for 600.5 nm
        >>> camera.min_wavelength = 600.
        >>> camera.max_wavelength = 601.
    """
    cdef:
        double _dr, _dphi, _dz, _period, _rmin

    def __init__(self, dict line_emission, tuple grid_steps, max_wavelength_step=1., VolumeIntegrator integrator=None, double rmin=0):

        cdef:
            double def_integration_step, period

        def_integration_step = 0.25 * min(grid_steps[0], grid_steps[-1])
        integrator = integrator or CylindricalRegularIntegrator(def_integration_step)
        super().__init__(line_emission, grid_steps, max_wavelength_step=max_wavelength_step, integrator=integrator)
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
            int ir, iphi, iz, ispec, ibin
            double r, phi

        if self._min_wavelength != spectrum.min_wavelength:
            raise ValueError("Attributes 'min_wavelength' of the objects 'material' and 'spectrum' must be equal.")
        if self._max_wavelength != spectrum.max_wavelength:
            raise ValueError("Attributes 'max_wavelength' of the objects 'material' and 'spectrum' must be equal.")
        if self._bins != spectrum.bins:
            raise ValueError("Attributes 'bins' of the objects 'material' and 'spectrum' must be equal.")
        iz = <int>(point.z / self._dz)  # Z-index of grid cell, in which the point is located
        r = sqrt(point.x * point.x + point.y * point.y)  # R coordinates of the points
        ir = <int>((r - self._rmin) / self._dr)  # R-index of grid cell, in which the points is located
        if self._grid_shape[1] == 1:  # axisymmetric case
            iphi = 0
        else:
            phi = (180. / pi) * atan2(point.y, point.x)  # phi coordinate of the point (in degrees)
            phi = (phi + 360) % self._period  # moving into the [0, period) sector (periodic emitter)
            iphi = <int>(phi / self._dphi)  # phi-index of grid cell, in which the point is located
        for ispec in range(self._n_spec):
            ibin = spectral_index_mv[ispec]
            if ibin > -1:
                spectrum.samples_mv[ibin] += self.emission_mv[ir, iphi, iz, ispec]

        return spectrum


cdef class CartesianRegularLineEmitter(RegularGridLineEmitter):
    """
    A unit emitter defined on a regular 3D :math:`(X, Y, Z)` grid, which can be used
    to calculate ray transfer matrices (geometry matrices).
    Note that for performance reason there are no boundary checks in `emission_function()`,
    or in `CartesianRayTranferIntegrator`, so this emitter must be placed inside a bounding box.

    :param tuple grid_shape: The shape of regular :math:`(X, Y, Z)` grid.
        The number of points in `X`, `Y` and `Z` directions.
    :param tuple grid_steps: The sizes of grid cells in `X`, `Y` and `Z`
        directions (in meters).
    :param np.ndarray voxel_map: An array with shape `grid_shape` containing the indices
        of the light sources. This array maps the cells in :math:`(X, Y, Z)` space to the
        respective voxels (light sources). The cells with identical indices in `voxel_map`
        array form a single voxel (light source). If `voxel_map[ix, iy, iz] == -1`,
        the cell with indices `(ix, iy, iz)` will not be mapped to any light source.
        This parameters allows to apply a custom geometry (pixelated though) to the
        light sources. Default value: `voxel_map=None`.
    :param np.ndarray mask: A boolean mask array with shape `grid_shape`.
        Allows to include (`mask[ix, iy, iz] == True`) or exclude (`mask[ix, iy, iz] == False`)
        the cells from the calculation. The ray tranfer matrix will be calculated only for
        those cells for which mask is True. This parameter is ignored if `voxel_map` is
        provided, defaults to `mask=None` (all cells are included).
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator,
        defaults to `integrator=CartesianRayTransferIntegrator(step=0.1 * min(grid_steps))`

    :ivar float dx: The size of grid cell in `X` direction (equals to `grid_shape[0]`).
    :ivar float dy: The size of grid cell in `Y` direction (equals to `grid_shape[1]`).
    :ivar float dz: The size of grid cell in `Z` direction (equals to `grid_shape[2]`).

     .. code-block:: pycon

        >>> from raysect.optical import World, translate, Point3D
        >>> from raysect.primitive import Box
        >>> from cherab.tools.raytransfer import CartesianRayTransferEmitter
        >>> world = World()
        >>> grid_shape = (10, 10, 10)
        >>> grid_steps = (0.5, 0.5, 0.5)
        >>> material = CartesianRayTransferEmitter(grid_shape, grid_steps)
        >>> eps = 1.e-6  # ray must never leave the grid when passing through the volume
        >>> upper = Point3D(grid_shape[0] * grid_steps[0] - eps,
                            grid_shape[1] * grid_steps[1] - eps,
                            grid_shape[2] * grid_steps[2] - eps)
        >>> bounding_box = Box(lower=Point3D(0, 0, 0), upper=upper, material=material,
                               parent=world)
        >>> bounding_box.transform = translate(-2.5, -2.5, -2.5)
        ...
        >>> camera.spectral_bins = material.bins
        >>> # ray transfer matrix will be calculated for 600.5 nm
        >>> camera.min_wavelength = 600.
        >>> camera.max_wavelength = 601.
    """

    cdef:
        double _dx, _dy, _dz

    def __init__(self, dict line_emission, tuple grid_steps, max_wavelength_step=1., VolumeIntegrator integrator=None):

        cdef:
            double def_integration_step

        def_integration_step = 0.25 * min(grid_steps)
        integrator = integrator or CartesianRayTransferIntegrator(def_integration_step)
        super().__init__(line_emission, grid_steps, max_wavelength_step=max_wavelength_step, integrator=integrator)
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

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            int isource, ix, iy, iz, ispec, ibin

        if self._min_wavelength != spectrum.min_wavelength:
            raise ValueError("Attributes 'min_wavelength' of the objects 'material' and 'spectrum' must be equal.")
        if self._max_wavelength != spectrum.max_wavelength:
            raise ValueError("Attributes 'max_wavelength' of the objects 'material' and 'spectrum' must be equal.")
        if self._bins != spectrum.bins:
            raise ValueError("Attributes 'bins' of the objects 'material' and 'spectrum' must be equal.")
        ix = <int>(point.x / self._dx)  # X-index of grid cell, in which the point is located
        iy = <int>(point.y / self._dy)  # Y-index of grid cell, in which the point is located
        iz = <int>(point.z / self._dz)  # Z-index of grid cell, in which the point is located
        for ispec in range(self._n_spec):
            ibin = spectral_index_mv[ispec]
            if ibin > -1:
                spectrum.samples_mv[ibin] += self.emission_mv[ix, iy, iz, ispec]

        return spectrum
