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
The following emitters and integrators are used in Regular Grid Volumes.
They allow fast integration along the ray's trajectory as they use pre-calculated
values of spectral emissivity on a regular grid.
Note that these emitters support other integrators as well, however high performance
with other integrators is not guaranteed.
"""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from raysect.core.math.cython cimport find_index
from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material cimport VolumeIntegrator, InhomogeneousVolumeEmitter
from libc.math cimport sqrt, atan2, M_PI as pi
cimport numpy as np
cimport cython


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
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D direction
            int it, ir, iphi, iz, ir_current, iphi_current, iz_current, n
            double length, t, dt, x, y, z, r, phi, ray_path
            CylindricalRegularEmitter emitter

        if not isinstance(material, CylindricalRegularEmitter):
            raise TypeError('Only CylindricalRegularEmitter material is supported by CylindricalRegularIntegrator.')

        emitter = material
        # Building the cache if required
        emitter.cache_build(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_bins())

        if emitter.cache_empty():  # emitter does not emit at this wavelength range
            return spectrum

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
            iz = <int>(z / emitter.get_dz())  # Z-indices of grid cells, in which the points are located
            r = sqrt(x * x + y * y)  # R coordinates of the points
            ir = <int>((r - emitter.get_rmin()) / emitter.get_dr())  # R-indices of grid cells, in which the points are located
            if emitter.get_grid_shape_1() == 1:  # axisymmetric case
                iphi = 0
            else:
                phi = (180. / pi) * atan2(y, x)  # phi coordinates of the points (in the range [-180, 180))
                phi = (phi + 360) % emitter.get_period()  # moving into the [0, period) sector (periodic emitter)
                iphi = <int>(phi / emitter.get_dphi())  # phi-indices of grid cells, in which the points are located
            if ir != ir_current or iphi != iphi_current or iz != iz_current:  # we moved to the next cell
                emitter.add_to_memoryview(spectrum.samples_mv, ir_current, iphi_current, iz_current, ray_path)
                ir_current = ir
                iphi_current = iphi
                iz_current = iz
                ray_path = 0  # zeroing ray's path along the cell
            ray_path += dt
            emitter.add_to_memoryview(spectrum.samples_mv, ir_current, iphi_current, iz_current, ray_path)

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
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D direction
            int it, ix, iy, iz, ix_current, iy_current, iz_current, n
            double length, t, dt, x, y, z, ray_path
            CartesianRegularEmitter emitter

        if not isinstance(material, CartesianRegularEmitter):
            raise TypeError('Only CartesianRegularEmitter material is supported by CartesianRegularIntegrator')

        emitter = material
        # Building the cache if required
        emitter.cache_build(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_bins())

        if emitter.cache_empty():  # material does not emit at this wavelength range
            return spectrum

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
            ix = <int>(x / emitter.get_dx())  # X-indices of grid cells, in which the points are located
            iy = <int>(y / emitter.get_dy())  # Y-indices of grid cells, in which the points are located
            iz = <int>(z / emitter.get_dz())  # Z-indices of grid cells, in which the points are located
            if ix != ix_current or iy != iy_current or iz != iz_current:  # we moved to the next cell
                emitter.add_to_memoryview(spectrum.samples_mv, ix_current, iy_current, iz_current, ray_path)
                ix_current = ix
                iy_current = iy
                iz_current = iz
                ray_path = 0  # zeroing ray's path along the cell
            ray_path += dt
            emitter.add_to_memoryview(spectrum.samples_mv, ix_current, iy_current, iz_current, ray_path)

        return spectrum


cdef class RegularGridEmitter(InhomogeneousVolumeEmitter):
    """
    Basic class for the emitters defined on a regular 3D grid with directrly accessible cache.
    The emission anywhere outside the specified grid is zero.

    :param tuple grid_shape: The number of grid cells along each direction.
    :param tuple grid_steps: The sizes of grid cells along each direction.
    :param object ~.emission: The 2D or 4D array or sparse matrix containing the
        emission defined on a regular 3D grid in :math:`W/(str\,m^3\,nm)` (contineous spectrum)
        or in :math:`W/(str\,m^3)` (discrete spectrum, like in atoms).
        Spectral emission can be provided either for selected cells of the regular
        grid (2D array or sparse matrix) or for all grid cells (4D array).
        Note that if provided as a 2D array (or sparse matrix), the spatial index `(i, j, k)`
        must be flattened in a row-major order:
        `iflat = grid_shape[1] * grid_shape[2] * i + grid_shape[2] * j + k`.
        Regardless of the form in which the emission is provided, the last axis is the
        spectral one.  The emission will be stored as a сompressed sparse column matrix
        (`scipy.sparse.csc_matrix`).
        To reduce memory consumption, provide it as `csc_matrix`.
    :param ndarray wavelengths: The 1D array of wavelengths corresponding to the last axis of
        provided emission array. The size of this array must be equal to `emission.shape[-1]`.
        To increase initialisation speed, this array must be sorted.
    :param bool interpolate: Defines, whether the emitter has contineous (`interpolate=True`)
        or discrete (`interpolate=False`) spectrum. In the case of discrete spectrum, the
        emission must be provided in :math:`W/(str\,m^3)`. Defaults to `interpolate=True`
        (spectral emission is interpolated between the provided wavelengths).
    :param bool extrapolate: In the case of contineous spectrum defines whether the emission
        outside the provided spectral range will be equal the emission at the borders of this
        range (nearest-neighbour extrapolation) or to zero. This option is ignored if
        `interpolate=False`. Defaults to `extrapolate=True`.
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator,
        defaults to `integrator=NumericalIntegrator()`.

    :ivar tuple grid_shape: The shape of regular grid.
    :ivar tuple grid_steps: The sizes of grid cells along each direction.
    :ivar csc_matrix ~.emission: The emission defined on a regular grid stored as a a сompressed
        sparse column matrix (`scipy.sparse.csc_matrix`).
    :ivar np.ndarray wavelengths: The wavelengths corresponding to the emission array.
    :ivar int nvoxel: Total number of grid cells in the spatial grid.
    :ivar bool interpolate: Defines whether the emitter has a contineous (`True`) or discrete
        (`False`) spectrum.
    :ivar bool interpolate: Defines whether the emission spectrum is interpolated outside the
        provided wavelength range (`True`) or not (`False`).
    :ivar csr_matrix cache: Cached spectral emission corresponding to the current spectral
        settings of the ray. The cache is rebuilt each time the ray's spectral properties change.

    Performance tips:

      * Current version of `RegularGridEmitter` does not supports grids with more than
        2147483647 grid cells or the caches with more than 2147483647 non-zero data points
        (> 16 GB of data). If this an issue, try to divide the grid into several parts and
        distribure it between multiple emitters.

      * If dispesive rendering is off (`camera.spectral_rays = 1`) and spectral properties of
        rays do not change during rendering, consider calling:
        .. code-block:: pycon
          >>> emitter.build_cache(camera.min_wavelength, camera.max_wavelength,
                                  camera.spectral_bins)
        before the first call of `camera.observe()`. This will save a lot of memory in case of
        multi-process rendering, as well as some time between the calls of `camera.observe()`.

      * In case of insufficient memory, one can initialise the emitter with a dummy emission
        array and then populate the cache directly with a pre-calculated `csr_matrix`.
        .. code-block:: pycon
          >>> grid_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
          >>> wavelengths = np.ones(1)
          >>> emission = csc_matrix((grid_size, 1))
          >>> emitter = RegularGridEmitter(grid_shape, grid_steps, emission, wavelengths)
          >>> emitter.cache_override(cache, camera.min_wavelength, camera.max_wavelength)
        Note that `cache.shape` must be equal to `(grid_size, camera.spectral_bins)`.
        This solution will work only if dispesive rendering is off (`camera.spectral_rays = 1`)
        and spectral properties of rays do not change during rendering.
    """

    cdef:
        int[3] _grid_shape
        double[3] _grid_steps
        int _nvoxel
        double _cache_min_wvl, _cache_max_wvl
        int _cache_num_samp
        int _cache_data_size
        bint _interpolate, _extrapolate
        np.ndarray _wavelengths
        object _emission, _cache

        public:
            double[::1] wavelengths_mv
            const double[::1] cache_data_mv
            const int[::1] cache_indptr_mv
            const int[::1] cache_indices_mv

    def __init__(self, tuple grid_shape, tuple grid_steps, object emission, np.ndarray wavelengths,
                 bint interpolate=True, bint extrapolate=True, VolumeIntegrator integrator=None):

        cdef:
            np.ndarray indx_sorted
            double step
            int i

        for step in grid_steps:
            if step <= 0:
                raise ValueError('Grid steps must be > 0.')
        self._grid_steps = grid_steps

        for i in grid_shape:
            if i <= 0:
                raise ValueError('Grid sizes must be > 0.')
        if self._grid_shape[0] * self._grid_shape[1] * self._grid_shape[2] > np.iinfo('int32').max:
            raise ValueError('Grids with more than %d cells are not supported.' % np.iinfo('int32').max +
                             'Divide the grid into several parts and distribure it between mutiple emitters.')
        self._grid_shape = grid_shape

        self._nvoxel = self._grid_shape[0] * self._grid_shape[1] * self._grid_shape[2]

        if emission.ndim == 2:
            if emission.shape[0] != self._nvoxel:
                raise ValueError("The number of rows in 'emission' array does not match the grid size.")
            self._emission = csc_matrix(emission)  # this does not create a copy if emission is already a csc_matrix

        elif emission.ndim == 4:
            if emission.shape[0] != self._grid_shape[0] or emission.shape[1] != self._grid_shape[1] or emission.shape[2] != self._grid_shape[2]:
                raise ValueError("The shape of 'emission' array does not match the grid shape.")
            self._emission = csc_matrix(emission.reshape(self._nvoxel, emission.shape[3]))

        else:
            raise ValueError("Argument 'emission' must be a 4D or 2D array.")

        if self._emission.indptr.dtype != np.int32 or self._emission.indices.dtype != np.int32:
            raise RuntimeError("Constructed 'emission' sparse matrix has np.int64 indices." +
                               "Probably, emission data is too large (> 16 GB)." +
                               "Try to divide the grid into several parts and distribure it between multiple emitters.")

        if wavelengths.size != self._emission.shape[1]:
            raise ValueError("The size of 'wavelengths' array does not match 'emission.shape[-1]'.")
        if np.any(wavelengths < 0):
            raise ValueError("Wavelengths must be >= 0.")

        if np.any(np.diff(wavelengths) < 0):  # sorting the arrays if required
            indx_sorted = np.argsort(wavelengths)
            self._wavelengths = wavelengths[indx_sorted].astype(np.float64)
            self._emission = self._emission[:, indx_sorted]
        else:
            self._wavelengths = wavelengths.astype(np.float64)

        self.wavelengths_mv = self._wavelengths

        self._interpolate = interpolate
        self._extrapolate = extrapolate

        self._cache_init()

        super().__init__(integrator)

    @property
    def grid_shape(self):
        return <tuple>self._grid_shape

    @property
    def grid_steps(self):
        return <tuple>self._grid_steps

    @property
    def nvoxel(self):
        return self._nvoxel

    @property
    def wavelengths(self):
        return self._wavelengths

    @property
    def emission(self):
        return self._emission

    @property
    def cache(self):
        return self._cache

    @property
    def interpolate(self):
        return self._interpolate

    @interpolate.setter
    def interpolate(self, bint value):
        self._interpolate = value

    @property
    def extrapolate(self):
        return self._extrapolate

    @extrapolate.setter
    def extrapolate(self, bint value):
        self._extrapolate = value

    cdef int get_grid_shape_0(self) nogil:

        return self._grid_shape[0]

    cdef int get_grid_shape_1(self) nogil:

        return self._grid_shape[1]

    cdef int get_grid_shape_2(self) nogil:

        return self._grid_shape[2]

    cdef double get_grid_steps_0(self) nogil:

        return self._grid_steps[0]

    cdef double get_grid_steps_1(self) nogil:

        return self._grid_steps[1]

    cdef double get_grid_steps_2(self) nogil:

        return self._grid_steps[2]

    @cython.nonecheck(False)
    cdef int get_voxel_index(self, int i, int j, int k) nogil:
        if i < 0 or i >= self._grid_shape[0] or j < 0 or j >= self._grid_shape[1] or k < 0 or k >= self._grid_shape[2]:
            return -1  # out of grid

        return i * self._grid_shape[1] * self._grid_shape[2] + j * self._grid_shape[2] + k

    cpdef int voxel_index(self, int i, int j, int k):

        return self.get_voxel_index(i, j, k)

    cdef void _cache_init(self):
        """
        Initialises the cache.
        """

        # initialise cache with invalid values
        self._cache = None
        self.cache_data_mv = None
        self.cache_indptr_mv = None
        self.cache_indices_mv = None
        self._cache_data_size = -1
        self._cache_min_wvl = -1
        self._cache_max_wvl = -1
        self._cache_num_samp = -1

    cpdef bint cache_valid(self, double min_wavelength, double max_wavelength, int bins):
        """
        Returns true if a suitable cached data are available.

            :param double min_wavelength: The minimum wavelength in nanometers.
            :param double max_wavelength: The maximum wavelength in nanometers.
            :param int bins: The number of spectral bins.
        """

        return (
            self._cache_min_wvl == min_wavelength and
            self._cache_max_wvl == max_wavelength and
            self._cache_num_samp == bins
        )

    cpdef bint cache_empty(self):
        """
        Returns true if the cached data does not contain non-zero elements or the cache is not
        built.
        """

        return self._cache_data_size <= 0

    cpdef void cache_override(self, object cache, double min_wavelength, double max_wavelength):
        """
        Overrides the cache with the provided compressed sparse row matrix.

            :param csr_matrix cache: The cache pre-calculated for the spectral properties of rays.
            :param double min_wavelength: The minimum wavelength in nanometers.
            :param double max_wavelength: The maximum wavelength in nanometers.

        Use this in case of insufficient memory.
        .. code-block:: pycon
          >>> grid_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
          >>> wavelengths = np.ones(1)
          >>> emission = csc_matrix((grid_size, 1))
          >>> emitter = RegularGridEmitter(grid_shape, grid_steps, emission, wavelengths)
          >>> emitter.cache_override(cache, camera.min_wavelength, camera.max_wavelength)

        Note that `cache.shape` must be equal to `(grid_size, camera.spectral_bins)`.
        This solution will work only if dispesive rendering is off (`camera.spectral_rays = 1`)
        and spectral properties of rays do not change during rendering.
        """

        if not isinstance(cache, csr_matrix):
            raise TypeError("Argument 'cache' must be a 'csr_matrix' instance.")

        if cache.shape[0] != self._nvoxel:
            raise ValueError('Provided cache matrix does not match the grid size.')

        if cache.indptr.dtype != np.int32 or cache.indices.dtype != np.int32:
            raise ValueError('Provided cache matrix must have np.int64 indices.' +
                             'Divide the grid into several parts and distribure it between mutiple emitters if it is too large.')

        self._cache = cache
        self.cache_data_mv = self._cache.data
        self.cache_indptr_mv = self.cache.indptr
        self.cache_indices_mv = self.cache.indices
        self._cache_data_size = self._cache.data.size
        self._cache_min_wvl = min_wavelength
        self._cache_max_wvl = max_wavelength
        self._cache_num_samp = self._cache.shape[1]

    cpdef void cache_build(self, double min_wavelength, double max_wavelength, int bins, bint forced=False):
        """
        Builds a new cache if the old one does not match the wavelength range.
            :param double min_wavelength: The minimum wavelength in nanometers.
            :param double max_wavelength: The maximum wavelength in nanometers.
            :param int bins: The number of spectral bins.
            :param bool forces: Rebuild the cache even if the old cache matches the wavelength
                range, defaults to `forced=False`
        """

        if (not forced) and self.cache_valid(min_wavelength, max_wavelength, bins):
            return

        self._cache_init()  # deleting current cache

        if self._interpolate:
            self._cache = self._cache_build_contineous(min_wavelength, max_wavelength, bins)
        else:
            self._cache = self._cache_build_discrete(min_wavelength, max_wavelength, bins)

        if self._cache.indptr.dtype != np.int32 or self._cache.indices.dtype != np.int32:
            raise ValueError('Constructed cache matrix has np.int64 indices.' +
                             'Try to divide the grid into several parts and distribure it between mutiple emitters.')

        self.cache_data_mv = self._cache.data
        self.cache_indptr_mv = self.cache.indptr
        self.cache_indices_mv = self.cache.indices
        self._cache_data_size = self._cache.data.size
        self._cache_min_wvl = min_wavelength
        self._cache_max_wvl = max_wavelength
        self._cache_num_samp = bins

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cdef object _cache_build_discrete(self, double min_wavelength, double max_wavelength, int bins):
        """
        Builds the cache with discrete spectrum.
        """

        cdef:
            np.ndarray data, row_ind, col_ind
            double delta
            int i, indx, istart, iend

        data = np.array([])
        row_ind = np.array([], dtype=np.int32)
        col_ind = np.array([], dtype=np.int32)
        delta = (max_wavelength - min_wavelength) / bins
        for i in range(self.wavelengths.size):
            indx = <int>((self.wavelengths_mv[i] - min_wavelength) / delta)
            if -1 < indx < bins:
                istart = self._emission.indptr[i]
                iend = self._emission.indptr[i + 1]
                data = np.concatenate((data, self._emission.data[istart:iend] / delta))
                col_ind = np.concatenate((col_ind, indx * np.ones(iend - istart, dtype=np.int32)))
                row_ind = np.concatenate((row_ind, self._emission.indices[istart:iend]))

        return csr_matrix((data, (row_ind, col_ind)), shape=(self._nvoxel, bins))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cdef object _cache_build_contineous(self, double min_wavelength, double max_wavelength, int bins):
        """
        Builds the cache with contineous spectrum.
        """

        cdef:
            object bin_integral
            np.ndarray data, row_ind, col_inds
            double delta, lower, upper
            int i

        data = np.array([])
        row_ind = np.array([], dtype=np.int32)
        col_ind = np.array([], dtype=np.int32)
        delta = (max_wavelength - min_wavelength) / bins
        lower = min_wavelength
        for i in range(bins):
            upper = min_wavelength + (i + 1) * delta
            bin_integral = self.integrate(lower, upper)
            data = np.concatenate((data, bin_integral.data / delta))
            col_ind = np.concatenate((col_ind, i * np.ones(bin_integral.data.size, dtype=np.int32)))
            row_ind = np.concatenate((row_ind, bin_integral.indices))
            lower = upper

        return csr_matrix((data, (row_ind, col_ind)), shape=(self._nvoxel, bins))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef object integrate(self, double min_wavelength, double max_wavelength):
        """
        Integrate the emission in the specified wavelength range and returns the result in the
        form of one-column `csc_matrix`.

            :param float min_wavelength: The minimum wavelength in nanometers.
            :param float max_wavelength: The maximum wavelength in nanometers.

            :return: Integrated emission in :math:`W/(str\,m^3)`.
        """

        cdef:
            object integral_sum, y
            double weight, x0, x1
            double[::1] x
            int index, lower_index, upper_index, top_index

        # invalid range
        if max_wavelength <= min_wavelength:
            return csc_matrix((self._nvoxel, 1))

        # doing this for code readability
        x0 = min_wavelength
        x1 = max_wavelength
        x = self.wavelengths_mv
        y = self._emission

        # identify array indices that lie between requested values
        lower_index = find_index(x, x0) + 1
        upper_index = find_index(x, x1)

        # are both points below the bottom of the array?
        if upper_index == -1:

            if self._extrapolate:
                # extrapolate from first array value (nearest-neighbour)
                return y[:, 0] * (x1 - x0)
            # return zero matrix if extrapolate is set to False
            return csc_matrix((self._nvoxel, 1))

        # are both points beyond the top of the array?
        top_index = x.shape[0] - 1
        if lower_index > top_index:

            if self._extrapolate:
                # extrapolate from last array value (nearest-neighbour)
                return y[:, top_index] * (x1 - x0)
            # return zero matrix if extrapolate is set to False
            return csc_matrix((self._nvoxel, 1))

        # numerically integrate array
        if lower_index > upper_index:

            # both values lie inside the same array segment
            # the names lower_index and upper_index are now misnomers, they are swapped!
            weight = (0.5 * (x1 + x0) - x[upper_index]) / (x[lower_index] - x[upper_index])

            # trapezium rule integration
            return (y[:, upper_index] + weight * (y[:, lower_index] - y[:, upper_index])) * (x1 - x0)

        else:

            integral_sum = csc_matrix((self._nvoxel, 1))

            if lower_index == 0:

                # add contribution from point below array
                integral_sum += y[:, 0] * (x[0] - x0)

            else:

                # add lower range partial cell contribution
                weight = (x0 - x[lower_index - 1]) / (x[lower_index] - x[lower_index - 1])

                # trapezium rule integration
                integral_sum += (0.5 * (x[lower_index] - x0)) * (y[:, lower_index - 1] + y[:, lower_index] +
                                                                 weight * (y[:, lower_index] - y[:, lower_index - 1]))

            # sum up whole cell contributions
            for index in range(lower_index, upper_index):

                # trapezium rule integration
                integral_sum += 0.5 * (y[:, index] + y[:, index + 1]) * (x[index + 1] - x[index])

            if upper_index == top_index:

                # add contribution from point above array
                integral_sum += y[:, top_index] * (x1 - x[top_index])

            else:

                # add upper range partial cell contribution
                weight = (x1 - x[upper_index]) / (x[upper_index + 1] - x[upper_index])

                # trapezium rule integration
                integral_sum += (0.5 * (x1 - x[upper_index])) * (2 * y[:, upper_index] + weight * (y[:, upper_index + 1] - y[:, upper_index]))

            return integral_sum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cdef inline void add_to_memoryview(self, double[::1] samples_mv, int i, int j, int k, double ray_path) nogil:

        cdef:
            int ivoxel, ispec

        ivoxel = self.get_voxel_index(i, j, k)
        if ivoxel > -1:  # checking if we are inside the grid
            for ispec in range(self.cache_indptr_mv[ivoxel], self.cache_indptr_mv[ivoxel + 1]):
                samples_mv[self.cache_indices_mv[ispec]] += ray_path * self.cache_data_mv[ispec]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef void add_to_array(self, np.ndarray samples, int i, int j, int k, double ray_path):

        cdef:
            int ivoxel, ispec

        ivoxel = self.get_voxel_index(i, j, k)
        if ivoxel > -1:  # checking if we are inside the grid
            for ispec in range(self.cache_indptr_mv[ivoxel], self.cache_indptr_mv[ivoxel + 1]):
                samples[self.cache_indices_mv[ispec]] += ray_path * self.cache_data_mv[ispec]


cdef class CylindricalRegularEmitter(RegularGridEmitter):
    """
    Spectral emitter defined on a regular 3D grid in cylindrical coordinates:
    :math:`(R, \phi, Z)`. This emitter is periodic in :math:`\phi` direction.
    Note that for performance reason there are no boundary checks in `emission_function()`,
    or in `CylindricalRegularIntegrator`, so this emitter must be placed between a couple
    of coaxial cylinders that act like a bounding box.

    :param np.ndarray ~.emission: The 2D (row-major) or 4D array containing the spectral emission
        (in :math:`W/(str\,m^3\,nm)`) defined on a regular 3D grid in cylindrical coordinates:
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
    :param tuple grid_steps: The sizes of grid cells in `R`, :math:`\phi` and `Z`
        directions. The size in :math:`\phi` must be provided in degrees (sizes in `R` and `Z`
        are provided in meters). The period in :math:`\phi` direction is defined as
        `n_phi * grid_steps[1]`, where n_phi is the grid resolution in phi direction.
        Note that the period must be a multiple of 360.
    :param double min_wavelength: The minimal wavelength which must be equal to
        `camera.min_wavelength`. This parameter is required to correctly process
        dispersive rendering.
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
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator, defaults to
        `CylindricalRegularIntegrator(step = 0.25 * min(grid_steps[0], grid_steps[2]))`.
    :param float rmin: Lower bound of grid in `R` direction (in meters), defaults to `rmin=0`.

    :ivar float period: The period in :math:`\phi` direction (equals to
        `n_phi * grid_steps[1]`, where n_phi is the grid resolution in phi direction).
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
        >>> emission[:, :, :, 0] = emission_4574 / (4. * np.pi * delta_wavelength) # W/(m^3 sr nm)
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
        >>> material = CylindricalRegularEmitter(emission, grid_steps, min_wavelength,
                                                 spectral_map=spectral_map, rmin=rmin)
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
        >>> radiance_4574 = pipeline.frame.mean[:, :, spectral_map[0]] * delta_wavelength
        >>> radiance_5272 = pipeline.frame.mean[:, :, spectral_map[1]] * delta_wavelength

    """
    cdef:
        double _dr, _dphi, _dz, _period, _rmin

    def __init__(self, tuple grid_shape, tuple grid_steps, object emission, np.ndarray wavelengths,
                 bint interpolate=True, bint extrapolate=True, VolumeIntegrator integrator=None, double rmin=0):

        cdef:
            double period

        integrator = integrator or CylindricalRegularIntegrator(0.25 * min(grid_steps[0], grid_steps[2]))
        super().__init__(grid_shape, grid_steps, emission, wavelengths, interpolate=interpolate, extrapolate=extrapolate, integrator=integrator)
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

    cdef double get_rmin(self) nogil:
        return self._rmin

    cdef double get_period(self) nogil:
        return self._period

    cdef double get_dr(self) nogil:
        return self._dr

    cdef double get_dphi(self) nogil:
        return self._dphi

    cdef double get_dz(self) nogil:
        return self._dz

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            int ir, iphi, iz
            double r, phi, delta_wavelength

        # Building the cache if required
        self.cache_build(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_bins())

        if self.cache_empty():  # emitter does not emit at this wavelength range
            return spectrum

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
        self.add_to_memoryview(spectrum.samples_mv, ir, iphi, iz, 1.0)

        return spectrum


cdef class CartesianRegularEmitter(RegularGridEmitter):
    """
    Spectral emitter defined on a regular 3D grid in Cartesian coordinates.
    Note that for performance reason there are no boundary checks in `emission_function()`,
    or in `CartesianRegularIntegrator`, so this emitter must be placed inside a bounding box.

    :param np.ndarray ~.emission: The 2D (row-major) or 4D array containing the spectral emission
        (in :math:`W/(str\,m^3\,nm)`) defined on a regular 3D grid in Cartesian coordinates.
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
    :param tuple grid_steps: The sizes of grid cells in `X`, `Y` and `Z`
        directions in meters.
    :param double min_wavelength: The minimal wavelength which must be equal to
        `camera.min_wavelength`. This parameter is required to correctly process
        dispersive rendering.
    :param np.ndarray spectral_map: The 1D array with
        `spectral_map.size == emission.shape[-1]`, which maps the emission
        array to the respective bins of spectral array specified in the camera
        settings. If not provided, it is assumed that `emission` array contains the data
        for all spectral bins of the spectral range. Defaults to `spectral_map=None`.
    :param np.ndarray voxel_map: The 3D array containing for each grid cell the row index of
        `emission` array (or -1 for the grid cells with zero emission or no data). This array maps
        3D spatial grid to the `emission` array. This parameter is ignored if spectral emission is
        provided as a 4D array. Defaults to `voxel_map=None`.
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
        >>> emission[:, :, :, 0] = emission_4574 / (4. * np.pi * delta_wavelength) # W/(m^3 sr nm)
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
        >>> material = CartesianRegularEmitter(emission, grid_steps, min_wavelength,
                                               spectral_map=spectral_map)
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
        >>> radiance_4574 = pipeline.frame.mean[:, :, spectral_map[0]] * delta_wavelength
        >>> radiance_5272 = pipeline.frame.mean[:, :, spectral_map[1]] * delta_wavelength

    """

    cdef:
        double _dx, _dy, _dz

    def __init__(self, tuple grid_shape, tuple grid_steps, object emission, np.ndarray wavelengths,
                 bint interpolate=True, bint extrapolate=True, VolumeIntegrator integrator=None):

        integrator = integrator or CartesianRegularIntegrator(0.25 * min(grid_steps))
        super().__init__(grid_shape, grid_steps, emission, wavelengths, interpolate=interpolate, extrapolate=extrapolate, integrator=integrator)
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

    cdef double get_dx(self) nogil:
        return self._dx

    cdef double get_dy(self) nogil:
        return self._dy

    cdef double get_dz(self) nogil:
        return self._dz

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            int ix, iy, iz
            double delta_wavelength

        # Building the cache if required
        self.cache_build(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_bins())

        if self.cache_empty():  # emitter does not emit at this wavelength range
            return spectrum

        ix = <int>(point.x / self._dx)  # X-index of grid cell, in which the point is located
        iy = <int>(point.y / self._dy)  # Y-index of grid cell, in which the point is located
        iz = <int>(point.z / self._dz)  # Z-index of grid cell, in which the point is located
        self.add_to_memoryview(spectrum.samples_mv, ix, iy, iz, 1.0)

        return spectrum
