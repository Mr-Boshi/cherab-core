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
Integration utilities for regular grid emitters.
"""

from scipy.sparse import csc_matrix
from raysect.core.math.cython cimport find_index
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef object integrate_contineous(double[::1] x, object y, double x0, double x1, bint extrapolate=True):
    """
    Integrate the csc_matrix over its column axis in the specified range.

    :param double[::1] x: A memory view to a double array containing monotonically increasing
        values.
    :param csc_matrix y: A csc_matrix to integrate with the columns corresponding to
        the x array points.
    :param double x0: Start point of integration.
    :param double x1: End point of integration.
    :param bool extrapolate: If True, the values of y outside the provided range
        will be equal to the values at the borders of this range (nearest-neighbour
        extrapolation), otherwise it will be zero. Defaults to `extrapolate=True`.

        :return: Integrated csc_matrix (one-column csc_matrix).
    """

    cdef:
        object integral_sum
        double weight
        int index, lower_index, upper_index, top_index, nvoxel

    nvoxel = y.shape[0]

    # invalid range
    if x1 <= x0:
        return csc_matrix((nvoxel, 1))

    # identify array indices that lie between requested values
    lower_index = find_index(x, x0) + 1
    upper_index = find_index(x, x1)

    # are both points below the bottom of the array?
    if upper_index == -1:

        if extrapolate:
            # extrapolate from first array value (nearest-neighbour)
            return y[:, 0] * (x1 - x0)
        # return zero matrix if extrapolate is set to False
        return csc_matrix((nvoxel, 1))

    # are both points beyond the top of the array?
    top_index = x.shape[0] - 1
    if lower_index > top_index:

        if extrapolate:
            # extrapolate from last array value (nearest-neighbour)
            return y[:, top_index] * (x1 - x0)
        # return zero matrix if extrapolate is set to False
        return csc_matrix((nvoxel, 1))

    # numerically integrate array
    if lower_index > upper_index:

        # both values lie inside the same array segment
        # the names lower_index and upper_index are now misnomers, they are swapped!
        weight = (0.5 * (x1 + x0) - x[upper_index]) / (x[lower_index] - x[upper_index])

        # trapezium rule integration
        return (y[:, upper_index] + weight * (y[:, lower_index] - y[:, upper_index])) * (x1 - x0)

    else:

        integral_sum = csc_matrix((nvoxel, 1))

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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef object integrate_delta_function(double[::1] x, object y, double x0, double x1):
    """
    Integrate delta-function-like csc_matrix over its column axis in the specified range.

    :param double[::1] x: A memory view to a double array containing monotonically increasing
        values.
    :param csc_matrix y: A delta-function-like csc_matrix to integrate with the columns
        corresponding to the x array points.
    :param double x0: Start point of integration.
    :param double x1: End point of integration.

    :return: Integrated csc_matrix (one-column csc_matrix).
    """
    cdef:
        object integral_sum
        int i, nvoxel, nspec

    nvoxel = y.shape[0]
    nspec = y.shape[1]

    # invalid range
    if x1 <= x0:
        return csc_matrix((nvoxel, 1))

    integral_sum = csc_matrix((nvoxel, 1))

    for i in range(nspec):
        if x0 <= x[i] < x1:
            integral_sum += y[:, i]

    return integral_sum
