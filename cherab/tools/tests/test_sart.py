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

import unittest
import os
import numpy as np
from cherab.tools.inversions import invert_sart, invert_constrained_sart, invert_saart, invert_constrained_saart


class TestSART(unittest.TestCase):
    """
    Test cases for SART and SAART solvers.
    """

    def setUp(self):
        # geometry matrix in float64, shape: (npixel_x, npixel_y, nsource)
        gm = np.load(os.path.join(os.path.dirname(__file__), 'data/geometry_matrix.npy'))
        self.gm = gm.reshape((gm.shape[0] * gm.shape[1], gm.shape[2])).astype(np.float64)
        # receiver in float64, shape: (npixel_x, npixel_y)
        receiver = np.load(os.path.join(os.path.dirname(__file__), 'data/receiver.npy'))
        self.receiver = receiver.flatten().astype(np.float64)
        # true emissivity in float64, shape: (11, 8)
        true_emissivity = np.load(os.path.join(os.path.dirname(__file__), 'data/true_emissivity.npy'))
        self.true_emissivity = true_emissivity.flatten().astype(np.float64)

    def test_inversion(self):
        solution, residual = invert_sart(self.gm, self.receiver)
        self.assertTrue(np.allclose(solution, self.true_emissivity, atol=1.e-2))

    def test_adaptive_inversion(self):
        solution, residual = invert_saart(self.gm, self.receiver, conv_tol=1.e-6)
        self.assertTrue(np.allclose(solution, self.true_emissivity, atol=1.e-2))

    def test_inversion_constrained(self):
        # The emission profile is a sharp function here, so in this test the regularisation leads to inaccurate results.
        # The beta_laplace parameter is set to just 0.001 to reduce the impact of regularisation. This is a technical test only.
        laplacian_matrix = np.identity(self.gm.shape[1])
        solution, residual = invert_constrained_sart(self.gm, laplacian_matrix, self.receiver, beta_laplace=0.001)
        self.assertTrue(np.allclose(solution / solution.max(), self.true_emissivity, atol=1.e-2))

    def test_adaptive_inversion_constrained(self):
        laplacian_matrix = np.identity(self.gm.shape[1])
        solution, residual = invert_constrained_saart(self.gm, laplacian_matrix, self.receiver, beta_laplace=0.001, conv_tol=1.e-6)
        self.assertTrue(np.allclose(solution / solution.max(), self.true_emissivity, atol=1.e-2))
