# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

# cython: language_level=3

from numpy cimport ndarray
from cherab.core.atomic cimport FreeFreeGauntFactor
from cherab.core.plasma cimport PlasmaModel


cdef class Bremsstrahlung(PlasmaModel):

    cdef:
        FreeFreeGauntFactor _gaunt_factor
        bint _user_provided_gaunt_factor
        ndarray _species_charge, _species_density
        double[::1] _species_density_mv, _species_charge_mv

    cdef double _bremsstrahlung(self, double wvl, double te, double ne)

    cdef int _populate_cache(self) except -1


