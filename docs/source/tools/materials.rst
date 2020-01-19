
Materials
=========

See the Raysect reference documentation for the full catalogue of materials and associated
API interfaces. In CHERAB we provide a few useful materials.


.. autoclass:: cherab.tools.emitters.radiation_function.RadiationFunction
   :show-inheritance:


Regular Grid Volumes
--------------------

Regular Grid Volumes accelerate integration through inhomogeneous emitting volume  
as they use pre-calculated values of spectral emissivity on a regular grid.

.. autoclass:: cherab.tools.emitters.regular_grid_volumes.RegularGridVolume

.. autoclass:: cherab.tools.emitters.regular_grid_volumes.RegularGridBox
   :show-inheritance:

.. autoclass:: cherab.tools.emitters.regular_grid_volumes.RegularGridCylinder
   :show-inheritance:

**Emitters and integrators**

The following emitters and integrators are used in Regular Grid Volumes.
Note that these emitters support other integrators as well, however high performance
with other integrators is not guaranteed.

.. autoclass:: cherab.tools.emitters.regular_grid_emitters.RegularGridEmitter
   :show-inheritance:

.. autoclass:: cherab.tools.emitters.regular_grid_emitters.RegularGridIntegrator
   :show-inheritance:

.. autoclass:: cherab.tools.emitters.regular_grid_emitters.CartesianRegularEmitter
   :show-inheritance:

.. autoclass:: cherab.tools.emitters.regular_grid_emitters.CartesianRegularIntegrator
   :show-inheritance:

.. autoclass:: cherab.tools.emitters.regular_grid_emitters.CylindricalRegularEmitter
   :show-inheritance:

.. autoclass:: cherab.tools.emitters.regular_grid_emitters.CylindricalRegularIntegrator
   :show-inheritance:
