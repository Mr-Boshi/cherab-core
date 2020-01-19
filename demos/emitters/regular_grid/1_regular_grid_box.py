
import os
import time
from matplotlib.pyplot import *
import numpy as np

from raysect.optical import World, translate, rotate, Point3D
from raysect.optical.library import RoughTitanium
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.primitive import Box
from cherab.tools.emitters import RegularGridBox

"""
RegularGridBox Demonstration
----------------------------

This file demonstrates how to use RegularGridBox to effectively integrate through
the emission profiles defined on a regular grid.

This demonstration uses exactly the same emission profile as the Raysect's
demos/material/volume.py demonstration. It is recomended to run the Raysect's
volume.py demo first for better understanding.

Notice tenfold speedup compared to the Raysect's volume.py demo achieved by
pre-calculating the emission profile on a regular grid.

Even higher speedup can be achieved for smaller integration steps. Reducing the
integration step to 0.01 in both demos (along with doubling the regular grid resolution
in this demo) results in 40x speedup.

"""

# grid parameters
xmin = ymin = -1.
xmax = ymax = 1.
zmin = -0.25
zmax = 0.25
x, dx = np.linspace(xmin, xmax, 101, retstep=True)
y, dy = np.linspace(ymin, ymax, 101, retstep=True)
z, dz = np.linspace(zmin, zmax, 26, retstep=True)
integration_step = 0.05
# x, dx = np.linspace(xmin, xmax, 201, retstep=True)
# y, dy = np.linspace(ymin, ymax, 201, retstep=True)
# z, dz = np.linspace(zmin, zmax, 51, retstep=True)
# integration_step = 0.01
x = x[:-1] + 0.5 * dx  # moving to the grid cell centers
y = y[:-1] + 0.5 * dy
z = z[:-1] + 0.5 * dz

# spectral emission profile
min_wavelength = 375.
max_wavelength = 740.
spectral_bins = 15
wavelengths, delta_wavelength = np.linspace(min_wavelength, max_wavelength, spectral_bins + 1, retstep=True)
wavelengths = wavelengths[:-1] + 0.5 * delta_wavelength
wvl_centre = 0.5 * (max_wavelength + min_wavelength)
wvl_range = min_wavelength - max_wavelength
shift = 2 * (wavelengths - wvl_centre) / wvl_range + 5.
radius = np.sqrt((x * x)[:, None] + (y * y)[None, :])
emission = np.cos(shift[None, None, None, :] * radius[:, :, None, None] * np.ones(z.size)[None, None, :, None])**4

# scene
world = World()
emitter = RegularGridBox(emission, min_wavelength, xmax=xmax - xmin, ymax=ymax - ymin, zmax=zmax - zmin,
                         step=integration_step, parent=world, transform=translate(xmin, ymin + 1., zmin) * rotate(30, 0, 0))
floor = Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=RoughTitanium(0.1))

# camera
rgb_pipeline = RGBPipeline2D(display_update_time=5)
sampler = RGBAdaptiveSampler2D(rgb_pipeline, min_samples=100, fraction=0.2)
camera = PinholeCamera((512, 512), parent=world, transform=translate(0, 4, -3.5) * rotate(0, -45, 0), pipelines=[rgb_pipeline], frame_sampler=sampler)
camera.min_wavelength = min_wavelength
camera.max_wavelength = max_wavelength
camera.spectral_bins = spectral_bins
camera.spectral_rays = 1
camera.pixel_samples = 200

# start ray tracing
os.nice(15)
ion()
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
for p in range(1, 1000):
    print("Rendering pass {}...".format(p))
    camera.observe()
    rgb_pipeline.save("demo_regular_grid_box_{}_pass_{}.png".format(timestamp, p))
    print()

# display final result
ioff()
rgb_pipeline.display()
show()
