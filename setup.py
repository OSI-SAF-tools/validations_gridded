#!/usr/bin/env python

from distutils.core import setup

setup(name='validations_gridded',
      version='0.1',
      description='Validation For Gridded Fields',
      author='John Lavelle',
      author_email='jol@dmi.dk',
      url='https://github.com/OSI-SAF-tools/validations_gridded',
      packages=['scipy', 'pandas', 'xarray', 'dask', 'docopt', 'numpy', 'scikit_image', 'skimage', 'PyYAML',
                'bottleneck'],
     )

