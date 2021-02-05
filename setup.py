from setuptools import setup, find_packages

setup(name='BboxToolkit',
      version='1.0',
      description='a tiny toolkit for bounding boxes',
      author='jbwang',
      packages=find_packages(),
      install_requires=[
          'opencv-python',
          'terminaltables',
          'pillow',
          'shapely',
          'numpy'])
