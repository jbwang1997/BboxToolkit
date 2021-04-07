from setuptools import setup, find_packages

setup(name='BboxToolkit',
      version='1.0',
      description='a tiny toolkit for special bounding boxes',
      author='jbwang1997',
      packages=find_packages(),
      install_requires=[
          'opencv-python',
          'terminaltables',
          'pillow',
          'shapely',
          'numpy'])
