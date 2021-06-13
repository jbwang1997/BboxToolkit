from setuptools import setup, find_packages

long_descriptions = '''


setup(name='BboxToolkit',
      version='2.0',
      description='a tiny toolkit for spcial bounding boxes',
      author='jbwang',
      packages=find_packages(),
      install_requires=[
          'opencv-python',
          'terminaltables',
          'pillow',
          'shapely',
          'numpy'])
