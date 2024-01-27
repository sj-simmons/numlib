from setuptools import setup, find_packages

from mypyc.build import mypycify

def readme():
  with open('README.rst', 'r') as fh:
    return fh.read()

setup(
  #entry_points={
  #    'console_scripts': [
  #        'bernoulli = polylib.bernoulli:cli',
  #    ],
  #    'gui_scripts': [],
  #},
  name='numlib',
  url='https://github.com/sj-simmons/numlib',
  download_url='https://github.com/sj-simmons/numlib/archive/v0.3.tar.gz',
  author='Scott Simmons',
  author_email='ssimmons@drury.edu',
  packages=find_packages(),
  #ext_modules=mypycify([
  #    'numlib/__init__.py',
  #    'numlib/modular_ints.py',
  #])
  python_requires='>=3.6',
  install_requires=[],
  version="0.3",
  license='Apache 2.0',
  description='basic number theory tools',
  long_description=readme(),
  include_package_data=True,
  zip_safe=False,
  project_urls={'Upstream Repository': 'https://gihub.com/sj-simmons/numlib'},
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'Intended Audience :: Education',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      'Programming Language :: Python :: 3.9',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Mathematics',
      'License :: OSI Approved :: Apache Software License'
  ]
)
