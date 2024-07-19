from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='WavelengthCalibrationTool',
      version='0.1',
      description='Tool for doing wavelength calibration of arc spectrum',
      long_description = readme(),
      classifiers=[
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Programming Language :: Python :: 3.6+',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      keywords='Wavelength Calibration Spectrum Astronomy',
      url='https://github.com/indiajoe/WavelengthCalibrationTool',
      author='Joe Ninan',
      author_email='indiajoe@gmail.com',
      license='GPLv3+',
      packages=['WavelengthCalibrationTool'],
      entry_points = {
          'console_scripts': ['iidentify=WavelengthCalibrationTool.iidentify:main',
                              'reidentify=WavelengthCalibrationTool.reidentify:main',
                              'recalibrate=WavelengthCalibrationTool.recalibrate:main'],
      },
      install_requires = [
          'numpy',
          'matplotlib',
          'astropy',
      ],
      include_package_data=True,
      zip_safe=False)
