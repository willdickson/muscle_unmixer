from setuptools import setup

setup(
        name='muscle_unmixer', 
        version='0.1',
        description="Flyranch's muscle unmixer",
        author='Thad Lindsay',
        author_email='wbd@caltech.edu',
        license='MIT',
        packages=['muscle_unmixer'],
        zip_safe=False,
        include_package_data=True,
        install_requires = [
            'h5py',
            'numpy',
            'scipy',
            'pyqtgraph'
            ],
        entry_points={
            'console_scripts': ['muscle-unmixer=muscle_unmixer.viewer:main'],
            },
        )

