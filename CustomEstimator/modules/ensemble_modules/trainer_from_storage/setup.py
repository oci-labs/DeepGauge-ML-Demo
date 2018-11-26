from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['pandas>=0.23.4',
                     'multiprocessing']

setup(
    name='deepGauge_custom_estimator',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description=''
)