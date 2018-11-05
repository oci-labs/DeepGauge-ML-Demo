from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['argparse',
                     'tensorflow==1.10.0',
                     'glob',
                     'os',
                     'sklearn==1.19.2',
                     'numpy==1.14.5',
                     'pandas==0.23.4',
                     'MultiColProcessor',
                     'pickle',
                     'json',
                     'collections',
                     'multiprocessing']

setup(
    name='deepGauge_custom_estimator',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description=''
)