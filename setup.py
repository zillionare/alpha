#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'apscheduler==3.6.3',
    'arrow==0.15.5',
    'cfg4py>=0.6.0',
    'ruamel.yaml==0.16',
    'aioredis==1.3.1',
    'hiredis==1.0.1',
    'pyemit>=0.4.0',
    'numpy>=1.18.1',
    'numba==0.49.1',
    'aiohttp==3.7.4',
    'pytz==2020.1',
    'xxhash==1.4.3',
    'zillionare-omicron>=0.1.2',
    'psutil==5.7.0',
    'termcolor==1.1.0',
    'arrow==0.15.5',
    'aiohttp==3.7.4',
    'scikit-learn==0.23.1',
    'joblib==0.16.0',
    'zillionare-omicron>=0.1.2',
    'websocket-client==0.57.0', 'pandas'
]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Aaron Yang",
    author_email='code@jieyu.ai',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python Boilerplate contains all the boilerplate you need to create a Python package.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='alpha',
    name='alpha',
    packages=find_packages(include=['alpha', 'alpha.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/zillionare/alpha',
    version='0.1.0',
    zip_safe=False,
    entry_points={
        'console_scripts': ['alpha=alpha.cli:main']
    }
)
