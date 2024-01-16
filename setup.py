from setuptools import setup, find_packages

setup(
    name='OSM2AT',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'osmnx',
        'shapely',
        'scikit-learn',
        'torch',
        'tqdm',
        'scipy',
        'gower',
        'kmedoids'
    ],
)