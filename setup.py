from setuptools import setup, find_packages

setup(
    name='logicML',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'h2o',
        'scikit-learn',
        'matplotlib',
        'numpy',
        'pandas',
        'seaborn',
    ],
)
