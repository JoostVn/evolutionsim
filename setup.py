from setuptools import setup, find_packages


setup(
    name='evolution_learning',
    version='1',
    packages=find_packages(),
    description='Visual simulations of evolutionairy algorithms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',


   # author='Joost Venselaar',

    #author_email='your.email@example.com',
    #url='https://github.com/yourusername/my_project',
    #install_requires=[
        # List your project's dependencies here
        # e.g., 'numpy', 'pandas', etc.
    #],


    # If your project contains data files or non-python code, include them here
    include_package_data=False,
    package_data={
        'gui': ['resources/*.json']
    },

    #classifiers=[
    #    # Choose classifiers from https://pypi.org/classifiers/
    #    'Programming Language :: Python :: 3',
    #    'License :: OSI Approved :: MIT License',
    #    'Operating System :: OS Independent',
    #],
)


