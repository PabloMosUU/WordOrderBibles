from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

version = {}
with open(path.join(here, 'lstmlm', '__version__.py')) as fp:
    exec(fp.read(), version)

with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    name='lstmlm',

    version=version['__version__'],

    description='LstmLM',
    long_description=readme,
    long_description_content_type="text/markdown",

    # The project's main homepage.
    url='https://github.com/PabloMosUU/WordOrderBibles/',

    # Author details
    author='Pablo Mosteiro',
    author_email='p.mosteiro@uu.nl',

    packages=['lstmlm'],

    # Data files
    package_data={'lstmlm': ['data/*']},

    # Choose your license
    license='GNU LGPLv3',

    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    # What does your project relate to?
    keywords='de-identification',

    install_requires=['nltk'],
)
