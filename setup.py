from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="sigway",
    version="0.1",
    packages=find_packages(),
    description="A package for computing scalar induced gravitational waves "
    "from inflation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jonaselgammal/SIGWAY",
    author="Jonas El Gammal",
    author_email="jonas.el.gammal@rwth-aachen.de",
    license="LGPL",
    install_requires=required_packages,
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: GNU Lesser General Public License"
        " v3 (LGPLv3)",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
