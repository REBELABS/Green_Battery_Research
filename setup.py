import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diffKDE",
    version="1",
    author="Maria-Theresia Pelz, Thomas Slawig",
    author_email="mtv@informatik.uni-kiel.de , ts@informatik.uni-kiel.de",
    description="Diffusion-based kernel density estimator for the approximation of 1D probability density functions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
