import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.readlines()

setuptools.setup(
    name="audiodsp",
    version="0.0.1",
    author="Bramantyo Supriyatno",
    url="httpsL//github.com/bramantyois/audio_dsp",
    install_requires=requirements,
    description="Personal audio DSP library",
    long_description=long_description,

    packages=setuptools.find_packages(),
)

