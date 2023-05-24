from setuptools import setup

setup(
    name="ZipDAS",
    version="0.1.1",
    long_description="DAS Data Compression",
    long_description_content_type="text/markdown",
    packages=["zipdas"],
    install_requires=["numpy",  "h5py", "matplotlib", "pandas"],
)
