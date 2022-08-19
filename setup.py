from setuptools import find_packages, setup

setup(
    name="",
    packages=find_packages(),
    python_requires=">=3.7, <4",
    install_requires=[
        "large_image>=>=1.8.0",
        "large-image-source-openslide>=1.8.0",
        "large-image-source-tiff>=1.8.0",
        "numpy",
        "pillow",
        "tqdm",
        # CLAM dependencies (for patching)
        "h5py",
        "matplotlib",
        "opencv-python-headless>=4.0.0",
        "pandas",
    ],
)
