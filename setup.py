from setuptools import setup, find_packages


setup(
    name="wsi_inference",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.6, <4",
    install_requires=[
        "h5py",
        "large_image[sources]>=>=1.8.0",
        "matplotlib",
        "numpy",
        "opencv-python-headless>=4.0.0",
        "pandas",
        "pillow",
        "pretrainedmodels",
        "tqdm",
    ],
    extras_require=dict(dev=["black", "flake8", "mypy"]),
    entry_points={
        "console_scripts": [
            "wsi_create_patches=wsi_inference.pathlib.create_patches_fp:cli",
        ],
    },
)
