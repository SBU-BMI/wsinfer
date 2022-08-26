from setuptools import setup, find_packages

import versioneer


setup(
    name="wsi_inference",
    packages=find_packages(),
    python_requires=">=3.6, <4",
    install_requires=[
        "h5py",
        "large_image[sources]>=>=1.8.0",
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
            "wsi_create_patches=wsi_inference.patchlib.create_patches_fp:cli",
            "wsi_model_inference=wsi_inference.modellib.run_inference:cli",
            "wsi_run=wsi_inference.main:cli",
            "wsi_convert_to_geojson=wsi_inference.convert_csv_to_geojson:cli",
        ],
    },
    # This is for the tissue masking presets (in patchlib).
    package_data={"wsi_inference": ["*.csv"]},
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
