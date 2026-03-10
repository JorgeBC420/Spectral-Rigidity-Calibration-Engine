from setuptools import setup, find_packages

setup(
    name="spectral-rigidity-calibration-engine",
    version="2.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0,<2.0.0",
        "mpmath>=1.3.0",
        "numba>=0.58.0",
        "streamlit>=1.31.0",
        "plotly>=5.18.0",
        "matplotlib>=3.8.0",
        "pandas>=2.1.0",
        "psutil>=5.9.0",
        "h5py>=3.10.0",
        "pytest>=7.4.0",
    ],
    python_requires=">=3.9",
)
