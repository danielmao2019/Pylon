from setuptools import setup, find_packages

setup(
    name="pylon",
    description="A deep learning framework for 3D vision tasks",
    author="Daniel",
    packages=find_packages(include=["agents*", "criteria*", "data*", "metrics*", "models*", 
                                  "optimizers*", "project*", "runners*", "schedulers*", "utils*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pytest>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
    },
    use_scm_version={
        "write_to": "pylon/_version.py",
        "write_to_template": '__version__ = "{version}"',
    },
)
