from setuptools import find_packages, setup

setup(
    name="KNN_CUDA",
    version="0.2",
    packages=find_packages(),
    package_data={"knn_cuda": ["csrc/cuda/*.cpp", "csrc/cuda/*.cu"]},
    include_package_data=True,
)
