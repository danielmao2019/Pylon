from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os.path as osp

_ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

setup(
    name='pointnet2_ops',
    ext_modules=[
        CUDAExtension(
            name='_ext',
            sources=_ext_sources,
            include_dirs=[osp.join(_ext_src_root, "include")],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-Xfatbin', '-compress-all']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)