# Vendored KNN_CUDA 0.2

Upstream `github.com/unlimblue/KNN_CUDA` and its `0.2` release wheel both now return 404, and no fork survives, so this is the package source preserved verbatim from the last working install. It is consumed by the BUFFER point-cloud registration model (`models/point_cloud_registration/buffer/`). The CUDA kernel is JIT-compiled from `knn_cuda/csrc/cuda/` at import. This vendored copy was verified to produce KNN indices and distances bit-identical to the original 0.2 package.
