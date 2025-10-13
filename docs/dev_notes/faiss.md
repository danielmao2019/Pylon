Error:

```
Faiss assertion 'err == CUBLAS_STATUS_SUCCESS' failed in void faiss::gpu::runMatrixMult(faiss::gpu::Tensor<float, 2, true>&, bool, faiss::gpu::Tensor<T, 2, true>&, bool, faiss::gpu::Tensor<IndexType, 2, true>&, bool, float, float, cublasHandle_t, cudaStream_t) [with AT = float; BT = float; cublasHandle_t = cublasContext*; cudaStream_t = CUstream_st*] at /project/faiss/faiss/gpu/utils/MatrixMult-inl.cuh:265; details: cublas failed (13): (340, 3) x (131072, 3)' = (340, 131072) gemm params m 131072 n 340 k 3 trA T trB N lda 3 ldb 3 ldc 131072
```

Use case:
Multi-GPU
Installed faiss-gpu using pip.

Solution:
Uninstall faiss-gpu.
conda install -c pytorch faiss-gpu
