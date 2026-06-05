// ROCm-only op. Written in the post-hipify form (hipBLASLt symbols) because
// the bundled hipify_torch v1 mapping table does not translate cublasLt*.

#ifdef USE_ROCM

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/core/ScalarType.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hipblaslt/hipblaslt.h>

namespace mslk::gemm::hipblas {

namespace {

inline void checkHipblasStatus(hipblasStatus_t status) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    printf("hipBLASLt API failed with status %d\n", status);
    throw std::logic_error("hipBLASLt API failed");
  }
}

} // namespace

at::Tensor f8f8bf16(
    at::Tensor XQ, // FP8 e4m3, row-major [..., K]
    at::Tensor WQ, // FP8 e4m3, row-major [N, K]
    at::Tensor scale, // FP32 scalar on device, applied to D
    bool /* use_fast_accum */) {
  // Why: schema parity with CUDA f8f8bf16; hipBLASLt has no fast-accum toggle.
  c10::hip::HIPGuardMasqueradingAsCUDA deviceGuard(XQ.device());

  int M = at::size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int K = XQ.size(-1);
  int N = at::size_to_dim_(WQ.dim() - 1, WQ.sizes());

  auto out_sizes = XQ.sizes().vec();
  out_sizes.back() = N;

  if (M == 0 || N == 0 || K == 0) {
    return at::zeros(out_sizes, XQ.options().dtype(at::kBFloat16));
  }

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(WQ.size(-1) == K);
  TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == at::kFloat);

  auto Y = at::empty(out_sizes, XQ.options().dtype(at::kBFloat16));

  // MI300 / gfx94x uses the FNUZ FP8 encoding natively.
  const hipDataType kFp8Type = HIP_R_8F_E4M3_FNUZ;

  // Row-major in PyTorch == column-major-transposed in BLAS. Compute
  // D^T[N x M] = W[N,K] @ X^T[K,M] by transposing A=W and not transposing B=X.
  hipblasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
  hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

  hipblasLtMatmulDesc_t opDesc = nullptr;
  hipblasLtMatrixLayout_t aLayout = nullptr; // W
  hipblasLtMatrixLayout_t bLayout = nullptr; // X
  hipblasLtMatrixLayout_t dLayout = nullptr; // Y
  hipblasLtMatmulPreference_t pref = nullptr;

  checkHipblasStatus(
      hipblasLtMatmulDescCreate(&opDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

  hipblasOperation_t opT = HIPBLAS_OP_T;
  hipblasOperation_t opN = HIPBLAS_OP_N;
  checkHipblasStatus(hipblasLtMatmulDescSetAttribute(
      opDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
  checkHipblasStatus(hipblasLtMatmulDescSetAttribute(
      opDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

  // Read alpha from device memory (per-tensor scale lives there). Pairs with
  // POINTER_MODE_DEVICE below. Mirrors the CUTLASS f8f8bf16.cu epilogue
  // which passes scale.data_ptr<float>() with stride 0 (scalar broadcast).
  hipblasLtPointerMode_t pmode = HIPBLASLT_POINTER_MODE_DEVICE;
  checkHipblasStatus(hipblasLtMatmulDescSetAttribute(
      opDesc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pmode, sizeof(pmode)));

  // A = W as [K, N] col-major (W row-major [N, K])  opT transposes to [N, K].
  checkHipblasStatus(hipblasLtMatrixLayoutCreate(&aLayout, kFp8Type, K, N, K));
  // B = X as [K, M] col-major (X row-major [M, K]).
  checkHipblasStatus(hipblasLtMatrixLayoutCreate(&bLayout, kFp8Type, K, M, K));
  // D = Y as [N, M] col-major (Y row-major [M, N]).
  checkHipblasStatus(hipblasLtMatrixLayoutCreate(&dLayout, HIP_R_16BF, N, M, N));

  TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.is_cuda(),
              "scale must be a single FP32 tensor on device");
  // hipBLASLt's POINTER_MODE_DEVICE reads alpha as an N-length vector indexed
  // by output column (no scalar-broadcast mode like CUTLASS's {ptr, stride=0}
  // exists in this version). Broadcast the per-tensor scale to a length-N
  // device buffer ourselves. Stays on stream — graph-capturable.
  at::Tensor alpha_buf = scale.expand({N}).contiguous();
  // Beta also lives on device under POINTER_MODE_DEVICE; persistent zero.
  static thread_local at::Tensor beta_dev =
      at::zeros({1}, at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
  const void* alpha_ptr = alpha_buf.data_ptr<float>();
  const void* beta_ptr = beta_dev.data_ptr<float>();

  checkHipblasStatus(hipblasLtMatmulPreferenceCreate(&pref));
  constexpr uint64_t kWorkspaceBytes = 32ull * 1024 * 1024; // 32 MiB
  checkHipblasStatus(hipblasLtMatmulPreferenceSetAttribute(
      pref,
      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &kWorkspaceBytes,
      sizeof(kWorkspaceBytes)));

  hipblasLtMatmulHeuristicResult_t heuristic{};
  int returnedAlgoCount = 0;
  checkHipblasStatus(hipblasLtMatmulAlgoGetHeuristic(
      handle,
      opDesc,
      aLayout,
      bLayout,
      dLayout, // C (unused since beta=0)
      dLayout, // D
      pref,
      1,
      &heuristic,
      &returnedAlgoCount));
  TORCH_CHECK(
      returnedAlgoCount > 0,
      "hipBLASLt returned no algorithm for f8f8bf16 (M=",
      M,
      ", N=",
      N,
      ", K=",
      K,
      ")");

  auto workspace = at::empty(
      {static_cast<int64_t>(heuristic.workspaceSize)},
      XQ.options().dtype(at::kByte));

  checkHipblasStatus(hipblasLtMatmul(
      handle,
      opDesc,
      alpha_ptr,
      WQ.data_ptr(), // A
      aLayout,
      XQ.data_ptr(), // B
      bLayout,
      beta_ptr,
      Y.data_ptr(), // C (unused at beta=0)
      dLayout,
      Y.data_ptr(), // D
      dLayout,
      &heuristic.algo,
      workspace.data_ptr(),
      heuristic.workspaceSize,
      stream));

  hipblasLtMatmulPreferenceDestroy(pref);
  hipblasLtMatrixLayoutDestroy(dLayout);
  hipblasLtMatrixLayoutDestroy(bLayout);
  hipblasLtMatrixLayoutDestroy(aLayout);
  hipblasLtMatmulDescDestroy(opDesc);

  return Y;
}

} // namespace mslk::gemm::hipblas

#endif // USE_ROCM
