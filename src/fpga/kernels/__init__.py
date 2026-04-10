"""FPGA kernel helpers and simulator-facing numeric blocks."""

from .dot import DotProductResult, simulate_dot_product, software_dot_product
from .gelu import (
    GELU_LUT_Q8_8,
    GELU_SATURATION_POINT_Q8_8,
    GeluComparison,
    gelu_pwl_q8_8_block,
    gelu_pwl_q8_8_scalar,
    gelu_tanh_reference,
    simulate_gelu_block,
)
from .gemm import (
    GemmComparison,
    MatrixI16,
    MatrixI64,
    simulate_gemm_tile,
    simulate_gemm_tile_with_accumulator,
    software_gemm,
    software_gemm_with_accumulator,
)
from .linear import (
    LinearComparison,
    LinearLayerI16,
    format_vector_i64,
    simulate_linear,
    software_linear,
)
from .logmel import (
    FFT_BINS,
    LOG_OUTPUT_FRAC_BITS,
    MEL_BINS,
    MEL_COEFF_FRAC_BITS,
    LogMelFrameComparison,
    build_mel_filterbank,
    quantize_mel_filterbank,
    simulate_logmel_frame,
    software_logmel_frame,
)

__all__ = [
    "DotProductResult",
    "GELU_LUT_Q8_8",
    "GELU_SATURATION_POINT_Q8_8",
    "GemmComparison",
    "GeluComparison",
    "LinearComparison",
    "LinearLayerI16",
    "LogMelFrameComparison",
    "LOG_OUTPUT_FRAC_BITS",
    "MatrixI16",
    "MatrixI64",
    "FFT_BINS",
    "MEL_BINS",
    "MEL_COEFF_FRAC_BITS",
    "build_mel_filterbank",
    "format_vector_i64",
    "gelu_pwl_q8_8_block",
    "gelu_pwl_q8_8_scalar",
    "gelu_tanh_reference",
    "quantize_mel_filterbank",
    "simulate_dot_product",
    "simulate_gelu_block",
    "simulate_gemm_tile",
    "simulate_gemm_tile_with_accumulator",
    "simulate_linear",
    "simulate_logmel_frame",
    "software_dot_product",
    "software_gemm",
    "software_gemm_with_accumulator",
    "software_linear",
    "software_logmel_frame",
]
