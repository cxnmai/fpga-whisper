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

__all__ = [
    "DotProductResult",
    "GELU_LUT_Q8_8",
    "GELU_SATURATION_POINT_Q8_8",
    "GemmComparison",
    "GeluComparison",
    "LinearComparison",
    "LinearLayerI16",
    "MatrixI16",
    "MatrixI64",
    "format_vector_i64",
    "gelu_pwl_q8_8_block",
    "gelu_pwl_q8_8_scalar",
    "gelu_tanh_reference",
    "simulate_dot_product",
    "simulate_gelu_block",
    "simulate_gemm_tile",
    "simulate_gemm_tile_with_accumulator",
    "simulate_linear",
    "software_dot_product",
    "software_gemm",
    "software_gemm_with_accumulator",
    "software_linear",
]
