from __future__ import annotations

from dataclasses import dataclass

from ..quant import FixedPointConfig
from ..transport import FpgaExecutor
from .gemm import (
    GemmComparison,
    MatrixI16,
    MatrixI64,
    simulate_gemm_tile_with_accumulator,
    software_gemm_with_accumulator,
)


@dataclass(slots=True)
class LinearLayerI16:
    input_dim: int
    output_dim: int
    weights: MatrixI16
    bias: list[int]
    quant: FixedPointConfig

    def validate(self) -> None:
        if self.weights.rows != self.input_dim or self.weights.cols != self.output_dim:
            raise ValueError(
                "weight matrix shape mismatch: "
                f"expected {self.input_dim}x{self.output_dim}, "
                f"got {self.weights.rows}x{self.weights.cols}"
            )
        if len(self.bias) != self.output_dim:
            raise ValueError(
                f"bias length mismatch: expected {self.output_dim}, got {len(self.bias)}"
            )


@dataclass(slots=True)
class LinearComparison:
    gemm: GemmComparison
    software_output: list[int]
    rtl_output: list[int]
    matched: bool
    notes: list[str]


def software_linear(layer: LinearLayerI16, input_values: list[int]) -> list[int]:
    layer.validate()
    if len(input_values) != layer.input_dim:
        raise ValueError(
            f"input length mismatch: expected {layer.input_dim}, got {len(input_values)}"
        )

    input_matrix = MatrixI16(rows=1, cols=layer.input_dim, values=list(input_values))
    bias_accumulator = MatrixI64(
        rows=1,
        cols=layer.output_dim,
        values=[
            layer.quant.bias_to_accumulator(bias_value) for bias_value in layer.bias
        ],
    )
    gemm = software_gemm_with_accumulator(
        input_matrix,
        layer.weights,
        bias_accumulator,
    )
    return gemm.values


def simulate_linear(
    executor: FpgaExecutor,
    output_dir,
    audio_path: str,
    layer: LinearLayerI16,
    input_values: list[int],
) -> LinearComparison:
    layer.validate()
    if len(input_values) != layer.input_dim:
        raise ValueError(
            f"input length mismatch: expected {layer.input_dim}, got {len(input_values)}"
        )

    input_matrix = MatrixI16(rows=1, cols=layer.input_dim, values=list(input_values))
    bias_accumulator = MatrixI64(
        rows=1,
        cols=layer.output_dim,
        values=[
            layer.quant.bias_to_accumulator(bias_value) for bias_value in layer.bias
        ],
    )

    gemm = simulate_gemm_tile_with_accumulator(
        executor=executor,
        output_dir=output_dir,
        audio_path=audio_path,
        lhs=input_matrix,
        rhs=layer.weights,
        accumulator=bias_accumulator,
    )
    software_output = list(gemm.software.values)
    rtl_output = list(gemm.rtl.values)
    matched = gemm.matched and software_output == rtl_output

    notes = [
        f"Linear layer quantization contract: {layer.quant.description()}",
        f"Bias values: {layer.bias!r}",
        *gemm.notes,
    ]

    return LinearComparison(
        gemm=gemm,
        software_output=software_output,
        rtl_output=rtl_output,
        matched=matched,
        notes=notes,
    )


def format_vector_i64(values: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"
