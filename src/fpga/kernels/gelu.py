from __future__ import annotations

import math
from dataclasses import dataclass

from ..quant import FixedPointConfig
from ..transport import FpgaExecutor, GeluBlockRequest

GELU_LUT_Q8_8: tuple[int, ...] = (
    0,
    38,
    89,
    148,
    215,
    286,
    358,
    430,
    500,
    569,
    636,
    702,
    767,
)
GELU_SATURATION_POINT_Q8_8 = 3 * 256


@dataclass(slots=True)
class GeluComparison:
    input_quantized: list[int]
    software_output: list[int]
    rtl_output: list[int]
    float_reference: list[float]
    software_dequantized: list[float]
    rtl_dequantized: list[float]
    max_abs_error: float
    matched: bool
    notes: list[str]


def gelu_tanh_reference(value: float) -> float:
    cubic = value * value * value
    inner = 0.797_884_6 * (value + 0.044_715 * cubic)
    return 0.5 * value * (1.0 + math.tanh(inner))


def gelu_pwl_nonnegative_q8_8(value: int) -> int:
    if value >= GELU_SATURATION_POINT_Q8_8:
        return value

    value_u16 = value & 0xFFFF
    index = value_u16 >> 6
    frac = value_u16 & 0x3F
    y0 = GELU_LUT_Q8_8[index]
    y1 = GELU_LUT_Q8_8[index + 1]
    delta = y1 - y0
    interpolated = y0 + ((delta * frac) >> 6)
    return max(min(interpolated, 32767), -32768)


def gelu_pwl_q8_8_scalar(value: int) -> int:
    if value == -32768:
        magnitude = 32767
    elif value < 0:
        magnitude = -value
    else:
        magnitude = value

    positive_output = gelu_pwl_nonnegative_q8_8(magnitude)
    if value < 0:
        result = positive_output - magnitude
        return max(min(result, 32767), -32768)
    return positive_output


def gelu_pwl_q8_8_block(values: list[int] | tuple[int, ...]) -> list[int]:
    return [gelu_pwl_q8_8_scalar(value) for value in values]


def simulate_gelu_block(
    executor: FpgaExecutor,
    output_dir,
    audio_path: str,
    input_block: list[int] | tuple[int, ...],
    float_input: list[float] | tuple[float, ...],
    quant: FixedPointConfig,
) -> GeluComparison:
    input_values = [int(value) for value in input_block]
    float_values = [float(value) for value in float_input]

    if len(input_values) != 8:
        raise ValueError("gelu simulator expects exactly 8 lanes")
    if len(input_values) != len(float_values):
        raise ValueError(
            f"gelu float/quantized input length mismatch: {len(input_values)} vs {len(float_values)}"
        )

    software_output = gelu_pwl_q8_8_block(input_values)
    response = executor.execute_gelu_block(
        GeluBlockRequest(
            audio_path=audio_path,
            input_block=input_values,
            expected_output=software_output,
        ),
        output_dir,
    )

    float_reference = [gelu_tanh_reference(value) for value in float_values]
    software_dequantized = [quant.dequantize_scalar(value) for value in software_output]
    rtl_dequantized = [quant.dequantize_scalar(value) for value in response.rtl_output]
    max_abs_error = max(
        (
            abs(expected - actual)
            for expected, actual in zip(float_reference, rtl_dequantized, strict=False)
        ),
        default=0.0,
    )

    return GeluComparison(
        input_quantized=input_values,
        software_output=software_output,
        rtl_output=list(response.rtl_output),
        float_reference=float_reference,
        software_dequantized=software_dequantized,
        rtl_dequantized=rtl_dequantized,
        max_abs_error=max_abs_error,
        matched=response.matched,
        notes=list(response.notes),
    )
