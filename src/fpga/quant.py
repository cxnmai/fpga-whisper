from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FixedPointConfig:
    bits: int
    fractional_bits: int

    @classmethod
    def q8_8(cls) -> "FixedPointConfig":
        return cls(bits=16, fractional_bits=8)

    @property
    def integer_bits(self) -> int:
        return self.bits - self.fractional_bits

    @property
    def min_scalar(self) -> int:
        return -(1 << (self.bits - 1))

    @property
    def max_scalar(self) -> int:
        return (1 << (self.bits - 1)) - 1

    def description(self) -> str:
        return f"Q{self.integer_bits}.{self.fractional_bits}"

    def scale_factor(self) -> float:
        return float(1 << self.fractional_bits)

    def integer_scale_factor(self) -> int:
        return 1 << self.fractional_bits

    def quantize_array(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        scaled = np.rint(array * self.integer_scale_factor())
        clipped = np.clip(scaled, self.min_scalar, self.max_scalar)
        return clipped.astype(np.int16, copy=False)

    def quantize_scalar(self, value: float) -> int:
        scaled = round(value * self.integer_scale_factor())
        return max(self.min_scalar, min(self.max_scalar, int(scaled)))

    def quantize_slice(self, values: list[float] | tuple[float, ...]) -> list[int]:
        return self.quantize_array(values).astype(np.int64, copy=False).tolist()

    def dequantize_scalar(self, value: int) -> float:
        return float(value) / self.integer_scale_factor()

    def dequantize_scalar_array(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return array / self.integer_scale_factor()

    def bias_to_accumulator(self, value: int) -> int:
        return int(value) << self.fractional_bits

    def dequantize_accumulator(self, value: int) -> float:
        scale = 1 << (self.fractional_bits * 2)
        return float(value) / float(scale)

    def dequantize_accumulator_array(self, values) -> np.ndarray:
        scale = float(1 << (self.fractional_bits * 2))
        array = np.asarray(values, dtype=np.float32)
        return array / scale

    def requantize_accumulator_to_scalar(self, value: int) -> int:
        rounding = 1 << (self.fractional_bits - 1)
        if value >= 0:
            shifted = (value + rounding) >> self.fractional_bits
        else:
            shifted = (value - rounding) >> self.fractional_bits
        return max(self.min_scalar, min(self.max_scalar, shifted))

    def requantize_accumulator_slice(
        self, values: list[int] | tuple[int, ...]
    ) -> list[int]:
        values_array = np.asarray(values, dtype=np.int64)
        rounding = 1 << (self.fractional_bits - 1)
        adjusted = np.where(
            values_array >= 0,
            values_array + rounding,
            values_array - rounding,
        )
        shifted = adjusted >> self.fractional_bits
        clipped = np.clip(shifted, self.min_scalar, self.max_scalar)
        return clipped.astype(np.int16, copy=False).astype(np.int64, copy=False).tolist()


Q8_8 = FixedPointConfig.q8_8()
