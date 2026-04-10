from __future__ import annotations

from dataclasses import dataclass


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

    def quantize_scalar(self, value: float) -> int:
        scaled = round(value * self.scale_factor())
        return max(self.min_scalar, min(self.max_scalar, int(scaled)))

    def quantize_slice(self, values: list[float] | tuple[float, ...]) -> list[int]:
        return [self.quantize_scalar(value) for value in values]

    def dequantize_scalar(self, value: int) -> float:
        return float(value) / self.scale_factor()

    def bias_to_accumulator(self, value: int) -> int:
        return int(value) << self.fractional_bits

    def dequantize_accumulator(self, value: int) -> float:
        scale = 1 << (self.fractional_bits * 2)
        return float(value) / float(scale)

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
        return [self.requantize_accumulator_to_scalar(value) for value in values]


Q8_8 = FixedPointConfig.q8_8()
