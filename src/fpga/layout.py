from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TileShape:
    rows: int
    cols: int
    inner: int

    def validate(self, lhs_len: int, rhs_len: int) -> None:
        expected_lhs = self.rows * self.inner
        expected_rhs = self.inner * self.cols

        if lhs_len != expected_lhs:
            raise ValueError(
                f"lhs tile length mismatch: expected {expected_lhs}, got {lhs_len}"
            )
        if rhs_len != expected_rhs:
            raise ValueError(
                f"rhs tile length mismatch: expected {expected_rhs}, got {rhs_len}"
            )

    @property
    def lhs_size(self) -> int:
        return self.rows * self.inner

    @property
    def rhs_size(self) -> int:
        return self.inner * self.cols

    @property
    def output_size(self) -> int:
        return self.rows * self.cols
