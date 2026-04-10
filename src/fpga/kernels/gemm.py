from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..transport import FpgaExecutor, GemmTileI16Request, GemmTileShape


@dataclass(slots=True)
class MatrixI16:
    rows: int
    cols: int
    values: list[int]

    def row(self, index: int) -> list[int]:
        start = index * self.cols
        end = start + self.cols
        return self.values[start:end]


@dataclass(slots=True)
class MatrixI64:
    rows: int
    cols: int
    values: list[int]

    def get(self, row: int, col: int) -> int:
        return self.values[row * self.cols + col]


@dataclass(slots=True)
class GemmComparison:
    software: MatrixI64
    rtl: MatrixI64
    matched: bool
    notes: list[str]


def software_gemm(lhs: MatrixI16, rhs: MatrixI16) -> MatrixI64:
    return software_gemm_with_accumulator(lhs, rhs, None)


def software_gemm_with_accumulator(
    lhs: MatrixI16,
    rhs: MatrixI16,
    accumulator: MatrixI64 | None,
) -> MatrixI64:
    if lhs.cols != rhs.rows:
        raise ValueError(
            f"incompatible GEMM dimensions: lhs {lhs.rows}x{lhs.cols}, rhs {rhs.rows}x{rhs.cols}"
        )

    if accumulator is not None:
        if accumulator.rows != lhs.rows or accumulator.cols != rhs.cols:
            raise ValueError(
                "accumulator shape mismatch: "
                f"expected {lhs.rows}x{rhs.cols}, got {accumulator.rows}x{accumulator.cols}"
            )

    values: list[int] = []
    for row in range(lhs.rows):
        for col in range(rhs.cols):
            total = 0 if accumulator is None else accumulator.get(row, col)
            for inner in range(lhs.cols):
                lhs_value = int(lhs.values[row * lhs.cols + inner])
                rhs_value = int(rhs.values[inner * rhs.cols + col])
                total += lhs_value * rhs_value
            values.append(total)

    return MatrixI64(rows=lhs.rows, cols=rhs.cols, values=values)


def simulate_gemm_tile(
    executor: FpgaExecutor,
    output_dir: Path,
    audio_path: str,
    lhs: MatrixI16,
    rhs: MatrixI16,
) -> GemmComparison:
    return simulate_gemm_tile_with_accumulator(
        executor=executor,
        output_dir=output_dir,
        audio_path=audio_path,
        lhs=lhs,
        rhs=rhs,
        accumulator=None,
    )


def simulate_gemm_tile_with_accumulator(
    executor: FpgaExecutor,
    output_dir: Path,
    audio_path: str,
    lhs: MatrixI16,
    rhs: MatrixI16,
    accumulator: MatrixI64 | None,
) -> GemmComparison:
    if lhs.cols != rhs.rows:
        raise ValueError(
            f"incompatible GEMM dimensions: lhs {lhs.rows}x{lhs.cols}, rhs {rhs.rows}x{rhs.cols}"
        )

    software = software_gemm_with_accumulator(lhs, rhs, accumulator)

    response = executor.execute_gemm_tile(
        GemmTileI16Request(
            audio_path=audio_path,
            shape=GemmTileShape(rows=lhs.rows, cols=rhs.cols, inner=lhs.cols),
            lhs_tile=list(lhs.values),
            rhs_tile=list(rhs.values),
            accumulator_input=(
                [0 for _ in range(lhs.rows * rhs.cols)]
                if accumulator is None
                else list(accumulator.values)
            ),
            expected_output=list(software.values),
        ),
        output_dir,
    )

    rtl = MatrixI64(rows=lhs.rows, cols=rhs.cols, values=list(response.rtl_output))
    return GemmComparison(
        software=software,
        rtl=rtl,
        matched=bool(response.matched),
        notes=list(response.notes),
    )
