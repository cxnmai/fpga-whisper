from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .layout import TileShape


@dataclass(slots=True)
class DotProductRequest:
    audio_path: str
    vector_a: list[int]
    vector_b: list[int]
    expected_result: int

    def to_dict(self) -> dict[str, object]:
        return {
            "audio_path": self.audio_path,
            "vector_a": list(self.vector_a),
            "vector_b": list(self.vector_b),
            "expected_result": self.expected_result,
        }


@dataclass(slots=True)
class DotProductResponse:
    rtl_result: int
    expected_result: int
    matched: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "rtl_result": self.rtl_result,
            "expected_result": self.expected_result,
            "matched": self.matched,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class GemmTileShape:
    rows: int
    cols: int
    inner: int

    def as_layout(self) -> TileShape:
        return TileShape(rows=self.rows, cols=self.cols, inner=self.inner)

    def to_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "inner": self.inner,
        }


@dataclass(slots=True)
class GemmTileI16Request:
    audio_path: str
    shape: GemmTileShape
    lhs_tile: list[int]
    rhs_tile: list[int]
    accumulator_input: list[int]
    expected_output: list[int]

    def validate(self) -> None:
        self.shape.as_layout().validate(len(self.lhs_tile), len(self.rhs_tile))
        expected_accumulator_len = self.shape.rows * self.shape.cols
        if len(self.accumulator_input) != expected_accumulator_len:
            raise ValueError(
                "accumulator tile length mismatch: "
                f"expected {expected_accumulator_len}, got {len(self.accumulator_input)}"
            )
        if len(self.expected_output) != expected_accumulator_len:
            raise ValueError(
                "expected output length mismatch: "
                f"expected {expected_accumulator_len}, got {len(self.expected_output)}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "audio_path": self.audio_path,
            "shape": self.shape.to_dict(),
            "lhs_tile": list(self.lhs_tile),
            "rhs_tile": list(self.rhs_tile),
            "accumulator_input": list(self.accumulator_input),
            "expected_output": list(self.expected_output),
        }


@dataclass(slots=True)
class GemmTileI64Response:
    shape: GemmTileShape
    rtl_output: list[int]
    expected_output: list[int]
    matched: bool
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        expected_len = self.shape.rows * self.shape.cols
        if len(self.rtl_output) != expected_len:
            raise ValueError(
                f"rtl output length mismatch: expected {expected_len}, got {len(self.rtl_output)}"
            )
        if len(self.expected_output) != expected_len:
            raise ValueError(
                "expected output length mismatch: "
                f"expected {expected_len}, got {len(self.expected_output)}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": self.shape.to_dict(),
            "rtl_output": list(self.rtl_output),
            "expected_output": list(self.expected_output),
            "matched": self.matched,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class GemmTileBatchI16Request:
    shape: GemmTileShape
    requests: list[GemmTileI16Request]

    def validate(self) -> None:
        for request in self.requests:
            request.validate()
            if request.shape != self.shape:
                raise ValueError(
                    "batch request shape mismatch: "
                    f"expected {self.shape.to_dict()}, got {request.shape.to_dict()}"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": self.shape.to_dict(),
            "case_count": len(self.requests),
            "requests": [request.to_dict() for request in self.requests],
        }


@dataclass(slots=True)
class GeluBlockRequest:
    audio_path: str
    input_block: list[int]
    expected_output: list[int]

    def validate(self) -> None:
        if len(self.input_block) != 8:
            raise ValueError("gelu simulator expects exactly 8 lanes")
        if len(self.expected_output) != 8:
            raise ValueError("gelu simulator expects exactly 8 expected outputs")

    def to_dict(self) -> dict[str, object]:
        return {
            "audio_path": self.audio_path,
            "input_block": list(self.input_block),
            "expected_output": list(self.expected_output),
        }


@dataclass(slots=True)
class GeluBlockResponse:
    rtl_output: list[int]
    expected_output: list[int]
    matched: bool
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if len(self.rtl_output) != 8:
            raise ValueError("gelu simulator expects exactly 8 RTL outputs")
        if len(self.expected_output) != 8:
            raise ValueError("gelu simulator expects exactly 8 expected outputs")

    def to_dict(self) -> dict[str, object]:
        return {
            "rtl_output": list(self.rtl_output),
            "expected_output": list(self.expected_output),
            "matched": self.matched,
            "notes": list(self.notes),
        }


class FpgaExecutor(Protocol):
    def name(self) -> str: ...

    def execute_dot_product(
        self,
        request: DotProductRequest,
        output_dir: Path,
    ) -> DotProductResponse: ...

    def execute_gemm_tile(
        self,
        request: GemmTileI16Request,
        output_dir: Path,
    ) -> GemmTileI64Response: ...

    def execute_gemm_tile_batch(
        self,
        request: GemmTileBatchI16Request,
        output_dir: Path,
    ) -> list[GemmTileI64Response]: ...

    def execute_gelu_block(
        self,
        request: GeluBlockRequest,
        output_dir: Path,
    ) -> GeluBlockResponse: ...
