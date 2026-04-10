from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..transport import DotProductRequest, DotProductResponse


class FpgaExecutor(Protocol):
    def name(self) -> str: ...
    def execute_dot_product(
        self,
        request: DotProductRequest,
        output_dir: Path,
    ) -> DotProductResponse: ...


@dataclass(slots=True, frozen=True)
class DotProductResult:
    rtl_result: int
    expected_result: int
    matched: bool
    notes: list[str]


def software_dot_product(
    lhs: list[int] | tuple[int, ...], rhs: list[int] | tuple[int, ...]
) -> int:
    return sum(int(left) * int(right) for left, right in zip(lhs, rhs, strict=False))


def simulate_dot_product(
    executor: FpgaExecutor,
    output_dir: Path,
    audio_path: str,
    lhs: list[int] | tuple[int, ...],
    rhs: list[int] | tuple[int, ...],
) -> DotProductResult:
    expected_result = software_dot_product(lhs, rhs)
    response = executor.execute_dot_product(
        DotProductRequest(
            audio_path=audio_path,
            vector_a=[int(value) for value in lhs],
            vector_b=[int(value) for value in rhs],
            expected_result=expected_result,
        ),
        output_dir,
    )

    return DotProductResult(
        rtl_result=response.rtl_result,
        expected_result=response.expected_result,
        matched=response.matched,
        notes=list(response.notes),
    )
