from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Sequence

from .transport import (
    DotProductRequest,
    DotProductResponse,
    GeluBlockRequest,
    GeluBlockResponse,
    GemmTileI16Request,
    GemmTileI64Response,
)


@dataclass(slots=True, frozen=True)
class IverilogSimExecutor:
    project_root: Path
    iverilog: Path = Path("iverilog")
    vvp: Path = Path("vvp")

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", self.project_root.resolve())

    def name(self) -> str:
        return "iverilog-sim"

    def execute_dot_product(
        self,
        request: DotProductRequest,
        output_dir: Path,
    ) -> DotProductResponse:
        if len(request.vector_a) != 8 or len(request.vector_b) != 8:
            raise ValueError("dot-product simulator expects exactly 8 lanes")

        scratch_dir = self._create_scratch_dir(output_dir, "dot_product")
        request_path = scratch_dir / "sim_request.json"
        response_path = scratch_dir / "sim_response.json"
        vector_include_path = scratch_dir / "dot_product_vectors.vh"
        result_path = scratch_dir / "dot_product_result.txt"
        vvp_output_path = scratch_dir / "dot_product_i16x8_tb.out"

        self._write_json(request_path, asdict(request))
        vector_include_path.write_text(
            self._build_vector_include(request.vector_a, request.vector_b),
            encoding="utf-8",
        )

        self._run_command(
            [
                str(self.iverilog),
                "-g2012",
                "-o",
                str(vvp_output_path),
                str(self.project_root / "fpga/rtl/dot_product_i16x8.v"),
                str(self.project_root / "fpga/tb/dot_product_i16x8_tb.v"),
            ],
            cwd=scratch_dir,
            display_name=str(self.iverilog),
        )
        self._run_command(
            [str(self.vvp), str(vvp_output_path)],
            cwd=scratch_dir,
            display_name=str(self.vvp),
        )

        rtl_result = int(result_path.read_text(encoding="utf-8").strip())
        response = DotProductResponse(
            rtl_result=rtl_result,
            expected_result=request.expected_result,
            matched=rtl_result == request.expected_result,
            notes=[
                "Executed real RTL through direct Python -> iverilog/vvp invocation.",
                f"Received audio path: {request.audio_path}",
                "Operation: dot-product",
                f"Scratch directory: {scratch_dir}",
                f"Waveform: {scratch_dir / 'dot_product_i16x8_tb.vcd'}",
                "This primitive is the first reusable numeric block for future FPGA offload.",
            ],
        )
        self._write_json(response_path, asdict(response))
        return response

    def execute_gemm_tile(
        self,
        request: GemmTileI16Request,
        output_dir: Path,
    ) -> GemmTileI64Response:
        request.shape.as_layout().validate(
            len(request.lhs_tile),
            len(request.rhs_tile),
        )
        if request.shape.inner != 8:
            raise ValueError(
                "current simulator GEMM path expects inner dimension of exactly 8"
            )
        if len(request.accumulator_input) != request.shape.rows * request.shape.cols:
            expected = request.shape.rows * request.shape.cols
            raise ValueError(
                f"simulator accumulator length mismatch: expected {expected}, got {len(request.accumulator_input)}"
            )

        scratch_dir = self._create_scratch_dir(output_dir, "gemm_tile")
        request_path = scratch_dir / "sim_request.json"
        response_path = scratch_dir / "sim_response.json"
        vector_include_path = scratch_dir / "gemm_tile_vectors.vh"
        result_path = scratch_dir / "gemm_tile_result.txt"
        vvp_output_path = scratch_dir / "gemm_tile_i16x8_tb.out"

        self._write_json(request_path, request.to_dict())
        vector_include_path.write_text(
            self._build_gemm_tile_include(
                request.shape.rows,
                request.shape.cols,
                request.shape.inner,
                request.lhs_tile,
                request.rhs_tile,
                request.accumulator_input,
            ),
            encoding="utf-8",
        )

        self._run_command(
            [
                str(self.iverilog),
                "-g2012",
                "-o",
                str(vvp_output_path),
                str(self.project_root / "fpga/rtl/dot_product_i16x8.v"),
                str(self.project_root / "fpga/rtl/gemm_tile_i16x8.v"),
                str(self.project_root / "fpga/rtl/gemm_tile_accum_i16x8.v"),
                str(self.project_root / "fpga/tb/gemm_tile_i16x8_tb.v"),
            ],
            cwd=scratch_dir,
            display_name=str(self.iverilog),
        )
        self._run_command(
            [str(self.vvp), str(vvp_output_path)],
            cwd=scratch_dir,
            display_name=str(self.vvp),
        )

        rtl_output = [
            int(line.strip())
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rtl_output) != len(request.expected_output):
            raise ValueError(
                "simulator GEMM output length mismatch: "
                f"expected {len(request.expected_output)}, got {len(rtl_output)}"
            )

        response = GemmTileI64Response(
            shape=request.shape,
            rtl_output=rtl_output,
            expected_output=list(request.expected_output),
            matched=rtl_output == request.expected_output,
            notes=[
                "Executed real RTL GEMM+accum tile "
                f"{request.shape.rows}x{request.shape.cols}x{request.shape.inner} "
                "through direct Python -> iverilog/vvp invocation.",
                f"Received audio path: {request.audio_path}",
                "Operation: gemm-tile-accum",
                f"Scratch directory: {scratch_dir}",
                f"Waveform: {scratch_dir / 'gemm_tile_i16x8_tb.vcd'}",
                "This accumulator tile primitive can keep partial sums on the FPGA boundary.",
            ],
        )
        self._write_json(response_path, response.to_dict())
        return response

    def execute_gelu_block(
        self,
        request: GeluBlockRequest,
        output_dir: Path,
    ) -> GeluBlockResponse:
        if len(request.input_block) != 8 or len(request.expected_output) != 8:
            raise ValueError("gelu simulator expects exactly 8 lanes")

        scratch_dir = self._create_scratch_dir(output_dir, "gelu")
        request_path = scratch_dir / "sim_request.json"
        response_path = scratch_dir / "sim_response.json"
        vector_include_path = scratch_dir / "gelu_vectors.vh"
        result_path = scratch_dir / "gelu_result.txt"
        vvp_output_path = scratch_dir / "gelu_pwl_q8_8x8_tb.out"

        self._write_json(request_path, request.to_dict())
        vector_include_path.write_text(
            self._build_gelu_include(request.input_block),
            encoding="utf-8",
        )

        self._run_command(
            [
                str(self.iverilog),
                "-g2012",
                "-o",
                str(vvp_output_path),
                str(self.project_root / "fpga/rtl/gelu_pwl_q8_8.v"),
                str(self.project_root / "fpga/rtl/gelu_pwl_q8_8x8.v"),
                str(self.project_root / "fpga/tb/gelu_pwl_q8_8x8_tb.v"),
            ],
            cwd=scratch_dir,
            display_name=str(self.iverilog),
        )
        self._run_command(
            [str(self.vvp), str(vvp_output_path)],
            cwd=scratch_dir,
            display_name=str(self.vvp),
        )

        rtl_output = [
            int(line.strip())
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        response = GeluBlockResponse(
            rtl_output=rtl_output,
            expected_output=list(request.expected_output),
            matched=rtl_output == request.expected_output,
            notes=[
                "Executed real RTL GELU PWL block through direct Python -> iverilog/vvp invocation.",
                f"Received audio path: {request.audio_path}",
                "Operation: gelu-pwl-q8.8x8",
                f"Scratch directory: {scratch_dir}",
                f"Waveform: {scratch_dir / 'gelu_pwl_q8_8x8_tb.vcd'}",
                "This block is the first non-linear activation primitive on the FPGA boundary.",
            ],
        )
        self._write_json(response_path, response.to_dict())
        return response

    def _create_scratch_dir(self, output_dir: Path, prefix: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = output_dir.resolve()
        scratch_dir = output_dir / f"{prefix}_{time.time_ns()}"
        scratch_dir.mkdir(parents=True, exist_ok=False)
        return scratch_dir

    def _run_command(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        display_name: str,
    ) -> CompletedProcess[str]:
        completed = run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{display_name} exited with status {completed.returncode}.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return completed

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def _build_gemm_tile_include(
        cls,
        rows: int,
        cols: int,
        inner: int,
        lhs_tile: Sequence[int],
        rhs_tile: Sequence[int],
        accumulator_input: Sequence[int],
    ) -> str:
        accum_len = rows * cols
        if len(accumulator_input) != accum_len:
            raise ValueError(
                f"accumulator tile length mismatch: expected {accum_len}, got {len(accumulator_input)}"
            )

        lhs_bits = rows * inner * 16
        rhs_bits = inner * cols * 16
        accum_bits = rows * cols * 64

        return (
            f"localparam integer TILE_ROWS = {rows};\n"
            f"localparam integer TILE_COLS = {cols};\n"
            f"localparam integer TILE_INNER = {inner};\n"
            f"localparam signed [{lhs_bits - 1}:0] LHS_TILE = {cls._build_packed_i16_literal(lhs_tile)};\n"
            f"localparam signed [{rhs_bits - 1}:0] RHS_TILE = {cls._build_packed_i16_literal(rhs_tile)};\n"
            f"localparam signed [{accum_bits - 1}:0] ACCUM_TILE = {cls._build_packed_i64_literal(accumulator_input)};\n"
        )

    @classmethod
    def _build_vector_include(
        cls,
        vector_a: Sequence[int],
        vector_b: Sequence[int],
    ) -> str:
        lines: list[str] = []
        for index, value in enumerate(vector_a):
            lines.append(
                f"localparam signed [15:0] VEC_A{index} = {cls._verilog_i16_literal(value)};"
            )
        for index, value in enumerate(vector_b):
            lines.append(
                f"localparam signed [15:0] VEC_B{index} = {cls._verilog_i16_literal(value)};"
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def _build_gelu_include(cls, input_block: Sequence[int]) -> str:
        return (
            "localparam signed [127:0] INPUT_BLOCK = "
            f"{cls._build_packed_i16_literal(input_block)};\n"
        )

    @classmethod
    def _build_packed_i16_literal(cls, values: Sequence[int]) -> str:
        parts = [cls._verilog_i16_literal(value) for value in reversed(list(values))]
        return "{" + ", ".join(parts) + "}"

    @classmethod
    def _build_packed_i64_literal(cls, values: Sequence[int]) -> str:
        parts = [cls._verilog_i64_literal(value) for value in reversed(list(values))]
        return "{" + ", ".join(parts) + "}"

    @staticmethod
    def _verilog_i16_literal(value: int) -> str:
        if value < 0:
            return f"-16'sd{abs(int(value))}"
        return f"16'sd{int(value)}"

    @staticmethod
    def _verilog_i64_literal(value: int) -> str:
        if value < 0:
            return f"-64'sd{abs(int(value))}"
        return f"64'sd{int(value)}"
