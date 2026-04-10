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
    MelFrameBatchRequest,
    MelFrameBatchResponse,
    GemmTileBatchI16Request,
    GemmTileI16Request,
    GemmTileI64Response,
    LogMelFrameRequest,
    LogMelFrameResponse,
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
        result_path = scratch_dir / "dot_product_result.txt"
        self._write_i16_mem(scratch_dir / "dot_product_a.mem", request.vector_a)
        self._write_i16_mem(scratch_dir / "dot_product_b.mem", request.vector_b)

        self._write_json(request_path, asdict(request))
        vvp_output_path = self._compiled_dot_product_binary(output_dir)
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
                "Executed real RTL through a cached iverilog/vvp simulation binary.",
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
        result_path = scratch_dir / "gemm_tile_result.txt"
        self._write_i16_mem(scratch_dir / "gemm_tile_lhs.mem", request.lhs_tile)
        self._write_i16_mem(scratch_dir / "gemm_tile_rhs.mem", request.rhs_tile)
        self._write_i64_mem(
            scratch_dir / "gemm_tile_accum.mem",
            request.accumulator_input,
        )

        self._write_json(request_path, request.to_dict())
        vvp_output_path = self._compiled_gemm_tile_binary(
            output_dir,
            request.shape.rows,
            request.shape.cols,
            request.shape.inner,
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
                "through a cached iverilog/vvp simulation binary.",
                f"Received audio path: {request.audio_path}",
                "Operation: gemm-tile-accum",
                f"Scratch directory: {scratch_dir}",
                f"Waveform: {scratch_dir / 'gemm_tile_i16x8_tb.vcd'}",
                "This accumulator tile primitive can keep partial sums on the FPGA boundary.",
            ],
        )
        self._write_json(response_path, response.to_dict())
        return response

    def execute_mel_frame_batch(
        self,
        request: MelFrameBatchRequest,
        output_dir: Path,
    ) -> MelFrameBatchResponse:
        request.validate()

        scratch_dir = self._create_scratch_dir(output_dir, "mel_frame_batch")
        request_path = scratch_dir / "sim_request.json"
        response_path = scratch_dir / "sim_response.json"
        result_path = scratch_dir / "mel_frame_batch_result.txt"
        self._write_u24_mem(scratch_dir / "power_frames.mem", request.power_frames)
        self._write_i16_mem(scratch_dir / "mel_coeff.mem", request.mel_coefficients)

        self._write_json(request_path, request.to_dict())
        vvp_output_path = self._compiled_mel_frame_batch_binary(
            output_dir,
            request.frame_count,
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
        response = MelFrameBatchResponse(
            frame_count=request.frame_count,
            rtl_output=rtl_output,
            expected_output=list(request.expected_output),
            matched=rtl_output == request.expected_output,
            notes=[
                "Executed real RTL mel frame batch through a cached iverilog/vvp simulation binary.",
                f"Received audio path: {request.audio_path}",
                f"Operation: mel-frame-batch ({request.frame_count} frames)",
                f"Scratch directory: {scratch_dir}",
                f"Waveform: {scratch_dir / 'mel_frame_batch_tb.vcd'}",
                "This is the chunkable frontend bridge toward real FPGA-assisted transcription.",
            ],
        )
        self._write_json(response_path, response.to_dict())
        return response

    def execute_gemm_tile_batch(
        self,
        request: GemmTileBatchI16Request,
        output_dir: Path,
    ) -> list[GemmTileI64Response]:
        request.validate()
        if request.shape.inner != 8:
            raise ValueError(
                "current simulator GEMM batch path expects inner dimension of exactly 8"
            )

        scratch_dir = self._create_scratch_dir(output_dir, "gemm_tile_batch")
        request_path = scratch_dir / "sim_request.json"
        response_path = scratch_dir / "sim_response.json"
        result_path = scratch_dir / "gemm_tile_batch_result.txt"

        lhs_flat: list[int] = []
        rhs_flat: list[int] = []
        accum_flat: list[int] = []
        for item in request.requests:
            lhs_flat.extend(item.lhs_tile)
            rhs_flat.extend(item.rhs_tile)
            accum_flat.extend(item.accumulator_input)

        self._write_i16_mem(scratch_dir / "gemm_tile_batch_lhs.mem", lhs_flat)
        self._write_i16_mem(scratch_dir / "gemm_tile_batch_rhs.mem", rhs_flat)
        self._write_i64_mem(scratch_dir / "gemm_tile_batch_accum.mem", accum_flat)
        self._write_json(request_path, request.to_dict())

        vvp_output_path = self._compiled_gemm_tile_batch_binary(
            output_dir,
            request.shape.rows,
            request.shape.cols,
            request.shape.inner,
            len(request.requests),
        )
        self._run_command(
            [str(self.vvp), str(vvp_output_path)],
            cwd=scratch_dir,
            display_name=str(self.vvp),
        )

        raw_lines = [
            int(line.strip())
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        output_len = request.shape.rows * request.shape.cols
        expected_total = len(request.requests) * output_len
        if len(raw_lines) != expected_total:
            raise ValueError(
                "simulator GEMM batch output length mismatch: "
                f"expected {expected_total}, got {len(raw_lines)}"
            )

        common_notes = [
            "Executed real RTL GEMM+accum tile batch "
            f"{request.shape.rows}x{request.shape.cols}x{request.shape.inner} "
            f"for {len(request.requests)} cases through a cached iverilog/vvp simulation binary.",
            f"Scratch directory: {scratch_dir}",
            f"Waveform: {scratch_dir / 'gemm_tile_batch_i16x8_tb.vcd'}",
        ]
        responses: list[GemmTileI64Response] = []
        for index, item in enumerate(request.requests):
            start = index * output_len
            end = start + output_len
            rtl_output = raw_lines[start:end]
            responses.append(
                GemmTileI64Response(
                    shape=item.shape,
                    rtl_output=rtl_output,
                    expected_output=list(item.expected_output),
                    matched=rtl_output == item.expected_output,
                    notes=[
                        *common_notes,
                        f"Received audio path: {item.audio_path}",
                        f"Operation: gemm-tile-batch case {index + 1}/{len(request.requests)}",
                    ],
                )
            )

        self._write_json(
            response_path,
            {
                "case_count": len(request.requests),
                "responses": [response.to_dict() for response in responses],
            },
        )
        return responses

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
        result_path = scratch_dir / "gelu_result.txt"
        self._write_i16_mem(scratch_dir / "gelu_input.mem", request.input_block)

        self._write_json(request_path, request.to_dict())
        vvp_output_path = self._compiled_gelu_binary(output_dir)
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
                "Executed real RTL GELU PWL block through a cached iverilog/vvp simulation binary.",
                f"Received audio path: {request.audio_path}",
                "Operation: gelu-pwl-q8.8x8",
                f"Scratch directory: {scratch_dir}",
                f"Waveform: {scratch_dir / 'gelu_pwl_q8_8x8_tb.vcd'}",
                "This block is the first non-linear activation primitive on the FPGA boundary.",
            ],
        )
        self._write_json(response_path, response.to_dict())
        return response

    def execute_logmel_frame(
        self,
        request: LogMelFrameRequest,
        output_dir: Path,
    ) -> LogMelFrameResponse:
        request.validate()

        scratch_dir = self._create_scratch_dir(output_dir, "logmel")
        request_path = scratch_dir / "sim_request.json"
        response_path = scratch_dir / "sim_response.json"
        result_path = scratch_dir / "logmel_result.txt"
        self._write_u24_mem(scratch_dir / "power_spectrum.mem", request.power_spectrum)
        self._write_i16_mem(scratch_dir / "mel_coeff.mem", request.mel_coefficients)

        self._write_json(request_path, request.to_dict())
        vvp_output_path = self._compiled_logmel_binary(output_dir)
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
        response = LogMelFrameResponse(
            rtl_output=rtl_output,
            expected_output=list(request.expected_output),
            matched=rtl_output == request.expected_output,
            notes=[
                "Executed real RTL log-mel frame block through a cached iverilog/vvp simulation binary.",
                f"Received audio path: {request.audio_path}",
                "Operation: logmel-frame",
                f"Scratch directory: {scratch_dir}",
                f"Waveform: {scratch_dir / 'log_mel_frame_tb.vcd'}",
                "This is the first frontend-oriented Verilog block on the future transcription path.",
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

    def _compiled_dot_product_binary(self, output_dir: Path) -> Path:
        binary_path = self._build_dir(output_dir) / "dot_product_i16x8_tb.vvp"
        sources = [
            self.project_root / "fpga/rtl/dot_product_i16x8.v",
            self.project_root / "fpga/tb/dot_product_i16x8_tb.v",
        ]
        self._ensure_compiled(binary_path, sources)
        return binary_path

    def _compiled_gemm_tile_binary(
        self,
        output_dir: Path,
        rows: int,
        cols: int,
        inner: int,
    ) -> Path:
        if inner != 8:
            raise ValueError("current simulator GEMM path expects inner dimension of exactly 8")
        binary_path = self._build_dir(output_dir) / f"gemm_tile_i16x8_tb_r{rows}_c{cols}_k{inner}.vvp"
        sources = [
            self.project_root / "fpga/rtl/dot_product_i16x8.v",
            self.project_root / "fpga/rtl/gemm_tile_i16x8.v",
            self.project_root / "fpga/rtl/gemm_tile_accum_i16x8.v",
            self.project_root / "fpga/tb/gemm_tile_i16x8_tb.v",
        ]
        parameters = [
            "-P",
            f"gemm_tile_i16x8_tb.TILE_ROWS={rows}",
            "-P",
            f"gemm_tile_i16x8_tb.TILE_COLS={cols}",
        ]
        self._ensure_compiled(binary_path, sources, parameters)
        return binary_path

    def _compiled_gemm_tile_batch_binary(
        self,
        output_dir: Path,
        rows: int,
        cols: int,
        inner: int,
        case_count: int,
    ) -> Path:
        if inner != 8:
            raise ValueError(
                "current simulator GEMM batch path expects inner dimension of exactly 8"
            )
        binary_path = (
            self._build_dir(output_dir)
            / f"gemm_tile_batch_i16x8_tb_r{rows}_c{cols}_k{inner}_n{case_count}.vvp"
        )
        sources = [
            self.project_root / "fpga/rtl/dot_product_i16x8.v",
            self.project_root / "fpga/rtl/gemm_tile_i16x8.v",
            self.project_root / "fpga/rtl/gemm_tile_accum_i16x8.v",
            self.project_root / "fpga/tb/gemm_tile_batch_i16x8_tb.v",
        ]
        parameters = [
            "-P",
            f"gemm_tile_batch_i16x8_tb.TILE_ROWS={rows}",
            "-P",
            f"gemm_tile_batch_i16x8_tb.TILE_COLS={cols}",
            "-P",
            f"gemm_tile_batch_i16x8_tb.CASE_COUNT={case_count}",
        ]
        self._ensure_compiled(binary_path, sources, parameters)
        return binary_path

    def _compiled_gelu_binary(self, output_dir: Path) -> Path:
        binary_path = self._build_dir(output_dir) / "gelu_pwl_q8_8x8_tb.vvp"
        sources = [
            self.project_root / "fpga/rtl/gelu_pwl_q8_8.v",
            self.project_root / "fpga/rtl/gelu_pwl_q8_8x8.v",
            self.project_root / "fpga/tb/gelu_pwl_q8_8x8_tb.v",
        ]
        self._ensure_compiled(binary_path, sources)
        return binary_path

    def _compiled_logmel_binary(self, output_dir: Path) -> Path:
        binary_path = self._build_dir(output_dir) / "log_mel_frame_tb.vvp"
        sources = [
            self.project_root / "fpga/rtl/mel_filterbank_201x80.v",
            self.project_root / "fpga/rtl/log_mel_q8_8.v",
            self.project_root / "fpga/rtl/log_mel_frame.v",
            self.project_root / "fpga/tb/log_mel_frame_tb.v",
        ]
        self._ensure_compiled(binary_path, sources)
        return binary_path

    def _compiled_mel_frame_batch_binary(
        self,
        output_dir: Path,
        frame_count: int,
    ) -> Path:
        binary_path = (
            self._build_dir(output_dir) / f"mel_frame_batch_tb_n{frame_count}.vvp"
        )
        sources = [
            self.project_root / "fpga/rtl/mel_filterbank_201x80.v",
            self.project_root / "fpga/tb/mel_frame_batch_tb.v",
        ]
        parameters = [
            "-P",
            f"mel_frame_batch_tb.FRAME_COUNT={frame_count}",
        ]
        self._ensure_compiled(binary_path, sources, parameters)
        return binary_path

    def _ensure_compiled(
        self,
        binary_path: Path,
        sources: Sequence[Path],
        extra_args: Sequence[str] = (),
    ) -> None:
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._needs_recompile(binary_path, sources):
            return

        command = [
            str(self.iverilog),
            "-g2012",
            *extra_args,
            "-o",
            str(binary_path),
            *(str(source) for source in sources),
        ]
        self._run_command(
            command,
            cwd=self.project_root,
            display_name=str(self.iverilog),
        )

    @staticmethod
    def _needs_recompile(binary_path: Path, sources: Sequence[Path]) -> bool:
        if not binary_path.exists():
            return True

        binary_mtime = binary_path.stat().st_mtime
        return any(source.stat().st_mtime > binary_mtime for source in sources)

    @staticmethod
    def _build_dir(output_dir: Path) -> Path:
        return output_dir.resolve() / "build"

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
    def _write_i16_mem(cls, path: Path, values: Sequence[int]) -> None:
        cls._write_mem(path, values, bits=16)

    @classmethod
    def _write_u24_mem(cls, path: Path, values: Sequence[int]) -> None:
        cls._write_mem(path, values, bits=24)

    @classmethod
    def _write_i64_mem(cls, path: Path, values: Sequence[int]) -> None:
        cls._write_mem(path, values, bits=64)

    @staticmethod
    def _write_mem(path: Path, values: Sequence[int], *, bits: int) -> None:
        mask = (1 << bits) - 1
        lines = [f"{(int(value) & mask):0{bits // 4}x}" for value in values]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
