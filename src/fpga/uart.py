"""UART transport for real FPGA hardware.

Drop-in replacement for IverilogSimExecutor -- implements the same
FpgaExecutor protocol but talks to the Arty S7 over a serial port
using the whisper_top command protocol.

Protocol (both directions):
    [0xAA] [CMD/STATUS] [LEN_HI] [LEN_LO] [payload ...]

Requires: pyserial  (pip install pyserial)
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import serial

from .transport import (
    DotProductRequest,
    DotProductResponse,
    GeluBlockRequest,
    GeluBlockResponse,
    GemmTileBatchI16Request,
    GemmTileI16Request,
    GemmTileI64Response,
    LogMelFrameRequest,
    LogMelFrameResponse,
    MelFrameBatchRequest,
    MelFrameBatchResponse,
)

SYNC = 0xAA
CMD_PING = 0x01
CMD_DOT_PRODUCT = 0x02
CMD_GELU_BLOCK = 0x03
CMD_LOAD_MEL_COEFF = 0x04
CMD_MEL_FRAME = 0x05
STATUS_OK = 0x00


def _i16_to_le(value: int) -> bytes:
    return struct.pack("<h", value)


def _u16_to_le(value: int) -> bytes:
    return struct.pack("<H", value)


def _u24_to_le(value: int) -> bytes:
    return struct.pack("<I", value & 0xFFFFFF)[:3]


def _i64_from_le(data: bytes) -> int:
    return struct.unpack("<q", data)[0]


def _i16_from_le(data: bytes) -> int:
    return struct.unpack("<h", data)[0]


class FpgaUartError(RuntimeError):
    pass


@dataclass(slots=True)
class FpgaUartExecutor:
    """Talks to whisper_top on a real FPGA over UART."""

    port: str = "/dev/ttyUSB1"
    baud: int = 115_200
    timeout: float = 5.0
    _ser: serial.Serial | None = field(default=None, repr=False)
    _coeffs_loaded: bool = field(default=False, repr=False)

    def _open(self) -> serial.Serial:
        if self._ser is None or not self._ser.is_open:
            self._ser = serial.Serial(
                self.port, self.baud, timeout=self.timeout
            )
            self._ser.reset_input_buffer()
        return self._ser

    def close(self) -> None:
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
            self._ser = None

    # ── low-level protocol ────────────────────────────────────────

    def _send(self, cmd: int, payload: bytes = b"") -> None:
        length = len(payload)
        header = bytes([SYNC, cmd, (length >> 8) & 0xFF, length & 0xFF])
        ser = self._open()
        ser.write(header + payload)
        ser.flush()

    def _recv(self) -> tuple[int, bytes]:
        ser = self._open()
        sync = ser.read(1)
        if len(sync) == 0:
            raise FpgaUartError("timeout waiting for response sync byte")
        if sync[0] != SYNC:
            raise FpgaUartError(f"bad sync byte: 0x{sync[0]:02x}")
        hdr = ser.read(3)
        if len(hdr) < 3:
            raise FpgaUartError("timeout reading response header")
        status = hdr[0]
        length = (hdr[1] << 8) | hdr[2]
        data = ser.read(length) if length > 0 else b""
        if len(data) < length:
            raise FpgaUartError(
                f"short read: expected {length} bytes, got {len(data)}"
            )
        if status != STATUS_OK:
            raise FpgaUartError(f"FPGA returned error status 0x{status:02x}")
        return status, data

    def _command(self, cmd: int, payload: bytes = b"") -> bytes:
        self._send(cmd, payload)
        _, data = self._recv()
        return data

    # ── public API (FpgaExecutor protocol) ────────────────────────

    def name(self) -> str:
        return f"fpga-uart:{self.port}"

    def ping(self) -> tuple[int, int, int]:
        data = self._command(CMD_PING)
        return data[0], data[1], data[2]

    def load_mel_coefficients(self, coefficients: Sequence[int]) -> None:
        """Send 16 080 u16 mel coefficients to the FPGA (one-time)."""
        if len(coefficients) != 16_080:
            raise ValueError(
                f"expected 16080 coefficients, got {len(coefficients)}"
            )
        payload = b"".join(_u16_to_le(c & 0xFFFF) for c in coefficients)
        self._command(CMD_LOAD_MEL_COEFF, payload)
        self._coeffs_loaded = True

    def execute_dot_product(
        self,
        request: DotProductRequest,
        output_dir: Path,
    ) -> DotProductResponse:
        payload = b"".join(
            _i16_to_le(v) for v in list(request.vector_a) + list(request.vector_b)
        )
        data = self._command(CMD_DOT_PRODUCT, payload)
        rtl_result = _i64_from_le(data[:8])
        return DotProductResponse(
            rtl_result=rtl_result,
            expected_result=request.expected_result,
            matched=rtl_result == request.expected_result,
            notes=[f"Executed on real FPGA via {self.name()}"],
        )

    def execute_gelu_block(
        self,
        request: GeluBlockRequest,
        output_dir: Path,
    ) -> GeluBlockResponse:
        payload = b"".join(_i16_to_le(v) for v in request.input_block)
        data = self._command(CMD_GELU_BLOCK, payload)
        rtl_output = [_i16_from_le(data[i * 2 : i * 2 + 2]) for i in range(8)]
        return GeluBlockResponse(
            rtl_output=rtl_output,
            expected_output=list(request.expected_output),
            matched=rtl_output == list(request.expected_output),
            notes=[f"Executed on real FPGA via {self.name()}"],
        )

    def _ensure_coefficients(self, mel_coefficients: Sequence[int]) -> None:
        if not self._coeffs_loaded:
            self.load_mel_coefficients(mel_coefficients)

    def _send_mel_frame(self, power_spectrum: Sequence[int]) -> list[int]:
        """Send one 201-bin power frame, get 80 log-mel values back."""
        payload = b"".join(_u24_to_le(v) for v in power_spectrum)
        data = self._command(CMD_MEL_FRAME, payload)
        return [_i16_from_le(data[i * 2 : i * 2 + 2]) for i in range(80)]

    def execute_logmel_frame(
        self,
        request: LogMelFrameRequest,
        output_dir: Path,
    ) -> LogMelFrameResponse:
        request.validate()
        self._ensure_coefficients(request.mel_coefficients)
        rtl_output = self._send_mel_frame(request.power_spectrum)
        return LogMelFrameResponse(
            rtl_output=rtl_output,
            expected_output=list(request.expected_output),
            matched=rtl_output == list(request.expected_output),
            notes=[f"Executed on real FPGA via {self.name()}"],
        )

    def execute_mel_frame_batch(
        self,
        request: MelFrameBatchRequest,
        output_dir: Path,
    ) -> MelFrameBatchResponse:
        request.validate()
        self._ensure_coefficients(request.mel_coefficients)

        rtl_output: list[int] = []
        for frame_idx in range(request.frame_count):
            start = frame_idx * 201
            power_frame = request.power_frames[start : start + 201]
            frame_result = self._send_mel_frame(power_frame)
            rtl_output.extend(frame_result)

        return MelFrameBatchResponse(
            frame_count=request.frame_count,
            rtl_output=rtl_output,
            expected_output=list(request.expected_output),
            matched=rtl_output == list(request.expected_output),
            notes=[
                f"Executed {request.frame_count} frames on real FPGA via {self.name()}"
            ],
        )

    # GEMM is not implemented in whisper_top yet -- stub so the protocol is satisfied
    def execute_gemm_tile(
        self, request: GemmTileI16Request, output_dir: Path
    ) -> GemmTileI64Response:
        raise NotImplementedError("GEMM not yet wired in whisper_top")

    def execute_gemm_tile_batch(
        self, request: GemmTileBatchI16Request, output_dir: Path
    ) -> list[GemmTileI64Response]:
        raise NotImplementedError("GEMM batch not yet wired in whisper_top")
