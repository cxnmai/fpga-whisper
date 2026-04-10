#!/usr/bin/env python3
"""Bridge script for the Rust -> RTL simulation scaffold.

The first real exercised primitive is a signed 8-lane int16 dot product.
Rust writes a JSON request, this script generates Verilog vector definitions,
runs iverilog/vvp, reads the RTL result, and writes a JSON response back.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


VECTOR_INCLUDE = Path("fpga/tmp/dot_product_vectors.vh")
RESULT_FILE = Path("fpga/tmp/dot_product_result.txt")
VVP_OUTPUT = Path("fpga/tmp/dot_product_i16x8_tb.out")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FPGA simulator bridge.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser.parse_args()


def write_vector_include(vector_a: list[int], vector_b: list[int]) -> None:
    def verilog_i16_literal(value: int) -> str:
        if value < 0:
            return f"-16'sd{abs(value)}"
        return f"16'sd{value}"

    lines = []
    for index, value in enumerate(vector_a):
        lines.append(
            f"localparam signed [15:0] VEC_A{index} = {verilog_i16_literal(value)};"
        )
    for index, value in enumerate(vector_b):
        lines.append(
            f"localparam signed [15:0] VEC_B{index} = {verilog_i16_literal(value)};"
        )
    VECTOR_INCLUDE.parent.mkdir(parents=True, exist_ok=True)
    VECTOR_INCLUDE.write_text("\n".join(lines) + "\n")


def run_iverilog() -> None:
    subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-o",
            str(VVP_OUTPUT),
            "fpga/rtl/dot_product_i16x8.v",
            "fpga/tb/dot_product_i16x8_tb.v",
        ],
        check=True,
    )
    subprocess.run(["vvp", str(VVP_OUTPUT)], check=True)


def main() -> int:
    args = parse_args()
    request = json.loads(args.request.read_text())

    vector_a = request["vector_a"]
    vector_b = request["vector_b"]
    if len(vector_a) != 8 or len(vector_b) != 8:
        raise SystemExit("dot-product simulator expects exactly 8 lanes")

    write_vector_include(vector_a, vector_b)
    run_iverilog()

    rtl_result = int(RESULT_FILE.read_text().strip())
    expected_result = int(request["expected_result"])
    response = {
        "operation": request["operation"],
        "rtl_result": rtl_result,
        "expected_result": expected_result,
        "matched": rtl_result == expected_result,
        "notes": [
            "Executed real RTL through iverilog/vvp.",
            f"Received audio path: {request['audio_path']}",
            f"Operation: {request['operation']}",
            f"Waveform: fpga/tmp/dot_product_i16x8_tb.vcd",
            "This primitive is the first reusable numeric block for future FPGA offload.",
        ],
    }

    args.response.parent.mkdir(parents=True, exist_ok=True)
    args.response.write_text(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
