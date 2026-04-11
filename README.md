# fpga-whisper

Two pathways:

- `ct2-python`: CPU baseline using the baked `distil-whisper/distil-small.en` model
- `fpga-hw`: real FPGA frontend over UART, with CT2 still handling encoder/decoder on the host

## Setup

```bash
uv sync
```

## Backends

### `ct2-python`

CPU baseline and correctness oracle.

```bash
uv run fpga-whisper transcribe samples/jfk.flac --backend ct2-python
uv run fpga-whisper benchmark samples/jfk.flac --backend ct2-python --iterations 5 --warmup 1
uv run fpga-whisper profile samples/jfk.flac --backend ct2-python --sample-interval-ms 250
```

### `fpga-hw`

Real board path.

- host: audio decode, STFT/power spectrum, CT2 transcription
- FPGA: frontend mel/log-mel over UART

```bash
uv run fpga-whisper transcribe samples/jfk.flac --backend fpga-hw --partition frontend
```

## FPGA Tests

UART and hardware smoke test:

```bash
uv run python -m src.scripts.test_fpga_uart --port /dev/ttyUSB1
```

Simple UART ping:

```bash
uv run python -u - <<'PY'
from src.fpga.uart import FpgaUartExecutor
fpga = FpgaUartExecutor(port="/dev/ttyUSB1", baud=115200, timeout=1.0)
print(fpga.ping())
fpga.close()
PY
```

Simulation/frontend validation commands:

```bash
uv run fpga-whisper logmel-frame-check
uv run fpga-whisper gemm-check
uv run fpga-whisper linear-check
uv run fpga-whisper projection-full-check
uv run fpga-whisper gelu-check
```

## Program The FPGA

Build for the connected S7-25 board:

```bash
vivado -mode batch -source fpga/scripts/build.tcl -tclargs xc7s25csga324-1
```

Program FPGA RAM:

```bash
vivado -mode batch -source fpga/scripts/program.tcl
```

Program SPI flash so it survives unplug/replug:

```bash
vivado -mode batch -source fpga/scripts/program.tcl -tclargs -flash
```

## UART Config

Defaults:

- port: `/dev/ttyUSB1`
- baud: `115200`

Override with:

```bash
export FPGA_WHISPER_UART_PORT=/dev/ttyUSB1
export FPGA_WHISPER_UART_BAUD=115200
export FPGA_WHISPER_UART_TIMEOUT=5.0
```

## Notes

- The current live FPGA path is `frontend` only.
- The flashed board now auto-boots the Whisper UART design.
