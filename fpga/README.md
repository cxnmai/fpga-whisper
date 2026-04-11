# FPGA workspace

This directory is the hardware-side root for the project.

Suggested subdirectories:

- `rtl/`: reusable DSP, BRAM, and matrix-engine blocks
- `sim/`: simulation artifacts and waveform-oriented validation
- `build/`: synthesis and bitstream outputs
- `scripts/`: export, packing, and memory-image tooling
- `tb/`: standalone test benches for `iverilog`/`vvp`
- `vectors/`: request/response vectors for host-simulator exchange
- `tmp/`: transient simulator I/O produced by the host runtime, one scratch directory per invocation

The target split for the FPGA is:

- feature extraction
- encoder
- decoder math

The host should continue to own:

- audio decoding
- beam search / greedy decode control
- tokenization
- post-processing

Current simulator scaffold:

- Python backend: `fpga-sim`
- Real board backend: `fpga-hw`
- transport contract: per-run scratch directories under `fpga/tmp/` with host-generated memfiles, simulator outputs, and optional debug artifacts
- simulator invocation: host runtime -> direct `iverilog` / `vvp`
- board invocation: host runtime -> UART transport -> Arty S7 firmware
- first real RTL primitive: `fpga/rtl/dot_product_i16x8.v`
- first real testbench: `fpga/tb/dot_product_i16x8_tb.v`
- first reusable tile primitive: `fpga/rtl/gemm_tile_i16x8.v`
- accumulator-aware tile primitive: `fpga/rtl/gemm_tile_accum_i16x8.v`
- first tile testbench: `fpga/tb/gemm_tile_i16x8_tb.v`
- first activation primitive: `fpga/rtl/gelu_pwl_q8_8.v`
- first activation vector wrapper: `fpga/rtl/gelu_pwl_q8_8x8.v`
- first activation testbench: `fpga/tb/gelu_pwl_q8_8x8_tb.v`
- first frontend mel primitive: `fpga/rtl/mel_filterbank_201x80.v`
- first frontend log primitive: `fpga/rtl/log_mel_q8_8.v`
- first frontend frame wrapper: `fpga/rtl/log_mel_frame.v`
- first frontend testbench: `fpga/tb/log_mel_frame_tb.v`
- first frontend batch testbench: `fpga/tb/mel_frame_batch_tb.v`

The current frontend simulation path is already able to drive a real transcription:

- host computes Whisper-compatible power spectra
- RTL computes batched mel accumulation
- host applies Whisper log/clamp normalization
- host passes the resulting feature tensor into the shared CTranslate2 worker path

The real-board path uses the same frontend shape, but over UART instead of file exchange.

The goal of this workspace is unchanged: keep the hardware boundary stable while progressively moving meaningful parts of Whisper onto the FPGA.
