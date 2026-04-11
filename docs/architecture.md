# Architecture

## Pipeline split

The intended runtime keeps high-level transcription control on the host and moves dense numeric work onto the FPGA as the project matures.

| Stage | Host baseline | FPGA target |
| --- | --- | --- |
| Audio decode | host | host |
| STFT / log-mel | host | FPGA |
| Encoder | host | FPGA |
| Decoder math | host | FPGA |
| Decode policy | host | host |
| Post-process | host | host |

## Backend modes

### `ct2-python`

Use this mode as the correctness oracle. It should mirror the host-only behavior and stay simple enough to compare against every FPGA milestone.

### `fpga-hybrid`

Use this mode for incremental handoff:

- `frontend`: STFT/log-mel only
- `encoder`: front-end plus encoder
- `hybrid`: front-end, encoder, and decoder math

### `fpga-hw`

Use this mode for the real Arty S7 backend. The first revision is frontend-only:

- host computes audio decode and power spectra
- FPGA computes mel accumulation and returns board log-mel output
- host converts the board output into Whisper features and hands them to CT2

## Hardware guidance

The Arty S7 is not large enough to treat each layer as a host-driven RPC. The practical boundary is coarse-grained execution with resident weights and cache state per chunk.

That implies:

- quantize the model before attempting meaningful offload
- use external DDR for weights and activations
- keep host/FPGA transfers at chunk or block granularity, not per-op granularity

## Milestones

1. Wire the CTranslate2 worker and validate transcripts against `faster-whisper`.
2. Add feature extraction offload and verify mel parity.
3. Bring up the real board backend over UART using the same frontend contract as `fpga-sim`.
4. Offload encoder blocks with chunk-level transfers.
5. Replace host decoder math with an FPGA matrix engine while keeping beam search on the host.
