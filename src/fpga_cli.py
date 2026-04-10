from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .fpga.kernels.gelu import GeluComparison, simulate_gelu_block
from .fpga.kernels.gemm import (
    MatrixI16,
    MatrixI64,
    simulate_gemm_tile,
    simulate_gemm_tile_with_accumulator,
    software_gemm_with_accumulator,
)
from .fpga.kernels.linear import (
    LinearComparison,
    LinearLayerI16,
    format_vector_i64,
    simulate_linear,
)
from .fpga.quant import Q8_8, FixedPointConfig
from .fpga.sim import IverilogSimExecutor
from .fpga.transport import GemmTileBatchI16Request, GemmTileI16Request, GemmTileShape
from .model.ct2 import Ct2ModelBin
from .model.reference import (
    ensure_reference_activation_export,
    load_reference_activation,
)

WEIGHT_NAME = "encoder/layer_0/ffn/linear_0/weight"
BIAS_NAME = "encoder/layer_0/ffn/linear_0/bias"
REFERENCE_AUDIO = Path("samples/jfk.flac")


@dataclass(slots=True)
class ProjectionCaseResult:
    sequence_index: int
    input_start: int
    output_start: int
    input_float: list[float]
    input_quantized: list[int]
    bias_float: list[float]
    bias_quantized: list[int]
    float_reference: list[float]
    rtl_dequantized: list[float]
    max_abs_error: float
    matched: bool
    comparison: LinearComparison


@dataclass(slots=True)
class ProjectionFullCaseResult:
    sequence_index: int
    output_start: int
    tile_inner: int
    tile_cols: int
    tile_count: int
    bias_float: list[float]
    bias_quantized: list[int]
    software_output: list[int]
    rtl_output: list[int]
    float_reference: list[float]
    rtl_dequantized: list[float]
    max_tile_error: float
    max_abs_error: float
    matched: bool


@dataclass(slots=True)
class GeluCaseResult:
    sequence_index: int
    output_start: int
    tile_cols: int
    projection_tiles: int
    projection_max_abs_error: float
    comparison: GeluComparison


def run_gemm_check(config: AppConfig) -> None:
    executor = IverilogSimExecutor(project_root=config.project_root)
    lhs = MatrixI16(
        rows=2,
        cols=8,
        values=[
            3,
            -2,
            7,
            4,
            -1,
            5,
            2,
            -3,
            1,
            6,
            -5,
            2,
            8,
            -4,
            3,
            7,
        ],
    )
    rhs = MatrixI16(
        rows=8,
        cols=2,
        values=[
            6,
            1,
            8,
            -3,
            -4,
            2,
            1,
            9,
            9,
            -1,
            -2,
            5,
            3,
            4,
            5,
            -6,
        ],
    )

    comparison = simulate_gemm_tile(
        executor=executor,
        output_dir=config.resolved_fpga_sim_io_dir,
        audio_path="samples/jfk.flac",
        lhs=lhs,
        rhs=rhs,
    )

    print(f"gemm_check: matched = {comparison.matched}")
    print("software result:")
    print_matrix_i64(comparison.software)
    print("rtl result:")
    print_matrix_i64(comparison.rtl)
    print("notes:")
    for note in dedupe_notes(comparison.notes):
        print(f"- {note}")


def run_linear_check(config: AppConfig) -> None:
    executor = IverilogSimExecutor(project_root=config.project_root)
    layer = LinearLayerI16(
        input_dim=8,
        output_dim=3,
        weights=MatrixI16(
            rows=8,
            cols=3,
            values=[
                6,
                1,
                -2,
                8,
                -3,
                4,
                -4,
                2,
                5,
                1,
                9,
                -1,
                9,
                -1,
                3,
                -2,
                5,
                7,
                3,
                4,
                -6,
                5,
                -6,
                2,
            ],
        ),
        bias=[4, -3, 9],
        quant=Q8_8,
    )
    input_values = [3, -2, 7, 4, -1, 5, 2, -3]

    comparison = simulate_linear(
        executor=executor,
        output_dir=config.resolved_fpga_sim_io_dir,
        audio_path="samples/jfk.flac",
        layer=layer,
        input_values=input_values,
    )

    print(f"linear_check: matched = {comparison.matched}")
    print(f"linear_check: quantization = {layer.quant.description()}")
    print(f"input: {input_values!r}")
    print(f"software raw output: {format_vector_i64(comparison.software_output)}")
    print(f"rtl raw output: {format_vector_i64(comparison.rtl_output)}")
    print(
        "software dequantized: "
        f"{format_vector_f32(dequantize_outputs(layer.quant, comparison.software_output))}"
    )
    print(
        "rtl dequantized: "
        f"{format_vector_f32(dequantize_outputs(layer.quant, comparison.rtl_output))}"
    )
    print("gemm tile output:")
    print_matrix_i64(comparison.gemm.rtl)
    print("notes:")
    for note in dedupe_notes(comparison.notes):
        print(f"- {note}")


def run_projection_tile_check(config: AppConfig) -> None:
    input_start = 0
    output_start = 0
    tile_inner = 8
    tile_cols = 3

    executor = IverilogSimExecutor(project_root=config.project_root)
    quant = Q8_8
    model_bin_path = config.model_bin_path()
    model = Ct2ModelBin.open(model_bin_path)
    weight = model.read_tensor_f32(WEIGHT_NAME)
    bias = model.read_tensor_f32(BIAS_NAME)

    output_dim, input_dim = expect_rank2(weight.info.shape, WEIGHT_NAME)
    if bias.info.shape != [output_dim]:
        raise RuntimeError(
            f"bias shape mismatch for {BIAS_NAME}: expected [{output_dim}], got {bias.info.shape!r}"
        )
    if input_start + tile_inner > input_dim or output_start + tile_cols > output_dim:
        raise RuntimeError(
            "requested tile "
            f"[{output_start}..{output_start + tile_cols}) x "
            f"[{input_start}..{input_start + tile_inner}) exceeds tensor shape "
            f"{output_dim}x{input_dim}"
        )

    activation_export_path, activation_export = load_reference_export(config)
    case = run_projection_case(
        executor=executor,
        config=config,
        model_bin_path=model_bin_path,
        weight_values=weight.values,
        input_dim=input_dim,
        bias_values=bias.values,
        activation=activation_export.activations[0],
        sequence_index=0,
        input_start=input_start,
        output_start=output_start,
        tile_inner=tile_inner,
        tile_cols=tile_cols,
        quant=quant,
    )

    print(f"projection_tile_check: matched = {case.matched}")
    print_model_info(model_bin_path, model)
    print(f"reference activation cache: {activation_export_path}")
    print(f"reference audio: {activation_export.audio_path}")
    print(f"reference layer: {activation_export.layer_name}")
    print(f"reference exported positions: {activation_export.exported_positions}")
    print(f"reference sequence length: {activation_export.sequence_length}")
    print(f"reference sequence index: {case.sequence_index}")
    print(f"projection tensor: {WEIGHT_NAME}")
    print(f"bias tensor: {BIAS_NAME}")
    print(
        f"tile: outputs [{output_start}..{output_start + tile_cols}), "
        f"inputs [{input_start}..{input_start + tile_inner})"
    )
    print(f"quantization: {quant.description()}")
    print(f"input float: {format_vector_f32(case.input_float)}")
    print(f"input quantized: {case.input_quantized!r}")
    print(f"bias float: {format_vector_f32(case.bias_float)}")
    print(f"bias quantized: {case.bias_quantized!r}")
    print(f"software raw output: {format_vector_i64(case.comparison.software_output)}")
    print(f"rtl raw output: {format_vector_i64(case.comparison.rtl_output)}")
    print(f"float reference: {format_vector_f32(case.float_reference)}")
    print(f"rtl dequantized: {format_vector_f32(case.rtl_dequantized)}")
    print(f"max_abs_error: {case.max_abs_error:.6f}")
    print("notes:")
    for note in dedupe_notes(case.comparison.notes):
        print(f"- {note}")


def run_projection_sweep_check(config: AppConfig) -> None:
    tile_inner = 8
    tile_cols = 3

    executor = IverilogSimExecutor(project_root=config.project_root)
    quant = Q8_8
    model_bin_path = config.model_bin_path()
    model = Ct2ModelBin.open(model_bin_path)
    weight = model.read_tensor_f32(WEIGHT_NAME)
    bias = model.read_tensor_f32(BIAS_NAME)
    output_dim, input_dim = expect_rank2(weight.info.shape, WEIGHT_NAME)
    if bias.info.shape != [output_dim]:
        raise RuntimeError(
            f"bias shape mismatch for {BIAS_NAME}: expected [{output_dim}], got {bias.info.shape!r}"
        )

    activation_export_path, activation_export = load_reference_export(config)
    output_starts = build_output_starts(output_dim, tile_cols)
    input_starts = build_input_starts(input_dim, tile_inner)

    cases: list[ProjectionCaseResult] = []
    for sequence_index, activation in enumerate(activation_export.activations):
        for input_start in input_starts:
            for output_start in output_starts:
                cases.append(
                    run_projection_case(
                        executor=executor,
                        config=config,
                        model_bin_path=model_bin_path,
                        weight_values=weight.values,
                        input_dim=input_dim,
                        bias_values=bias.values,
                        activation=activation,
                        sequence_index=sequence_index,
                        input_start=input_start,
                        output_start=output_start,
                        tile_inner=tile_inner,
                        tile_cols=tile_cols,
                        quant=quant,
                    )
                )

    all_matched = all(case.matched for case in cases)
    worst_error = max((case.max_abs_error for case in cases), default=0.0)
    avg_error = sum(case.max_abs_error for case in cases) / len(cases) if cases else 0.0

    print(f"projection_sweep_check: matched = {all_matched}")
    print_model_info(model_bin_path, model)
    print(f"reference activation cache: {activation_export_path}")
    print(f"reference audio: {activation_export.audio_path}")
    print(f"reference layer: {activation_export.layer_name}")
    print(f"reference exported positions: {activation_export.exported_positions}")
    print(f"reference sequence length: {activation_export.sequence_length}")
    print(f"projection tensor: {WEIGHT_NAME}")
    print(f"bias tensor: {BIAS_NAME}")
    print(f"quantization: {quant.description()}")
    print(f"cases: {len(cases)}")
    print(f"avg_max_abs_error: {avg_error:.6f}")
    print(f"worst_max_abs_error: {worst_error:.6f}")
    print()
    print(render_projection_sweep_table(cases))


def run_projection_full_check(config: AppConfig) -> None:
    output_start = 0
    tile_inner = 8
    tile_cols = 3

    executor = IverilogSimExecutor(project_root=config.project_root)
    quant = Q8_8
    model_bin_path = config.model_bin_path()
    model = Ct2ModelBin.open(model_bin_path)
    weight = model.read_tensor_f32(WEIGHT_NAME)
    bias = model.read_tensor_f32(BIAS_NAME)
    output_dim, input_dim = expect_rank2(weight.info.shape, WEIGHT_NAME)
    if bias.info.shape != [output_dim]:
        raise RuntimeError(
            f"bias shape mismatch for {BIAS_NAME}: expected [{output_dim}], got {bias.info.shape!r}"
        )
    if input_dim % tile_inner != 0:
        raise RuntimeError(
            "full projection check expects input dim "
            f"{input_dim} to be divisible by tile width {tile_inner}"
        )
    if output_start + tile_cols > output_dim:
        raise RuntimeError(
            f"requested output window [{output_start}..{output_start + tile_cols}) "
            f"exceeds output dim {output_dim}"
        )

    activation_export_path, activation_export = load_reference_export(config)
    case = run_projection_full_case(
        executor=executor,
        config=config,
        model_bin_path=model_bin_path,
        weight_values=weight.values,
        input_dim=input_dim,
        bias_values=bias.values,
        activation=activation_export.activations[0],
        sequence_index=0,
        output_start=output_start,
        tile_inner=tile_inner,
        tile_cols=tile_cols,
        quant=quant,
    )

    print(f"projection_full_check: matched = {case.matched}")
    print_model_info(model_bin_path, model)
    print(f"reference activation cache: {activation_export_path}")
    print(f"reference audio: {activation_export.audio_path}")
    print(f"reference layer: {activation_export.layer_name}")
    print(f"reference sequence index: {case.sequence_index}")
    print(f"projection tensor: {WEIGHT_NAME}")
    print(f"bias tensor: {BIAS_NAME}")
    print(f"quantization: {quant.description()}")
    print(f"output window: [{case.output_start}..{case.output_start + case.tile_cols})")
    print(f"tile width: {case.tile_inner}")
    print(f"tiles accumulated on FPGA path: {case.tile_count}")
    print(f"bias float: {format_vector_f32(case.bias_float)}")
    print(f"bias quantized: {case.bias_quantized!r}")
    print(f"software raw output: {format_vector_i64(case.software_output)}")
    print(f"rtl raw output: {format_vector_i64(case.rtl_output)}")
    print(f"float reference: {format_vector_f32(case.float_reference)}")
    print(f"rtl dequantized: {format_vector_f32(case.rtl_dequantized)}")
    print(f"max_tile_error: {case.max_tile_error:.6f}")
    print(f"max_abs_error: {case.max_abs_error:.6f}")


def run_projection_full_sweep_check(config: AppConfig) -> None:
    tile_inner = 8
    tile_cols = 8

    executor = IverilogSimExecutor(project_root=config.project_root)
    quant = Q8_8
    model_bin_path = config.model_bin_path()
    model = Ct2ModelBin.open(model_bin_path)
    weight = model.read_tensor_f32(WEIGHT_NAME)
    bias = model.read_tensor_f32(BIAS_NAME)
    output_dim, input_dim = expect_rank2(weight.info.shape, WEIGHT_NAME)
    if bias.info.shape != [output_dim]:
        raise RuntimeError(
            f"bias shape mismatch for {BIAS_NAME}: expected [{output_dim}], got {bias.info.shape!r}"
        )

    activation_export_path, activation_export = load_reference_export(config)
    output_starts = build_output_starts(output_dim, tile_cols)

    cases: list[ProjectionFullCaseResult] = []
    for sequence_index, activation in enumerate(activation_export.activations):
        for output_start in output_starts:
            cases.append(
                run_projection_full_case(
                    executor=executor,
                    config=config,
                    model_bin_path=model_bin_path,
                    weight_values=weight.values,
                    input_dim=input_dim,
                    bias_values=bias.values,
                    activation=activation,
                    sequence_index=sequence_index,
                    output_start=output_start,
                    tile_inner=tile_inner,
                    tile_cols=tile_cols,
                    quant=quant,
                )
            )

    all_matched = all(case.matched for case in cases)
    worst_error = max((case.max_abs_error for case in cases), default=0.0)
    avg_error = sum(case.max_abs_error for case in cases) / len(cases) if cases else 0.0
    max_tile_error = max((case.max_tile_error for case in cases), default=0.0)

    print(f"projection_full_sweep_check: matched = {all_matched}")
    print_model_info(model_bin_path, model)
    print(f"reference activation cache: {activation_export_path}")
    print(f"reference audio: {activation_export.audio_path}")
    print(f"reference layer: {activation_export.layer_name}")
    print(f"reference exported positions: {activation_export.exported_positions}")
    print(f"reference sequence length: {activation_export.sequence_length}")
    print(f"projection tensor: {WEIGHT_NAME}")
    print(f"bias tensor: {BIAS_NAME}")
    print(f"quantization: {quant.description()}")
    print(f"full output tile width: {tile_cols}")
    print(f"cases: {len(cases)}")
    print(f"avg_max_abs_error: {avg_error:.6f}")
    print(f"worst_max_abs_error: {worst_error:.6f}")
    print(f"worst_tile_accum_error: {max_tile_error:.6f}")
    print()
    print(render_projection_full_sweep_table(cases))


def run_gelu_check(config: AppConfig) -> None:
    output_start = 0
    tile_inner = 8
    tile_cols = 8

    executor = IverilogSimExecutor(project_root=config.project_root)
    quant = Q8_8
    model_bin_path = config.model_bin_path()
    model = Ct2ModelBin.open(model_bin_path)
    weight = model.read_tensor_f32(WEIGHT_NAME)
    bias = model.read_tensor_f32(BIAS_NAME)
    _, input_dim = expect_rank2(weight.info.shape, WEIGHT_NAME)

    activation_export_path, activation_export = load_reference_export(config)
    projection_case = run_projection_full_case(
        executor=executor,
        config=config,
        model_bin_path=model_bin_path,
        weight_values=weight.values,
        input_dim=input_dim,
        bias_values=bias.values,
        activation=activation_export.activations[0],
        sequence_index=0,
        output_start=output_start,
        tile_inner=tile_inner,
        tile_cols=tile_cols,
        quant=quant,
    )
    gelu_input = quant.requantize_accumulator_slice(projection_case.rtl_output)
    comparison = simulate_gelu_block(
        executor=executor,
        output_dir=config.resolved_fpga_sim_io_dir,
        audio_path=str(model_bin_path),
        input_block=gelu_input,
        float_input=projection_case.float_reference,
        quant=quant,
    )

    print(f"gelu_check: matched = {comparison.matched}")
    print_model_info(model_bin_path, model)
    print(f"reference activation cache: {activation_export_path}")
    print(f"reference audio: {activation_export.audio_path}")
    print(f"reference layer: {activation_export.layer_name}")
    print("reference sequence index: 0")
    print(f"projection tensor: {WEIGHT_NAME}")
    print(f"bias tensor: {BIAS_NAME}")
    print(f"quantization: {quant.description()}")
    print(f"output window: [{output_start}..{output_start + tile_cols})")
    print(f"projection tiles accumulated: {projection_case.tile_count}")
    print(
        f"projection float input: {format_vector_f32(projection_case.float_reference)}"
    )
    print(f"projection q8.8 input: {format_vector_i16(gelu_input)}")
    print(f"software gelu q8.8 output: {format_vector_i16(comparison.software_output)}")
    print(f"rtl gelu q8.8 output: {format_vector_i16(comparison.rtl_output)}")
    print(f"float gelu reference: {format_vector_f32(comparison.float_reference)}")
    print(f"rtl gelu dequantized: {format_vector_f32(comparison.rtl_dequantized)}")
    print(f"max_abs_error: {comparison.max_abs_error:.6f}")
    print("notes:")
    for note in dedupe_notes(comparison.notes):
        print(f"- {note}")


def run_gelu_sweep_check(config: AppConfig) -> None:
    tile_inner = 8
    tile_cols = 8

    executor = IverilogSimExecutor(project_root=config.project_root)
    quant = Q8_8
    model_bin_path = config.model_bin_path()
    model = Ct2ModelBin.open(model_bin_path)
    weight = model.read_tensor_f32(WEIGHT_NAME)
    bias = model.read_tensor_f32(BIAS_NAME)
    output_dim, input_dim = expect_rank2(weight.info.shape, WEIGHT_NAME)

    activation_export_path, activation_export = load_reference_export(config)
    output_starts = build_output_starts(output_dim, tile_cols)

    cases: list[GeluCaseResult] = []
    for sequence_index, activation in enumerate(activation_export.activations):
        for output_start in output_starts:
            projection_case = run_projection_full_case(
                executor=executor,
                config=config,
                model_bin_path=model_bin_path,
                weight_values=weight.values,
                input_dim=input_dim,
                bias_values=bias.values,
                activation=activation,
                sequence_index=sequence_index,
                output_start=output_start,
                tile_inner=tile_inner,
                tile_cols=tile_cols,
                quant=quant,
            )
            gelu_input = quant.requantize_accumulator_slice(projection_case.rtl_output)
            comparison = simulate_gelu_block(
                executor=executor,
                output_dir=config.resolved_fpga_sim_io_dir,
                audio_path=str(model_bin_path),
                input_block=gelu_input,
                float_input=projection_case.float_reference,
                quant=quant,
            )
            cases.append(
                GeluCaseResult(
                    sequence_index=sequence_index,
                    output_start=output_start,
                    tile_cols=tile_cols,
                    projection_tiles=projection_case.tile_count,
                    projection_max_abs_error=projection_case.max_abs_error,
                    comparison=comparison,
                )
            )

    all_matched = all(case.comparison.matched for case in cases)
    worst_error = max((case.comparison.max_abs_error for case in cases), default=0.0)
    avg_error = (
        sum(case.comparison.max_abs_error for case in cases) / len(cases)
        if cases
        else 0.0
    )

    print(f"gelu_sweep_check: matched = {all_matched}")
    print_model_info(model_bin_path, model)
    print(f"reference activation cache: {activation_export_path}")
    print(f"reference audio: {activation_export.audio_path}")
    print(f"reference layer: {activation_export.layer_name}")
    print(f"reference exported positions: {activation_export.exported_positions}")
    print(f"reference sequence length: {activation_export.sequence_length}")
    print(f"projection tensor: {WEIGHT_NAME}")
    print(f"bias tensor: {BIAS_NAME}")
    print(f"quantization: {quant.description()}")
    print(f"gelu block width: {tile_cols}")
    print(f"cases: {len(cases)}")
    print(f"avg_max_abs_error: {avg_error:.6f}")
    print(f"worst_max_abs_error: {worst_error:.6f}")
    print()
    print(render_gelu_sweep_table(cases))


def run_projection_case(
    *,
    executor: IverilogSimExecutor,
    config: AppConfig,
    model_bin_path: Path,
    weight_values: list[float],
    input_dim: int,
    bias_values: list[float],
    activation: list[float],
    sequence_index: int,
    input_start: int,
    output_start: int,
    tile_inner: int,
    tile_cols: int,
    quant: FixedPointConfig,
) -> ProjectionCaseResult:
    if len(activation) < input_start + tile_inner:
        raise RuntimeError(
            "reference activation for sequence "
            f"{sequence_index} is too short: need at least "
            f"{input_start + tile_inner}, got {len(activation)}"
        )

    input_float = activation[input_start : input_start + tile_inner]
    input_quantized = quant.quantize_slice(input_float)

    weight_tile_float: list[float] = []
    for input_offset in range(tile_inner):
        for output_offset in range(tile_cols):
            output_index = output_start + output_offset
            input_index = input_start + input_offset
            weight_tile_float.append(
                weight_values[output_index * input_dim + input_index]
            )

    weight_tile_quantized = quant.quantize_slice(weight_tile_float)
    bias_float = bias_values[output_start : output_start + tile_cols]
    bias_quantized = quant.quantize_slice(bias_float)

    layer = LinearLayerI16(
        input_dim=tile_inner,
        output_dim=tile_cols,
        weights=MatrixI16(
            rows=tile_inner, cols=tile_cols, values=weight_tile_quantized
        ),
        bias=list(bias_quantized),
        quant=quant,
    )

    comparison = simulate_linear(
        executor=executor,
        output_dir=config.resolved_fpga_sim_io_dir,
        audio_path=str(model_bin_path),
        layer=layer,
        input_values=input_quantized,
    )

    float_reference = compute_float_projection(
        input_values=input_float,
        weight_values=weight_values,
        input_dim=input_dim,
        bias_values=bias_values,
        input_start=input_start,
        output_start=output_start,
        tile_inner=tile_inner,
        tile_cols=tile_cols,
    )
    rtl_dequantized = dequantize_outputs(quant, comparison.rtl_output)
    max_abs_error = max_abs_diff(float_reference, rtl_dequantized)

    return ProjectionCaseResult(
        sequence_index=sequence_index,
        input_start=input_start,
        output_start=output_start,
        input_float=list(input_float),
        input_quantized=list(input_quantized),
        bias_float=list(bias_float),
        bias_quantized=list(bias_quantized),
        float_reference=float_reference,
        rtl_dequantized=rtl_dequantized,
        max_abs_error=max_abs_error,
        matched=comparison.matched,
        comparison=comparison,
    )


def run_projection_full_case(
    *,
    executor: IverilogSimExecutor,
    config: AppConfig,
    model_bin_path: Path,
    weight_values: list[float],
    input_dim: int,
    bias_values: list[float],
    activation: list[float],
    sequence_index: int,
    output_start: int,
    tile_inner: int,
    tile_cols: int,
    quant: FixedPointConfig,
) -> ProjectionFullCaseResult:
    if len(activation) != input_dim:
        raise RuntimeError(
            "reference activation width mismatch for sequence "
            f"{sequence_index}: expected {input_dim}, got {len(activation)}"
        )
    if input_dim % tile_inner != 0:
        raise RuntimeError(
            "full projection case expects input dim "
            f"{input_dim} to be divisible by tile width {tile_inner}"
        )

    bias_float = bias_values[output_start : output_start + tile_cols]
    bias_quantized = quant.quantize_slice(bias_float)
    software_accumulator = MatrixI64(
        rows=1,
        cols=tile_cols,
        values=[quant.bias_to_accumulator(value) for value in bias_quantized],
    )
    rtl_accumulator = MatrixI64(
        rows=software_accumulator.rows,
        cols=software_accumulator.cols,
        values=list(software_accumulator.values),
    )
    batch_requests: list[GemmTileI16Request] = []
    max_tile_error = 0.0

    for input_start in range(0, input_dim, tile_inner):
        input_float = activation[input_start : input_start + tile_inner]
        input_quantized = quant.quantize_slice(input_float)

        weight_tile_float: list[float] = []
        for input_offset in range(tile_inner):
            for output_offset in range(tile_cols):
                output_index = output_start + output_offset
                input_index = input_start + input_offset
                weight_tile_float.append(
                    weight_values[output_index * input_dim + input_index]
                )
        weight_tile_quantized = quant.quantize_slice(weight_tile_float)
        lhs = MatrixI16(rows=1, cols=tile_inner, values=input_quantized)
        rhs = MatrixI16(rows=tile_inner, cols=tile_cols, values=weight_tile_quantized)
        software_accumulator = software_gemm_with_accumulator(
            lhs,
            rhs,
            software_accumulator,
        )
        batch_requests.append(
            GemmTileI16Request(
                audio_path=str(model_bin_path),
                shape=GemmTileShape(rows=1, cols=tile_cols, inner=tile_inner),
                lhs_tile=list(lhs.values),
                rhs_tile=list(rhs.values),
                accumulator_input=list(rtl_accumulator.values),
                expected_output=list(software_accumulator.values),
            )
        )
        rtl_accumulator = MatrixI64(
            rows=rtl_accumulator.rows,
            cols=rtl_accumulator.cols,
            values=list(software_accumulator.values),
        )

    batch_responses = executor.execute_gemm_tile_batch(
        GemmTileBatchI16Request(
            shape=GemmTileShape(rows=1, cols=tile_cols, inner=tile_inner),
            requests=batch_requests,
        ),
        config.resolved_fpga_sim_io_dir,
    )
    for response in batch_responses:
        tile_rtl_dequantized = dequantize_outputs(quant, response.rtl_output)
        tile_software_dequantized = dequantize_outputs(quant, response.expected_output)
        tile_error = max_abs_diff(tile_software_dequantized, tile_rtl_dequantized)
        max_tile_error = max(max_tile_error, tile_error)
    rtl_accumulator = MatrixI64(
        rows=1,
        cols=tile_cols,
        values=list(batch_responses[-1].rtl_output),
    )

    float_reference = compute_float_projection(
        input_values=activation,
        weight_values=weight_values,
        input_dim=input_dim,
        bias_values=bias_values,
        input_start=0,
        output_start=output_start,
        tile_inner=input_dim,
        tile_cols=tile_cols,
    )
    rtl_dequantized = dequantize_outputs(quant, rtl_accumulator.values)
    max_abs_error = max_abs_diff(float_reference, rtl_dequantized)
    matched = all(response.matched for response in batch_responses) and (
        software_accumulator.values == rtl_accumulator.values
    )

    return ProjectionFullCaseResult(
        sequence_index=sequence_index,
        output_start=output_start,
        tile_inner=tile_inner,
        tile_cols=tile_cols,
        tile_count=len(batch_requests),
        bias_float=list(bias_float),
        bias_quantized=list(bias_quantized),
        software_output=list(software_accumulator.values),
        rtl_output=list(rtl_accumulator.values),
        float_reference=float_reference,
        rtl_dequantized=rtl_dequantized,
        max_tile_error=max_tile_error,
        max_abs_error=max_abs_error,
        matched=matched,
    )


def print_matrix_i64(matrix: MatrixI64) -> None:
    for row in range(matrix.rows):
        values = [str(matrix.get(row, col)) for col in range(matrix.cols)]
        print(f"[{', '.join(values)}]")


def format_vector_f32(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def format_vector_i16(values: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def dequantize_outputs(quant: FixedPointConfig, values: list[int]) -> list[float]:
    return [quant.dequantize_accumulator(value) for value in values]


def compute_float_projection(
    *,
    input_values: list[float],
    weight_values: list[float],
    input_dim: int,
    bias_values: list[float],
    input_start: int,
    output_start: int,
    tile_inner: int,
    tile_cols: int,
) -> list[float]:
    outputs: list[float] = []
    for output_offset in range(tile_cols):
        output_index = output_start + output_offset
        dot = 0.0
        for input_offset in range(tile_inner):
            input_index = input_start + input_offset
            dot += (
                input_values[input_offset]
                * weight_values[output_index * input_dim + input_index]
            )
        outputs.append(dot + bias_values[output_index])
    return outputs


def build_output_starts(output_dim: int, tile_cols: int) -> list[int]:
    starts = [0, 64, 256, output_dim // 2, max(output_dim - tile_cols, 0)]
    starts = sorted(set(starts))
    return [start for start in starts if start + tile_cols <= output_dim]


def build_input_starts(input_dim: int, tile_inner: int) -> list[int]:
    starts = [0, 64, 256, input_dim // 2, max(input_dim - tile_inner, 0)]
    starts = sorted(set(starts))
    return [start for start in starts if start + tile_inner <= input_dim]


def render_projection_sweep_table(cases: list[ProjectionCaseResult]) -> str:
    headers = ["seq", "in", "out", "matched", "max_abs_err", "float_ref", "rtl"]
    rows = [
        [
            str(case.sequence_index),
            str(case.input_start),
            str(case.output_start),
            str(case.matched),
            f"{case.max_abs_error:.6f}",
            format_vector_f32(case.float_reference),
            format_vector_f32(case.rtl_dequantized),
        ]
        for case in cases
    ]
    return render_table(headers, rows)


def render_projection_full_sweep_table(cases: list[ProjectionFullCaseResult]) -> str:
    headers = [
        "seq",
        "out",
        "cols",
        "tiles",
        "matched",
        "max_abs_err",
        "float_ref",
        "rtl",
    ]
    rows = [
        [
            str(case.sequence_index),
            str(case.output_start),
            str(case.tile_cols),
            str(case.tile_count),
            str(case.matched),
            f"{case.max_abs_error:.6f}",
            format_vector_f32(case.float_reference),
            format_vector_f32(case.rtl_dequantized),
        ]
        for case in cases
    ]
    return render_table(headers, rows)


def render_gelu_sweep_table(cases: list[GeluCaseResult]) -> str:
    headers = [
        "seq",
        "out",
        "cols",
        "proj_tiles",
        "proj_err",
        "gelu_err",
        "matched",
        "gelu_ref",
        "gelu_rtl",
    ]
    rows = [
        [
            str(case.sequence_index),
            str(case.output_start),
            str(case.tile_cols),
            str(case.projection_tiles),
            f"{case.projection_max_abs_error:.6f}",
            f"{case.comparison.max_abs_error:.6f}",
            str(case.comparison.matched),
            format_vector_f32(case.comparison.float_reference),
            format_vector_f32(case.comparison.rtl_dequantized),
        ]
        for case in cases
    ]
    return render_table(headers, rows)


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def expect_rank2(shape: list[int], tensor_name: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise RuntimeError(f"expected rank-2 tensor for {tensor_name}")
    return shape[0], shape[1]


def load_reference_export(config: AppConfig):
    reference_audio = config.resolve_project_path(REFERENCE_AUDIO)
    activation_export_path = ensure_reference_activation_export(
        config,
        reference_audio,
        False,
    )
    activation_export = load_reference_activation(activation_export_path)
    return activation_export_path, activation_export


def print_model_info(model_bin_path: Path, model: Ct2ModelBin) -> None:
    print(
        f"model_bin: {model_bin_path} "
        f"(spec {model.spec_name} v{model.version} rev {model.revision})"
    )


def max_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    return max((abs(a - b) for a, b in zip(lhs, rhs, strict=False)), default=0.0)


def dedupe_notes(notes: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for note in notes:
        if note in seen:
            continue
        seen.add(note)
        deduped.append(note)
    return deduped
