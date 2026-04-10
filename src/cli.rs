use anyhow::Result;
use clap::{Parser, Subcommand};
use comfy_table::{Cell, Table, presets::UTF8_FULL};
use std::time::Instant;

use crate::backend::{build_backend, describe_backend};
use crate::config::AppConfig;
use crate::fpga::kernels::gemm::{
    MatrixI16, MatrixI64, simulate_gemm_tile, simulate_gemm_tile_with_accumulator,
};
use crate::fpga::kernels::linear::{LinearLayerI16, format_vector_i64, simulate_linear};
use crate::fpga::quant::FixedPointConfig;
use crate::fpga::sim::IverilogSimExecutor;
use crate::model::ct2::Ct2ModelBin;
use crate::model::reference::{ensure_reference_activation_export, load_reference_activation};
use crate::profiling::{profile_request, render_samples_table, render_summary_table};
use crate::tui::run_tui;
use crate::types::{BackendKind, BenchmarkRun, PartitionPreset, Transcript, TranscriptionRequest};

#[derive(Debug, Parser)]
#[command(name = "fpga-whisper")]
#[command(about = "Rust frontend for a hybrid Whisper + FPGA transcription project.")]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Plan {
        #[arg(long, value_enum, default_value_t = BackendKind::FpgaHybrid)]
        backend: BackendKind,
        #[arg(long, value_enum, default_value_t = PartitionPreset::Hybrid)]
        partition: PartitionPreset,
    },
    Transcribe {
        audio: std::path::PathBuf,
        #[arg(long, value_enum, default_value_t = BackendKind::Ct2Python)]
        backend: BackendKind,
        #[arg(long, value_enum, default_value_t = PartitionPreset::Hybrid)]
        partition: PartitionPreset,
        #[arg(long)]
        initial_prompt: Option<String>,
    },
    Benchmark {
        audio: std::path::PathBuf,
        #[arg(long, value_enum, default_value_t = BackendKind::Ct2Python)]
        backend: BackendKind,
        #[arg(long, value_enum, default_value_t = PartitionPreset::Hybrid)]
        partition: PartitionPreset,
        #[arg(long, default_value_t = 3)]
        iterations: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long)]
        initial_prompt: Option<String>,
    },
    Profile {
        audio: std::path::PathBuf,
        #[arg(long, value_enum, default_value_t = BackendKind::Ct2Python)]
        backend: BackendKind,
        #[arg(long, value_enum, default_value_t = PartitionPreset::Hybrid)]
        partition: PartitionPreset,
        #[arg(long, default_value_t = 250)]
        sample_interval_ms: u64,
        #[arg(long)]
        initial_prompt: Option<String>,
    },
    GemmCheck,
    LinearCheck,
    ProjectionTileCheck,
    ProjectionSweepCheck,
    ProjectionFullCheck,
    Tui,
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let config = AppConfig::default();

    match cli.command {
        Commands::Plan { backend, partition } => {
            let descriptor = describe_backend(backend);
            println!("{descriptor}");
            println!();
            println!("Selected partition preset: {partition:?}");
            println!("Partition summary: {}", partition.summary());
        }
        Commands::Transcribe {
            audio,
            backend,
            partition,
            initial_prompt,
        } => {
            let request = TranscriptionRequest {
                audio_path: audio,
                backend,
                partition,
                initial_prompt,
            };

            let backend = build_backend(backend, &config);
            let transcript = backend.transcribe(&request)?;
            print_transcript(&transcript);
        }
        Commands::Benchmark {
            audio,
            backend,
            partition,
            iterations,
            warmup,
            initial_prompt,
        } => {
            let request = TranscriptionRequest {
                audio_path: audio,
                backend,
                partition,
                initial_prompt,
            };

            let backend = build_backend(backend, &config);
            let report = benchmark_backend(backend.as_ref(), &request, iterations, warmup)?;
            print_benchmark(&report);
        }
        Commands::Profile {
            audio,
            backend,
            partition,
            sample_interval_ms,
            initial_prompt,
        } => {
            let request = TranscriptionRequest {
                audio_path: audio,
                backend,
                partition,
                initial_prompt,
            };
            let report = profile_request(
                &config,
                &request,
                std::time::Duration::from_millis(sample_interval_ms),
            )?;
            println!("{}", render_summary_table(&report));
            println!();
            println!("{}", render_samples_table(&report));
            println!();
            print_transcript(&report.transcript);
        }
        Commands::GemmCheck => {
            run_gemm_check(&config)?;
        }
        Commands::LinearCheck => {
            run_linear_check(&config)?;
        }
        Commands::ProjectionTileCheck => {
            run_projection_tile_check(&config)?;
        }
        Commands::ProjectionSweepCheck => {
            run_projection_sweep_check(&config)?;
        }
        Commands::ProjectionFullCheck => {
            run_projection_full_check(&config)?;
        }
        Commands::Tui => run_tui(config)?,
    }

    Ok(())
}

fn run_gemm_check(config: &AppConfig) -> Result<()> {
    let executor = IverilogSimExecutor::new(config.project_root.clone());
    let lhs = MatrixI16::new(
        2,
        8,
        vec![
            3, -2, 7, 4, -1, 5, 2, -3, //
            1, 6, -5, 2, 8, -4, 3, 7,
        ],
    );
    let rhs = MatrixI16::new(
        8,
        2,
        vec![
            6, 1, //
            8, -3, //
            -4, 2, //
            1, 9, //
            9, -1, //
            -2, 5, //
            3, 4, //
            5, -6,
        ],
    );

    let comparison = simulate_gemm_tile(
        &executor,
        &config.fpga_sim_io_dir,
        "samples/jfk.flac",
        &lhs,
        &rhs,
    )?;

    println!("gemm_check: matched = {}", comparison.matched);
    println!("software result:");
    print_matrix_i64(&comparison.software);
    println!("rtl result:");
    print_matrix_i64(&comparison.rtl);
    println!("notes:");
    let mut deduped = std::collections::BTreeSet::<String>::new();
    for note in comparison.notes {
        if !deduped.insert(note.clone()) {
            continue;
        }
        println!("- {note}");
    }

    Ok(())
}

fn run_linear_check(config: &AppConfig) -> Result<()> {
    let executor = IverilogSimExecutor::new(config.project_root.clone());
    let layer = LinearLayerI16 {
        input_dim: 8,
        output_dim: 3,
        weights: MatrixI16::new(
            8,
            3,
            vec![
                6, 1, -2, //
                8, -3, 4, //
                -4, 2, 5, //
                1, 9, -1, //
                9, -1, 3, //
                -2, 5, 7, //
                3, 4, -6, //
                5, -6, 2,
            ],
        ),
        bias: vec![4, -3, 9],
        quant: FixedPointConfig::Q8_8,
    };
    let input = vec![3, -2, 7, 4, -1, 5, 2, -3];

    let comparison = simulate_linear(
        &executor,
        &config.fpga_sim_io_dir,
        "samples/jfk.flac",
        &layer,
        &input,
    )?;

    println!("linear_check: matched = {}", comparison.matched);
    println!("linear_check: quantization = {}", layer.quant.description());
    println!("input: {:?}", input);
    println!(
        "software raw output: {}",
        format_vector_i64(&comparison.software_output)
    );
    println!(
        "rtl raw output: {}",
        format_vector_i64(&comparison.rtl_output)
    );
    println!(
        "software dequantized: {}",
        format_vector_f32(&dequantize_outputs(
            layer.quant,
            &comparison.software_output
        ))
    );
    println!(
        "rtl dequantized: {}",
        format_vector_f32(&dequantize_outputs(layer.quant, &comparison.rtl_output))
    );
    println!("gemm tile output:");
    print_matrix_i64(&comparison.gemm.rtl);
    println!("notes:");
    let mut deduped = std::collections::BTreeSet::<String>::new();
    for note in comparison.notes {
        if !deduped.insert(note.clone()) {
            continue;
        }
        println!("- {note}");
    }

    Ok(())
}

fn run_projection_tile_check(config: &AppConfig) -> Result<()> {
    const WEIGHT_NAME: &str = "encoder/layer_0/ffn/linear_0/weight";
    const BIAS_NAME: &str = "encoder/layer_0/ffn/linear_0/bias";
    const INPUT_START: usize = 0;
    const OUTPUT_START: usize = 0;
    const TILE_INNER: usize = 8;
    const TILE_COLS: usize = 3;

    let executor = IverilogSimExecutor::new(config.project_root.clone());
    let quant = FixedPointConfig::Q8_8;
    let model_bin_path = config.model_bin_path()?;
    let model = Ct2ModelBin::open(&model_bin_path)?;
    let weight = model.read_tensor_f32(WEIGHT_NAME)?;
    let bias = model.read_tensor_f32(BIAS_NAME)?;

    let [output_dim, input_dim]: [usize; 2] = weight
        .info
        .shape
        .clone()
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected rank-2 tensor for {WEIGHT_NAME}"))?;
    if bias.info.shape != vec![output_dim] {
        anyhow::bail!(
            "bias shape mismatch for {BIAS_NAME}: expected [{}], got {:?}",
            output_dim,
            bias.info.shape
        );
    }
    if INPUT_START + TILE_INNER > input_dim || OUTPUT_START + TILE_COLS > output_dim {
        anyhow::bail!(
            "requested tile [{}..{}) x [{}..{}) exceeds tensor shape {}x{}",
            OUTPUT_START,
            OUTPUT_START + TILE_COLS,
            INPUT_START,
            INPUT_START + TILE_INNER,
            output_dim,
            input_dim
        );
    }

    let reference_audio = config.project_root.join("samples/jfk.flac");
    let activation_export_path =
        ensure_reference_activation_export(config, &reference_audio, false)?;
    let activation_export = load_reference_activation(&activation_export_path)?;
    let case = run_projection_case(
        &executor,
        config,
        &model_bin_path,
        &weight.values,
        input_dim,
        &bias.values,
        &activation_export.activations[0],
        0,
        INPUT_START,
        OUTPUT_START,
        TILE_INNER,
        TILE_COLS,
        quant,
    )?;

    println!("projection_tile_check: matched = {}", case.matched);
    println!(
        "model_bin: {} (spec {} v{} rev {})",
        model_bin_path.display(),
        model.spec_name,
        model.version,
        model.revision
    );
    println!(
        "reference activation cache: {}",
        activation_export_path.display()
    );
    println!("reference audio: {}", activation_export.audio_path);
    println!("reference layer: {}", activation_export.layer_name);
    println!(
        "reference exported positions: {}",
        activation_export.exported_positions
    );
    println!(
        "reference sequence length: {}",
        activation_export.sequence_length
    );
    println!("reference sequence index: {}", case.sequence_index);
    println!("projection tensor: {WEIGHT_NAME}");
    println!("bias tensor: {BIAS_NAME}");
    println!(
        "tile: outputs [{OUTPUT_START}..{}), inputs [{INPUT_START}..{})",
        OUTPUT_START + TILE_COLS,
        INPUT_START + TILE_INNER
    );
    println!("quantization: {}", quant.description());
    println!("input float: {}", format_vector_f32(&case.input_float));
    println!("input quantized: {:?}", case.input_quantized);
    println!("bias float: {}", format_vector_f32(&case.bias_float));
    println!("bias quantized: {:?}", case.bias_quantized);
    println!(
        "software raw output: {}",
        format_vector_i64(&case.comparison.software_output)
    );
    println!(
        "rtl raw output: {}",
        format_vector_i64(&case.comparison.rtl_output)
    );
    println!(
        "float reference: {}",
        format_vector_f32(&case.float_reference)
    );
    println!(
        "rtl dequantized: {}",
        format_vector_f32(&case.rtl_dequantized)
    );
    println!("max_abs_error: {:.6}", case.max_abs_error);
    println!("notes:");
    let mut deduped = std::collections::BTreeSet::<String>::new();
    for note in case.comparison.notes {
        if !deduped.insert(note.clone()) {
            continue;
        }
        println!("- {note}");
    }

    Ok(())
}

fn run_projection_sweep_check(config: &AppConfig) -> Result<()> {
    const WEIGHT_NAME: &str = "encoder/layer_0/ffn/linear_0/weight";
    const BIAS_NAME: &str = "encoder/layer_0/ffn/linear_0/bias";
    const TILE_INNER: usize = 8;
    const TILE_COLS: usize = 3;

    let executor = IverilogSimExecutor::new(config.project_root.clone());
    let quant = FixedPointConfig::Q8_8;
    let model_bin_path = config.model_bin_path()?;
    let model = Ct2ModelBin::open(&model_bin_path)?;
    let weight = model.read_tensor_f32(WEIGHT_NAME)?;
    let bias = model.read_tensor_f32(BIAS_NAME)?;
    let [output_dim, input_dim]: [usize; 2] = weight
        .info
        .shape
        .clone()
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected rank-2 tensor for {WEIGHT_NAME}"))?;
    if bias.info.shape != vec![output_dim] {
        anyhow::bail!(
            "bias shape mismatch for {BIAS_NAME}: expected [{}], got {:?}",
            output_dim,
            bias.info.shape
        );
    }

    let reference_audio = config.project_root.join("samples/jfk.flac");
    let activation_export_path =
        ensure_reference_activation_export(config, &reference_audio, false)?;
    let activation_export = load_reference_activation(&activation_export_path)?;

    let output_starts = build_output_starts(output_dim, TILE_COLS);
    let input_starts = build_input_starts(input_dim, TILE_INNER);
    let mut cases = Vec::new();
    for (sequence_index, activation) in activation_export.activations.iter().enumerate() {
        for input_start in &input_starts {
            for output_start in &output_starts {
                cases.push(run_projection_case(
                    &executor,
                    config,
                    &model_bin_path,
                    &weight.values,
                    input_dim,
                    &bias.values,
                    activation,
                    sequence_index,
                    *input_start,
                    *output_start,
                    TILE_INNER,
                    TILE_COLS,
                    quant,
                )?);
            }
        }
    }

    let all_matched = cases.iter().all(|case| case.matched);
    let worst_error = cases
        .iter()
        .map(|case| case.max_abs_error)
        .fold(0.0_f32, f32::max);
    let avg_error = cases.iter().map(|case| case.max_abs_error).sum::<f32>() / cases.len() as f32;

    println!("projection_sweep_check: matched = {all_matched}");
    println!(
        "model_bin: {} (spec {} v{} rev {})",
        model_bin_path.display(),
        model.spec_name,
        model.version,
        model.revision
    );
    println!(
        "reference activation cache: {}",
        activation_export_path.display()
    );
    println!("reference audio: {}", activation_export.audio_path);
    println!("reference layer: {}", activation_export.layer_name);
    println!(
        "reference exported positions: {}",
        activation_export.exported_positions
    );
    println!(
        "reference sequence length: {}",
        activation_export.sequence_length
    );
    println!("projection tensor: {WEIGHT_NAME}");
    println!("bias tensor: {BIAS_NAME}");
    println!("quantization: {}", quant.description());
    println!("cases: {}", cases.len());
    println!("avg_max_abs_error: {:.6}", avg_error);
    println!("worst_max_abs_error: {:.6}", worst_error);
    println!();
    println!("{}", render_projection_sweep_table(&cases));

    Ok(())
}

fn run_projection_full_check(config: &AppConfig) -> Result<()> {
    const WEIGHT_NAME: &str = "encoder/layer_0/ffn/linear_0/weight";
    const BIAS_NAME: &str = "encoder/layer_0/ffn/linear_0/bias";
    const OUTPUT_START: usize = 0;
    const TILE_INNER: usize = 8;
    const TILE_COLS: usize = 3;

    let executor = IverilogSimExecutor::new(config.project_root.clone());
    let quant = FixedPointConfig::Q8_8;
    let model_bin_path = config.model_bin_path()?;
    let model = Ct2ModelBin::open(&model_bin_path)?;
    let weight = model.read_tensor_f32(WEIGHT_NAME)?;
    let bias = model.read_tensor_f32(BIAS_NAME)?;
    let [output_dim, input_dim]: [usize; 2] = weight
        .info
        .shape
        .clone()
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected rank-2 tensor for {WEIGHT_NAME}"))?;
    if bias.info.shape != vec![output_dim] {
        anyhow::bail!(
            "bias shape mismatch for {BIAS_NAME}: expected [{}], got {:?}",
            output_dim,
            bias.info.shape
        );
    }
    if input_dim % TILE_INNER != 0 {
        anyhow::bail!(
            "full projection check expects input dim {} to be divisible by tile width {}",
            input_dim,
            TILE_INNER
        );
    }
    if OUTPUT_START + TILE_COLS > output_dim {
        anyhow::bail!(
            "requested output window [{}..{}) exceeds output dim {}",
            OUTPUT_START,
            OUTPUT_START + TILE_COLS,
            output_dim
        );
    }

    let reference_audio = config.project_root.join("samples/jfk.flac");
    let activation_export_path =
        ensure_reference_activation_export(config, &reference_audio, false)?;
    let activation_export = load_reference_activation(&activation_export_path)?;
    let activation = &activation_export.activations[0];
    if activation.len() != input_dim {
        anyhow::bail!(
            "reference activation width mismatch: expected {}, got {}",
            input_dim,
            activation.len()
        );
    }

    let bias_float = bias.values[OUTPUT_START..OUTPUT_START + TILE_COLS].to_vec();
    let bias_quantized = quant.quantize_slice(&bias_float);
    let mut software_accumulator = MatrixI64::new(
        1,
        TILE_COLS,
        bias_quantized
            .iter()
            .map(|bias| quant.bias_to_accumulator(*bias))
            .collect(),
    );
    let mut rtl_accumulator = software_accumulator.clone();
    let mut max_tile_error = 0.0_f32;
    let mut tile_count = 0usize;

    for input_start in (0..input_dim).step_by(TILE_INNER) {
        let input_float = activation[input_start..input_start + TILE_INNER].to_vec();
        let input_quantized = quant.quantize_slice(&input_float);

        let mut weight_tile_float = Vec::with_capacity(TILE_INNER * TILE_COLS);
        for input_offset in 0..TILE_INNER {
            for output_offset in 0..TILE_COLS {
                let output_index = OUTPUT_START + output_offset;
                let input_index = input_start + input_offset;
                weight_tile_float.push(weight.values[output_index * input_dim + input_index]);
            }
        }
        let weight_tile_quantized = quant.quantize_slice(&weight_tile_float);

        let comparison = simulate_gemm_tile_with_accumulator(
            &executor,
            &config.fpga_sim_io_dir,
            &model_bin_path.display().to_string(),
            &MatrixI16::new(1, TILE_INNER, input_quantized),
            &MatrixI16::new(TILE_INNER, TILE_COLS, weight_tile_quantized),
            Some(&software_accumulator),
        )?;

        let tile_rtl_dequantized = dequantize_outputs(quant, &comparison.rtl.values);
        let tile_software_dequantized = dequantize_outputs(quant, &comparison.software.values);
        let tile_error = tile_software_dequantized
            .iter()
            .zip(&tile_rtl_dequantized)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        max_tile_error = max_tile_error.max(tile_error);
        software_accumulator = comparison.software.clone();
        rtl_accumulator = comparison.rtl;
        tile_count += 1;
    }

    let float_reference = compute_float_projection(
        activation,
        &weight.values,
        input_dim,
        &bias.values,
        0,
        OUTPUT_START,
        input_dim,
        TILE_COLS,
    );
    let rtl_dequantized = dequantize_outputs(quant, &rtl_accumulator.values);
    let max_abs_error = float_reference
        .iter()
        .zip(&rtl_dequantized)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    let matched = software_accumulator.values == rtl_accumulator.values;

    println!("projection_full_check: matched = {matched}");
    println!(
        "model_bin: {} (spec {} v{} rev {})",
        model_bin_path.display(),
        model.spec_name,
        model.version,
        model.revision
    );
    println!(
        "reference activation cache: {}",
        activation_export_path.display()
    );
    println!("reference audio: {}", activation_export.audio_path);
    println!("reference layer: {}", activation_export.layer_name);
    println!("reference sequence index: 0");
    println!("projection tensor: {WEIGHT_NAME}");
    println!("bias tensor: {BIAS_NAME}");
    println!("quantization: {}", quant.description());
    println!(
        "output window: [{OUTPUT_START}..{})",
        OUTPUT_START + TILE_COLS
    );
    println!("tile width: {TILE_INNER}");
    println!("tiles accumulated on FPGA path: {tile_count}");
    println!("bias float: {}", format_vector_f32(&bias_float));
    println!("bias quantized: {:?}", bias_quantized);
    println!(
        "software raw output: {}",
        format_vector_i64(&software_accumulator.values)
    );
    println!(
        "rtl raw output: {}",
        format_vector_i64(&rtl_accumulator.values)
    );
    println!("float reference: {}", format_vector_f32(&float_reference));
    println!("rtl dequantized: {}", format_vector_f32(&rtl_dequantized));
    println!("max_tile_error: {:.6}", max_tile_error);
    println!("max_abs_error: {:.6}", max_abs_error);

    Ok(())
}

fn print_matrix_i64(matrix: &MatrixI64) {
    for row in 0..matrix.rows {
        let values = (0..matrix.cols)
            .map(|col| matrix.get(row, col).to_string())
            .collect::<Vec<_>>();
        println!("[{}]", values.join(", "));
    }
}

fn format_vector_f32(values: &[f32]) -> String {
    let parts = values
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>();
    format!("[{}]", parts.join(", "))
}

fn dequantize_outputs(quant: FixedPointConfig, values: &[i64]) -> Vec<f32> {
    values
        .iter()
        .map(|value| quant.dequantize_accumulator(*value))
        .collect()
}

fn compute_float_projection(
    input: &[f32],
    weight_values: &[f32],
    input_dim: usize,
    bias_values: &[f32],
    input_start: usize,
    output_start: usize,
    tile_inner: usize,
    tile_cols: usize,
) -> Vec<f32> {
    (0..tile_cols)
        .map(|output_offset| {
            let output_index = output_start + output_offset;
            let dot = (0..tile_inner)
                .map(|input_offset| {
                    let input_index = input_start + input_offset;
                    input[input_offset] * weight_values[output_index * input_dim + input_index]
                })
                .sum::<f32>();
            dot + bias_values[output_index]
        })
        .collect()
}

#[derive(Debug)]
struct ProjectionCaseResult {
    sequence_index: usize,
    input_start: usize,
    output_start: usize,
    input_float: Vec<f32>,
    input_quantized: Vec<i16>,
    bias_float: Vec<f32>,
    bias_quantized: Vec<i16>,
    float_reference: Vec<f32>,
    rtl_dequantized: Vec<f32>,
    max_abs_error: f32,
    matched: bool,
    comparison: crate::fpga::kernels::linear::LinearComparison,
}

#[allow(clippy::too_many_arguments)]
fn run_projection_case(
    executor: &IverilogSimExecutor,
    config: &AppConfig,
    model_bin_path: &std::path::Path,
    weight_values: &[f32],
    input_dim: usize,
    bias_values: &[f32],
    activation: &[f32],
    sequence_index: usize,
    input_start: usize,
    output_start: usize,
    tile_inner: usize,
    tile_cols: usize,
    quant: FixedPointConfig,
) -> Result<ProjectionCaseResult> {
    if activation.len() < input_start + tile_inner {
        anyhow::bail!(
            "reference activation for sequence {} is too short: need at least {}, got {}",
            sequence_index,
            input_start + tile_inner,
            activation.len()
        );
    }

    let input_float = activation[input_start..input_start + tile_inner].to_vec();
    let input_quantized = quant.quantize_slice(&input_float);

    let mut weight_tile_float = Vec::with_capacity(tile_inner * tile_cols);
    for input_offset in 0..tile_inner {
        for output_offset in 0..tile_cols {
            let output_index = output_start + output_offset;
            let input_index = input_start + input_offset;
            weight_tile_float.push(weight_values[output_index * input_dim + input_index]);
        }
    }
    let weight_tile_quantized = quant.quantize_slice(&weight_tile_float);

    let bias_float = bias_values[output_start..output_start + tile_cols].to_vec();
    let bias_quantized = quant.quantize_slice(&bias_float);

    let layer = LinearLayerI16 {
        input_dim: tile_inner,
        output_dim: tile_cols,
        weights: MatrixI16::new(tile_inner, tile_cols, weight_tile_quantized),
        bias: bias_quantized.clone(),
        quant,
    };

    let comparison = simulate_linear(
        executor,
        &config.fpga_sim_io_dir,
        &model_bin_path.display().to_string(),
        &layer,
        &input_quantized,
    )?;

    let float_reference = compute_float_projection(
        &input_float,
        weight_values,
        input_dim,
        bias_values,
        input_start,
        output_start,
        tile_inner,
        tile_cols,
    );
    let rtl_dequantized = dequantize_outputs(quant, &comparison.rtl_output);
    let max_abs_error = float_reference
        .iter()
        .zip(&rtl_dequantized)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    let matched = comparison.matched;

    Ok(ProjectionCaseResult {
        sequence_index,
        input_start,
        output_start,
        input_float,
        input_quantized,
        bias_float,
        bias_quantized,
        float_reference,
        rtl_dequantized,
        max_abs_error,
        matched,
        comparison,
    })
}

fn build_output_starts(output_dim: usize, tile_cols: usize) -> Vec<usize> {
    let mut starts = vec![
        0,
        64,
        256,
        output_dim / 2,
        output_dim.saturating_sub(tile_cols),
    ];
    starts.sort_unstable();
    starts.dedup();
    starts
        .into_iter()
        .filter(|start| *start + tile_cols <= output_dim)
        .collect()
}

fn build_input_starts(input_dim: usize, tile_inner: usize) -> Vec<usize> {
    let mut starts = vec![
        0,
        64,
        256,
        input_dim / 2,
        input_dim.saturating_sub(tile_inner),
    ];
    starts.sort_unstable();
    starts.dedup();
    starts
        .into_iter()
        .filter(|start| *start + tile_inner <= input_dim)
        .collect()
}

fn render_projection_sweep_table(cases: &[ProjectionCaseResult]) -> String {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec![
        Cell::new("seq"),
        Cell::new("in"),
        Cell::new("out"),
        Cell::new("matched"),
        Cell::new("max_abs_err"),
        Cell::new("float_ref"),
        Cell::new("rtl"),
    ]);

    for case in cases {
        table.add_row(vec![
            Cell::new(case.sequence_index),
            Cell::new(case.input_start),
            Cell::new(case.output_start),
            Cell::new(case.matched),
            Cell::new(format!("{:.6}", case.max_abs_error)),
            Cell::new(format_vector_f32(&case.float_reference)),
            Cell::new(format_vector_f32(&case.rtl_dequantized)),
        ]);
    }

    table.to_string()
}

fn print_transcript(transcript: &Transcript) {
    println!("backend: {}", transcript.backend);
    println!("model: {}", transcript.model);
    println!(
        "audio_duration_seconds: {:.3}",
        transcript.audio_duration_seconds
    );
    for note in &transcript.notes {
        println!("note: {note}");
    }
    for segment in &transcript.segments {
        println!(
            "[{:.2}..{:.2}] {}",
            segment.start_seconds, segment.end_seconds, segment.text
        );
    }
}

fn benchmark_backend(
    backend: &dyn crate::backend::TranscriptionBackend,
    request: &TranscriptionRequest,
    iterations: usize,
    warmup: usize,
) -> Result<BenchmarkReport> {
    let mut last_transcript = None;

    for _ in 0..warmup {
        last_transcript = Some(backend.transcribe(request)?);
    }

    let mut runs = Vec::with_capacity(iterations);
    for iteration in 1..=iterations {
        let started = Instant::now();
        let transcript = backend.transcribe(request)?;
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let realtime_factor = if transcript.audio_duration_seconds > 0.0 {
            Some(elapsed_seconds / f64::from(transcript.audio_duration_seconds))
        } else {
            None
        };

        runs.push(BenchmarkRun {
            iteration,
            elapsed_seconds,
            realtime_factor,
        });
        last_transcript = Some(transcript);
    }

    Ok(BenchmarkReport {
        warmup_runs: warmup,
        measured_runs: runs,
        transcript: last_transcript.expect("benchmark must have at least one run"),
    })
}

struct BenchmarkReport {
    warmup_runs: usize,
    measured_runs: Vec<BenchmarkRun>,
    transcript: Transcript,
}

fn print_benchmark(report: &BenchmarkReport) {
    println!("backend: {}", report.transcript.backend);
    println!("model: {}", report.transcript.model);
    println!(
        "audio_duration_seconds: {:.3}",
        report.transcript.audio_duration_seconds
    );
    println!("warmup_runs: {}", report.warmup_runs);
    println!("measured_runs: {}", report.measured_runs.len());

    for run in &report.measured_runs {
        match run.realtime_factor {
            Some(rtf) => println!(
                "run {}: {:.3}s (rtf {:.3}x)",
                run.iteration, run.elapsed_seconds, rtf
            ),
            None => println!("run {}: {:.3}s", run.iteration, run.elapsed_seconds),
        }
    }

    let elapsed_values = report
        .measured_runs
        .iter()
        .map(|run| run.elapsed_seconds)
        .collect::<Vec<_>>();
    let avg_seconds = elapsed_values.iter().sum::<f64>() / elapsed_values.len() as f64;
    let min_seconds = elapsed_values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_seconds = elapsed_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    println!("avg_seconds: {:.3}", avg_seconds);
    println!("min_seconds: {:.3}", min_seconds);
    println!("max_seconds: {:.3}", max_seconds);

    if report.transcript.audio_duration_seconds > 0.0 {
        let avg_rtf = avg_seconds / f64::from(report.transcript.audio_duration_seconds);
        println!("avg_realtime_factor: {:.3}x", avg_rtf);
    }

    println!("transcript_preview:");
    for segment in &report.transcript.segments {
        println!(
            "[{:.2}..{:.2}] {}",
            segment.start_seconds, segment.end_seconds, segment.text
        );
    }
}
