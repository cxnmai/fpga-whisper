use anyhow::Result;
use clap::{Parser, Subcommand};
use std::time::Instant;

use crate::backend::{build_backend, describe_backend};
use crate::config::AppConfig;
use crate::fpga::kernels::gemm::{MatrixI16, simulate_gemm_via_dot_products};
use crate::fpga::sim::IverilogSimExecutor;
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

    let comparison = simulate_gemm_via_dot_products(
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
    let mut deduped = std::collections::BTreeSet::new();
    for note in comparison.notes {
        if !deduped.insert(note.clone()) {
            continue;
        }
        println!("- {note}");
    }

    Ok(())
}

fn print_matrix_i64(matrix: &crate::fpga::kernels::gemm::MatrixI64) {
    for row in 0..matrix.rows {
        let values = (0..matrix.cols)
            .map(|col| matrix.get(row, col).to_string())
            .collect::<Vec<_>>();
        println!("[{}]", values.join(", "));
    }
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
