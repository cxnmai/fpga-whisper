use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::backend::{build_backend, describe_backend};
use crate::config::AppConfig;
use crate::tui::run_tui;
use crate::types::{BackendKind, PartitionPreset, TranscriptionRequest};

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
        language: Option<String>,
        #[arg(long)]
        initial_prompt: Option<String>,
    },
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
            language,
            initial_prompt,
        } => {
            let request = TranscriptionRequest {
                audio_path: audio,
                backend,
                partition,
                language,
                initial_prompt,
            };

            let backend = build_backend(backend, &config);
            let transcript = backend.transcribe(&request)?;
            println!("backend: {}", transcript.backend);
            println!("model: {}", transcript.model);
            for note in transcript.notes {
                println!("note: {note}");
            }
            for segment in transcript.segments {
                println!(
                    "[{:.2}..{:.2}] {}",
                    segment.start_seconds, segment.end_seconds, segment.text
                );
            }
        }
        Commands::Tui => run_tui(config)?,
    }

    Ok(())
}
