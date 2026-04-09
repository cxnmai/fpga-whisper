use std::fmt;
use std::path::PathBuf;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub const MODEL_HF_REPO: &str = "distil-whisper/distil-small.en";
pub const MODEL_CT2_ALIAS: &str = "distil-small.en";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum BackendKind {
    Ct2Python,
    FpgaHybrid,
}

impl BackendKind {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Ct2Python => "ct2-python",
            Self::FpgaHybrid => "fpga-hybrid",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStage {
    AudioDecode,
    FeatureExtraction,
    Encoder,
    DecoderMath,
    DecodePolicy,
    PostProcess,
}

impl PipelineStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::AudioDecode => "audio decode",
            Self::FeatureExtraction => "feature extraction",
            Self::Encoder => "encoder",
            Self::DecoderMath => "decoder math",
            Self::DecodePolicy => "decode policy",
            Self::PostProcess => "post-process",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum PartitionPreset {
    CpuOnly,
    Frontend,
    Encoder,
    Hybrid,
}

impl PartitionPreset {
    pub fn summary(self) -> &'static str {
        match self {
            Self::CpuOnly => "Reference path. All compute remains on the host.",
            Self::Frontend => "Offload STFT/log-mel to the FPGA and keep the model on the host.",
            Self::Encoder => {
                "Offload front-end plus encoder. Host keeps decoder generation and text logic."
            }
            Self::Hybrid => {
                "Target architecture. FPGA owns dense math blocks while host keeps decode control."
            }
        }
    }

    pub fn stages_on_fpga(self) -> &'static [PipelineStage] {
        match self {
            Self::CpuOnly => &[],
            Self::Frontend => &[PipelineStage::FeatureExtraction],
            Self::Encoder => &[PipelineStage::FeatureExtraction, PipelineStage::Encoder],
            Self::Hybrid => &[
                PipelineStage::FeatureExtraction,
                PipelineStage::Encoder,
                PipelineStage::DecoderMath,
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionRequest {
    pub audio_path: PathBuf,
    pub backend: BackendKind,
    pub partition: PartitionPreset,
    pub initial_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start_seconds: f32,
    pub end_seconds: f32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub backend: String,
    pub model: String,
    pub audio_duration_seconds: f32,
    pub notes: Vec<String>,
    pub segments: Vec<TranscriptSegment>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkRun {
    pub iteration: usize,
    pub elapsed_seconds: f64,
    pub realtime_factor: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct BackendDescriptor {
    pub id: BackendKind,
    pub summary: &'static str,
    pub partition: PartitionPreset,
    pub host_stages: Vec<PipelineStage>,
    pub fpga_stages: Vec<PipelineStage>,
}

impl fmt::Display for BackendDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.id.display_name())?;
        writeln!(f, "  summary: {}", self.summary)?;
        writeln!(f, "  partition: {:?}", self.partition)?;
        writeln!(
            f,
            "  host stages: {}",
            self.host_stages
                .iter()
                .map(|stage| stage.label())
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        write!(
            f,
            "  fpga stages: {}",
            self.fpga_stages
                .iter()
                .map(|stage| stage.label())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}
