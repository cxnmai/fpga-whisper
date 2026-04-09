mod ct2_python;
mod fpga_hybrid;

use anyhow::Result;

use crate::config::AppConfig;
use crate::types::{
    BackendDescriptor, BackendKind, PartitionPreset, PipelineStage, Transcript,
    TranscriptionRequest,
};

pub use ct2_python::Ct2PythonBackend;
pub use fpga_hybrid::FpgaHybridBackend;

pub trait TranscriptionBackend {
    fn descriptor(&self) -> BackendDescriptor;
    fn transcribe(&self, request: &TranscriptionRequest) -> Result<Transcript>;
}

pub fn build_backend(kind: BackendKind, config: &AppConfig) -> Box<dyn TranscriptionBackend> {
    match kind {
        BackendKind::Ct2Python => Box::new(Ct2PythonBackend::new(config.worker_script.clone())),
        BackendKind::FpgaHybrid => Box::new(FpgaHybridBackend::new()),
    }
}

pub fn describe_backend(kind: BackendKind) -> BackendDescriptor {
    match kind {
        BackendKind::Ct2Python => BackendDescriptor {
            id: BackendKind::Ct2Python,
            summary: "Host-side CTranslate2 baseline. Use this as the correctness oracle.",
            partition: PartitionPreset::CpuOnly,
            host_stages: vec![
                PipelineStage::AudioDecode,
                PipelineStage::FeatureExtraction,
                PipelineStage::Encoder,
                PipelineStage::DecoderMath,
                PipelineStage::DecodePolicy,
                PipelineStage::PostProcess,
            ],
            fpga_stages: vec![],
        },
        BackendKind::FpgaHybrid => BackendDescriptor {
            id: BackendKind::FpgaHybrid,
            summary: "Hybrid path. Host keeps control flow while FPGA absorbs dense math stages.",
            partition: PartitionPreset::Hybrid,
            host_stages: vec![
                PipelineStage::AudioDecode,
                PipelineStage::DecodePolicy,
                PipelineStage::PostProcess,
            ],
            fpga_stages: vec![
                PipelineStage::FeatureExtraction,
                PipelineStage::Encoder,
                PipelineStage::DecoderMath,
            ],
        },
    }
}
