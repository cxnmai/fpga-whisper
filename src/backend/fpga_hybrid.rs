use anyhow::Result;

use crate::backend::TranscriptionBackend;
use crate::types::{
    BackendDescriptor, BackendKind, MODEL_HF_REPO, Transcript, TranscriptSegment,
    TranscriptionRequest,
};

pub struct FpgaHybridBackend;

impl FpgaHybridBackend {
    pub fn new() -> Self {
        Self
    }
}

impl TranscriptionBackend for FpgaHybridBackend {
    fn descriptor(&self) -> BackendDescriptor {
        super::describe_backend(BackendKind::FpgaHybrid)
    }

    fn transcribe(&self, request: &TranscriptionRequest) -> Result<Transcript> {
        let fpga_stages = request
            .partition
            .stages_on_fpga()
            .iter()
            .map(|stage| stage.label())
            .collect::<Vec<_>>()
            .join(", ");

        Ok(Transcript {
            backend: "fpga-hybrid".to_owned(),
            model: MODEL_HF_REPO.to_owned(),
            notes: vec![
                request.partition.summary().to_owned(),
                format!("Planned FPGA stages: {fpga_stages}"),
                "Next implementation step: connect the host runtime to an FPGA transport and keep weights resident on-board during chunk execution."
                    .to_owned(),
            ],
            segments: vec![TranscriptSegment {
                start_seconds: 0.0,
                end_seconds: 0.0,
                text: "[skeleton] FPGA hybrid runtime is not wired yet.".to_owned(),
            }],
        })
    }
}
