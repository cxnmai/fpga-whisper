use std::path::PathBuf;

use anyhow::Result;

use crate::backend::TranscriptionBackend;
use crate::fpga::sim::IverilogSimExecutor;
use crate::fpga::transport::{FpgaExecutor, FpgaFeatureRequest};
use crate::types::{
    BackendDescriptor, BackendKind, MODEL_HF_REPO, Transcript, TranscriptSegment,
    TranscriptionRequest,
};

pub struct FpgaSimBackend {
    executor: IverilogSimExecutor,
    io_dir: PathBuf,
}

impl FpgaSimBackend {
    pub fn new(project_root: PathBuf, io_dir: PathBuf) -> Self {
        Self {
            executor: IverilogSimExecutor::new(project_root),
            io_dir,
        }
    }
}

impl TranscriptionBackend for FpgaSimBackend {
    fn descriptor(&self) -> BackendDescriptor {
        super::describe_backend(BackendKind::FpgaSim)
    }

    fn transcribe(&self, request: &TranscriptionRequest) -> Result<Transcript> {
        let fpga_stages = request
            .partition
            .stages_on_fpga()
            .iter()
            .map(|stage| stage.label())
            .collect::<Vec<_>>()
            .join(", ");

        let response = self.executor.execute_feature_stage(
            &FpgaFeatureRequest {
                audio_path: request.audio_path.display().to_string(),
                requested_stage: "feature-extraction".to_owned(),
            },
            &self.io_dir,
        )?;

        Ok(Transcript {
            backend: "fpga-sim".to_owned(),
            model: MODEL_HF_REPO.to_owned(),
            audio_duration_seconds: 0.0,
            notes: vec![
                "Rust host is talking directly to the simulated RTL boundary.".to_owned(),
                format!("Executor: {}", self.executor.name()),
                format!("Planned FPGA stages: {fpga_stages}"),
                format!(
                    "Simulator produced {} with {} frames x {} bins.",
                    response.produced_stage, response.frame_count, response.bin_count
                ),
            ]
            .into_iter()
            .chain(response.notes)
            .collect(),
            segments: vec![TranscriptSegment {
                start_seconds: 0.0,
                end_seconds: 0.0,
                text: "[skeleton] FPGA simulator returned staged feature metadata only.".to_owned(),
            }],
        })
    }
}
