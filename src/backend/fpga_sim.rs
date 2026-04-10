use std::path::PathBuf;

use anyhow::Result;

use crate::backend::TranscriptionBackend;
use crate::fpga::sim::IverilogSimExecutor;
use crate::fpga::transport::{FpgaExecutor, FpgaSimRequest};
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

        let vector_a = vec![3, -2, 7, 4, -1, 5, 2, -3];
        let vector_b = vec![6, 8, -4, 1, 9, -2, 3, 5];
        let expected_result = vector_a
            .iter()
            .zip(&vector_b)
            .map(|(lhs, rhs)| i64::from(*lhs) * i64::from(*rhs))
            .sum::<i64>();

        let response = self.executor.execute_stage(
            &FpgaSimRequest {
                operation: "dot-product".to_owned(),
                audio_path: request.audio_path.display().to_string(),
                vector_a: vector_a.clone(),
                vector_b: vector_b.clone(),
                expected_result,
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
                "First real RTL primitive: signed 8-lane int16 dot product.".to_owned(),
                format!("Input vector A: {vector_a:?}"),
                format!("Input vector B: {vector_b:?}"),
                format!("Expected software result: {}", response.expected_result),
                format!("RTL result: {}", response.rtl_result),
                format!("Matched: {}", response.matched),
            ]
            .into_iter()
            .chain(response.notes)
            .collect(),
            segments: vec![TranscriptSegment {
                start_seconds: 0.0,
                end_seconds: 0.0,
                text: format!(
                    "[fpga-sim] dot-product smoke test result = {}",
                    response.rtl_result
                ),
            }],
        })
    }
}
