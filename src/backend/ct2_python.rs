use std::path::PathBuf;

use anyhow::Result;

use crate::backend::TranscriptionBackend;
use crate::types::{
    BackendDescriptor, BackendKind, Transcript, TranscriptSegment, TranscriptionRequest,
};

pub struct Ct2PythonBackend {
    worker_script: PathBuf,
}

impl Ct2PythonBackend {
    pub fn new(worker_script: PathBuf) -> Self {
        Self { worker_script }
    }
}

impl TranscriptionBackend for Ct2PythonBackend {
    fn descriptor(&self) -> BackendDescriptor {
        super::describe_backend(BackendKind::Ct2Python)
    }

    fn transcribe(&self, request: &TranscriptionRequest) -> Result<Transcript> {
        let command = format!(
            "python3 {} --audio {} --model {}",
            self.worker_script.display(),
            request.audio_path.display(),
            request.model.as_hf_repo()
        );

        Ok(Transcript {
            backend: "ct2-python".to_owned(),
            model: request.model.as_hf_repo().to_owned(),
            notes: vec![
                "Skeleton mode only. The Rust app is ready to hand work to a Python worker."
                    .to_owned(),
                format!("Planned command: {command}"),
                "Replace this stub with JSON IPC once the worker is wired to CTranslate2."
                    .to_owned(),
            ],
            segments: vec![TranscriptSegment {
                start_seconds: 0.0,
                end_seconds: 0.0,
                text: "[skeleton] CTranslate2 baseline is not wired yet.".to_owned(),
            }],
        })
    }
}
