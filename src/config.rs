use std::path::PathBuf;

use crate::types::{BackendKind, ModelId, PartitionPreset, TranscriptionRequest};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub worker_launcher: PathBuf,
    pub worker_launcher_args: Vec<String>,
    pub worker_script: PathBuf,
    pub default_model: ModelId,
    pub default_backend: BackendKind,
    pub default_partition: PartitionPreset,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            worker_launcher: PathBuf::from("uv"),
            worker_launcher_args: vec!["run".to_owned()],
            worker_script: PathBuf::from("python/ct2_worker.py"),
            default_model: ModelId::DistilWhisperSmallEn,
            default_backend: BackendKind::Ct2Python,
            default_partition: PartitionPreset::Hybrid,
        }
    }
}

impl AppConfig {
    pub fn sample_request(&self) -> TranscriptionRequest {
        TranscriptionRequest {
            audio_path: PathBuf::from("samples/demo.wav"),
            model: self.default_model,
            backend: self.default_backend,
            partition: self.default_partition,
            language: Some("en".to_owned()),
            initial_prompt: None,
        }
    }
}
