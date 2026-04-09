use std::path::PathBuf;

use crate::types::{BackendKind, PartitionPreset, TranscriptionRequest};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub project_root: PathBuf,
    pub worker_launcher: PathBuf,
    pub worker_launcher_args: Vec<String>,
    pub worker_script: PathBuf,
    pub fpga_sim_io_dir: PathBuf,
    pub default_backend: BackendKind,
    pub default_partition: PartitionPreset,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            project_root: PathBuf::from("."),
            worker_launcher: PathBuf::from("uv"),
            worker_launcher_args: vec!["run".to_owned()],
            worker_script: PathBuf::from("python/ct2_worker.py"),
            fpga_sim_io_dir: PathBuf::from("fpga/tmp"),
            default_backend: BackendKind::Ct2Python,
            default_partition: PartitionPreset::Hybrid,
        }
    }
}

impl AppConfig {
    pub fn sample_request(&self) -> TranscriptionRequest {
        TranscriptionRequest {
            audio_path: PathBuf::from("samples/demo.wav"),
            backend: self.default_backend,
            partition: self.default_partition,
            initial_prompt: None,
        }
    }
}
