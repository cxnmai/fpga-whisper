use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::types::MODEL_CT2_CACHE_REPO_DIR;
use crate::types::{BackendKind, PartitionPreset, TranscriptionRequest};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub project_root: PathBuf,
    pub worker_launcher: PathBuf,
    pub worker_launcher_args: Vec<String>,
    pub worker_script: PathBuf,
    pub reference_python: PathBuf,
    pub reference_exporter_script: PathBuf,
    pub reference_cache_dir: PathBuf,
    pub reference_export_positions: usize,
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
            reference_python: PathBuf::from("python3"),
            reference_exporter_script: PathBuf::from("python/export_reference_activation.py"),
            reference_cache_dir: PathBuf::from("artifacts/reference"),
            reference_export_positions: 4,
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

    pub fn model_snapshot_dir(&self) -> Result<PathBuf> {
        let repo_dir = self
            .project_root
            .join("models")
            .join(MODEL_CT2_CACHE_REPO_DIR);
        let revision = std::fs::read_to_string(repo_dir.join("refs/main"))
            .with_context(|| "failed to read baked model revision from models/.../refs/main")?;
        Ok(repo_dir.join("snapshots").join(revision.trim()))
    }

    pub fn model_bin_path(&self) -> Result<PathBuf> {
        Ok(self.model_snapshot_dir()?.join("model.bin"))
    }

    pub fn reference_activation_cache_path(&self, audio_path: &std::path::Path) -> PathBuf {
        let stem = audio_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("sample");
        self.project_root
            .join(&self.reference_cache_dir)
            .join(format!(
                "encoder_layer0_ffn_inputs_{stem}_p{}.json",
                self.reference_export_positions
            ))
    }
}
