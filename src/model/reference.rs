use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::config::AppConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceActivationExport {
    pub model_repo: String,
    pub audio_path: String,
    pub layer_name: String,
    pub sequence_index: usize,
    pub hidden_size: usize,
    pub activation: Vec<f32>,
}

pub fn ensure_reference_activation_export(
    config: &AppConfig,
    audio_path: &Path,
    refresh: bool,
) -> Result<PathBuf> {
    let output_path = config.reference_activation_cache_path(audio_path);
    if output_path.exists() && !refresh {
        return Ok(output_path);
    }

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let output = Command::new(&config.reference_python)
        .arg(&config.reference_exporter_script)
        .arg("--audio")
        .arg(audio_path)
        .arg("--output")
        .arg(&output_path)
        .current_dir(&config.project_root)
        .output()
        .with_context(|| {
            format!(
                "failed to run activation exporter {}",
                config.reference_exporter_script.display()
            )
        })?;

    if !output.status.success() {
        bail!(
            "activation exporter exited with status {}.\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(output_path)
}

pub fn load_reference_activation(path: &Path) -> Result<ReferenceActivationExport> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}
