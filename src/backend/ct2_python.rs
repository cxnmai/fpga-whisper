use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::backend::TranscriptionBackend;
use crate::types::{BackendDescriptor, BackendKind, Transcript, TranscriptionRequest};

pub struct Ct2PythonBackend {
    python_executable: PathBuf,
    worker_script: PathBuf,
}

impl Ct2PythonBackend {
    pub fn new(python_executable: PathBuf, worker_script: PathBuf) -> Self {
        Self {
            python_executable,
            worker_script,
        }
    }
}

impl TranscriptionBackend for Ct2PythonBackend {
    fn descriptor(&self) -> BackendDescriptor {
        super::describe_backend(BackendKind::Ct2Python)
    }

    fn transcribe(&self, request: &TranscriptionRequest) -> Result<Transcript> {
        let mut command = Command::new(&self.python_executable);
        command
            .arg(&self.worker_script)
            .arg("--audio")
            .arg(&request.audio_path)
            .arg("--model")
            .arg(request.model.as_ct2_model_id())
            .arg("--model-repo")
            .arg(request.model.as_hf_repo());

        if let Some(language) = &request.language {
            command.arg("--language").arg(language);
        }

        if let Some(prompt) = &request.initial_prompt {
            command.arg("--initial-prompt").arg(prompt);
        }

        let output = command.output().with_context(|| {
            format!(
                "failed to run Python worker {}",
                self.worker_script.display()
            )
        })?;

        if !output.status.success() {
            bail!(
                "Python worker exited with status {}.\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        serde_json::from_slice(&output.stdout).with_context(|| {
            format!(
                "failed to parse worker JSON output.\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
    }
}
