use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::backend::TranscriptionBackend;
use crate::types::{
    BackendDescriptor, BackendKind, MODEL_CT2_ALIAS, MODEL_HF_REPO, Transcript,
    TranscriptionRequest,
};

pub struct Ct2PythonBackend {
    worker_launcher: PathBuf,
    worker_launcher_args: Vec<String>,
    worker_script: PathBuf,
}

impl Ct2PythonBackend {
    pub fn new(
        worker_launcher: PathBuf,
        worker_launcher_args: Vec<String>,
        worker_script: PathBuf,
    ) -> Self {
        Self {
            worker_launcher,
            worker_launcher_args,
            worker_script,
        }
    }

    pub fn build_worker_command(&self, request: &TranscriptionRequest) -> Command {
        let mut command = Command::new(&self.worker_launcher);
        command.args(&self.worker_launcher_args);
        command
            .env("UV_CACHE_DIR", ".uv-cache")
            .arg(&self.worker_script)
            .arg("--audio")
            .arg(&request.audio_path);

        if let Some(prompt) = &request.initial_prompt {
            command.arg("--initial-prompt").arg(prompt);
        }

        command
    }
}

impl TranscriptionBackend for Ct2PythonBackend {
    fn descriptor(&self) -> BackendDescriptor {
        super::describe_backend(BackendKind::Ct2Python)
    }

    fn transcribe(&self, request: &TranscriptionRequest) -> Result<Transcript> {
        let mut command = self.build_worker_command(request);
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

        serde_json::from_slice(&output.stdout)
            .with_context(|| {
                format!(
                    "failed to parse worker JSON output.\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )
            })
            .map(|mut transcript: Transcript| {
                transcript.model = MODEL_HF_REPO.to_owned();
                if transcript
                    .notes
                    .iter()
                    .all(|note| !note.contains("Model alias:"))
                {
                    transcript
                        .notes
                        .insert(1, format!("Model alias: {MODEL_CT2_ALIAS}"));
                }
                transcript
            })
    }
}
