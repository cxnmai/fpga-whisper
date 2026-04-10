use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::fpga::transport::{FpgaExecutor, FpgaSimRequest, FpgaSimResponse};

pub struct IverilogSimExecutor {
    pub runner: PathBuf,
    pub runner_args: Vec<String>,
    pub project_root: PathBuf,
}

impl IverilogSimExecutor {
    pub fn new(project_root: PathBuf) -> Self {
        Self {
            runner: PathBuf::from("python3"),
            runner_args: vec![
                project_root
                    .join("fpga/scripts/run_fpga_sim.py")
                    .display()
                    .to_string(),
            ],
            project_root,
        }
    }
}

impl FpgaExecutor for IverilogSimExecutor {
    fn name(&self) -> &'static str {
        "iverilog-sim"
    }

    fn execute_stage(
        &self,
        request: &FpgaSimRequest,
        output_dir: &Path,
    ) -> Result<FpgaSimResponse> {
        fs::create_dir_all(output_dir)?;
        let request_path = output_dir.join("sim_request.json");
        let response_path = output_dir.join("sim_response.json");
        fs::write(&request_path, serde_json::to_vec_pretty(request)?)?;

        let mut command = Command::new(&self.runner);
        command.args(&self.runner_args);
        command
            .arg("--request")
            .arg(&request_path)
            .arg("--response")
            .arg(&response_path)
            .current_dir(&self.project_root);

        let output = command
            .output()
            .with_context(|| format!("failed to run simulator bridge {}", self.runner.display()))?;

        if !output.status.success() {
            bail!(
                "simulator bridge exited with status {}.\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let bytes = fs::read(&response_path)
            .with_context(|| format!("missing simulator response {}", response_path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid simulator response {}", response_path.display()))
    }
}
