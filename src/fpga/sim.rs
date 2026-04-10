use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

use crate::fpga::transport::{
    DotProductRequest, DotProductResponse, FpgaExecutor, GemmTileI16Request, GemmTileI64Response,
};

pub struct IverilogSimExecutor {
    pub iverilog: PathBuf,
    pub vvp: PathBuf,
    pub project_root: PathBuf,
}

impl IverilogSimExecutor {
    pub fn new(project_root: PathBuf) -> Self {
        let project_root = project_root.canonicalize().unwrap_or(project_root);
        Self {
            iverilog: PathBuf::from("iverilog"),
            vvp: PathBuf::from("vvp"),
            project_root,
        }
    }
}

impl FpgaExecutor for IverilogSimExecutor {
    fn name(&self) -> &'static str {
        "iverilog-sim"
    }

    fn execute_dot_product(
        &self,
        request: &DotProductRequest,
        output_dir: &Path,
    ) -> Result<DotProductResponse> {
        if request.vector_a.len() != 8 || request.vector_b.len() != 8 {
            bail!("dot-product simulator expects exactly 8 lanes");
        }

        let scratch_dir = create_scratch_dir(output_dir, "dot_product")?;
        let request_path = scratch_dir.join("sim_request.json");
        let response_path = scratch_dir.join("sim_response.json");
        let vector_include_path = scratch_dir.join("dot_product_vectors.vh");
        let result_path = scratch_dir.join("dot_product_result.txt");
        let vvp_output_path = scratch_dir.join("dot_product_i16x8_tb.out");

        fs::write(&request_path, serde_json::to_vec_pretty(request)?)?;
        fs::write(
            &vector_include_path,
            build_vector_include(&request.vector_a, &request.vector_b),
        )?;

        let compile_output = Command::new(&self.iverilog)
            .arg("-g2012")
            .arg("-o")
            .arg(&vvp_output_path)
            .arg(self.project_root.join("fpga/rtl/dot_product_i16x8.v"))
            .arg(self.project_root.join("fpga/tb/dot_product_i16x8_tb.v"))
            .current_dir(&scratch_dir)
            .output()
            .with_context(|| format!("failed to run {}", self.iverilog.display()))?;

        if !compile_output.status.success() {
            bail!(
                "iverilog exited with status {}.\nstdout:\n{}\nstderr:\n{}",
                compile_output.status,
                String::from_utf8_lossy(&compile_output.stdout),
                String::from_utf8_lossy(&compile_output.stderr)
            );
        }

        let run_output = Command::new(&self.vvp)
            .arg(&vvp_output_path)
            .current_dir(&scratch_dir)
            .output()
            .with_context(|| format!("failed to run {}", self.vvp.display()))?;

        if !run_output.status.success() {
            bail!(
                "vvp exited with status {}.\nstdout:\n{}\nstderr:\n{}",
                run_output.status,
                String::from_utf8_lossy(&run_output.stdout),
                String::from_utf8_lossy(&run_output.stderr)
            );
        }

        let rtl_result = fs::read_to_string(&result_path)
            .with_context(|| format!("missing simulator result {}", result_path.display()))?
            .trim()
            .parse::<i64>()
            .with_context(|| format!("invalid simulator result {}", result_path.display()))?;

        let response = DotProductResponse {
            rtl_result,
            expected_result: request.expected_result,
            matched: rtl_result == request.expected_result,
            notes: vec![
                "Executed real RTL through direct Rust -> iverilog/vvp invocation.".to_owned(),
                format!("Received audio path: {}", request.audio_path),
                "Operation: dot-product".to_owned(),
                format!("Scratch directory: {}", scratch_dir.display()),
                format!(
                    "Waveform: {}",
                    scratch_dir.join("dot_product_i16x8_tb.vcd").display()
                ),
                "This primitive is the first reusable numeric block for future FPGA offload."
                    .to_owned(),
            ],
        };

        fs::write(&response_path, serde_json::to_vec_pretty(&response)?)?;
        Ok(response)
    }

    fn execute_gemm_tile(
        &self,
        request: &GemmTileI16Request,
        output_dir: &Path,
    ) -> Result<GemmTileI64Response> {
        request
            .shape
            .as_layout()
            .validate(request.lhs_tile.len(), request.rhs_tile.len())?;
        if request.shape.inner != 8 {
            bail!("current simulator GEMM path expects inner dimension of exactly 8");
        }
        if request.accumulator_input.len() != request.shape.rows * request.shape.cols {
            bail!(
                "simulator accumulator length mismatch: expected {}, got {}",
                request.shape.rows * request.shape.cols,
                request.accumulator_input.len()
            );
        }
        let scratch_dir = create_scratch_dir(output_dir, "gemm_tile")?;
        let request_path = scratch_dir.join("sim_request.json");
        let response_path = scratch_dir.join("sim_response.json");
        let vector_include_path = scratch_dir.join("gemm_tile_vectors.vh");
        let result_path = scratch_dir.join("gemm_tile_result.txt");
        let vvp_output_path = scratch_dir.join("gemm_tile_i16x8_tb.out");

        fs::write(&request_path, serde_json::to_vec_pretty(request)?)?;
        fs::write(
            &vector_include_path,
            build_gemm_tile_include(
                request.shape,
                &request.lhs_tile,
                &request.rhs_tile,
                &request.accumulator_input,
            )?,
        )?;

        let compile_output = Command::new(&self.iverilog)
            .arg("-g2012")
            .arg("-o")
            .arg(&vvp_output_path)
            .arg(self.project_root.join("fpga/rtl/dot_product_i16x8.v"))
            .arg(self.project_root.join("fpga/rtl/gemm_tile_i16x8.v"))
            .arg(self.project_root.join("fpga/rtl/gemm_tile_accum_i16x8.v"))
            .arg(self.project_root.join("fpga/tb/gemm_tile_i16x8_tb.v"))
            .current_dir(&scratch_dir)
            .output()
            .with_context(|| format!("failed to run {}", self.iverilog.display()))?;

        if !compile_output.status.success() {
            bail!(
                "iverilog exited with status {}.\nstdout:\n{}\nstderr:\n{}",
                compile_output.status,
                String::from_utf8_lossy(&compile_output.stdout),
                String::from_utf8_lossy(&compile_output.stderr)
            );
        }

        let run_output = Command::new(&self.vvp)
            .arg(&vvp_output_path)
            .current_dir(&scratch_dir)
            .output()
            .with_context(|| format!("failed to run {}", self.vvp.display()))?;

        if !run_output.status.success() {
            bail!(
                "vvp exited with status {}.\nstdout:\n{}\nstderr:\n{}",
                run_output.status,
                String::from_utf8_lossy(&run_output.stdout),
                String::from_utf8_lossy(&run_output.stderr)
            );
        }

        let rtl_output = fs::read_to_string(&result_path)
            .with_context(|| format!("missing simulator result {}", result_path.display()))?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(|line| {
                line.parse::<i64>().with_context(|| {
                    format!("invalid simulator result line in {}", result_path.display())
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if rtl_output.len() != request.expected_output.len() {
            bail!(
                "simulator GEMM output length mismatch: expected {}, got {}",
                request.expected_output.len(),
                rtl_output.len()
            );
        }

        let matched = rtl_output == request.expected_output;
        let notes = vec![
            format!(
                "Executed real RTL GEMM+accum tile {}x{}x{} through direct Rust -> iverilog/vvp invocation.",
                request.shape.rows, request.shape.cols, request.shape.inner
            ),
            format!("Received audio path: {}", request.audio_path),
            "Operation: gemm-tile-accum".to_owned(),
            format!("Scratch directory: {}", scratch_dir.display()),
            format!(
                "Waveform: {}",
                scratch_dir.join("gemm_tile_i16x8_tb.vcd").display()
            ),
            "This accumulator tile primitive can keep partial sums on the FPGA boundary."
                .to_owned(),
        ];

        let response = GemmTileI64Response {
            shape: request.shape,
            rtl_output,
            expected_output: request.expected_output.clone(),
            matched,
            notes,
        };
        fs::write(&response_path, serde_json::to_vec_pretty(&response)?)?;
        Ok(response)
    }
}

fn build_gemm_tile_include(
    shape: crate::fpga::transport::GemmTileShape,
    lhs_tile: &[i16],
    rhs_tile: &[i16],
    accumulator_input: &[i64],
) -> Result<String> {
    shape.as_layout().validate(lhs_tile.len(), rhs_tile.len())?;
    let accum_len = shape.rows * shape.cols;
    if accumulator_input.len() != accum_len {
        bail!(
            "accumulator tile length mismatch: expected {}, got {}",
            accum_len,
            accumulator_input.len()
        );
    }

    let lhs_bits = shape.rows * shape.inner * 16;
    let rhs_bits = shape.inner * shape.cols * 16;
    let accum_bits = shape.rows * shape.cols * 64;

    Ok(format!(
        "localparam integer TILE_ROWS = {};\nlocalparam integer TILE_COLS = {};\nlocalparam integer TILE_INNER = {};\nlocalparam signed [{}:0] LHS_TILE = {};\nlocalparam signed [{}:0] RHS_TILE = {};\nlocalparam signed [{}:0] ACCUM_TILE = {};\n",
        shape.rows,
        shape.cols,
        shape.inner,
        lhs_bits - 1,
        build_packed_i16_literal(lhs_tile),
        rhs_bits - 1,
        build_packed_i16_literal(rhs_tile),
        accum_bits - 1,
        build_packed_i64_literal(accumulator_input),
    ))
}

fn build_vector_include(vector_a: &[i16], vector_b: &[i16]) -> String {
    let mut lines = Vec::with_capacity(vector_a.len() + vector_b.len());
    for (index, value) in vector_a.iter().enumerate() {
        lines.push(format!(
            "localparam signed [15:0] VEC_A{index} = {};",
            verilog_i16_literal(*value)
        ));
    }
    for (index, value) in vector_b.iter().enumerate() {
        lines.push(format!(
            "localparam signed [15:0] VEC_B{index} = {};",
            verilog_i16_literal(*value)
        ));
    }
    lines.join("\n") + "\n"
}

fn create_scratch_dir(output_dir: &Path, prefix: &str) -> Result<PathBuf> {
    fs::create_dir_all(output_dir)?;
    let output_dir = output_dir
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", output_dir.display()))?;
    let timestamp_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is set before UNIX_EPOCH")?
        .as_nanos();
    let scratch_dir = output_dir.join(format!("{prefix}_{timestamp_ns}"));
    fs::create_dir_all(&scratch_dir)?;
    Ok(scratch_dir)
}

fn verilog_i16_literal(value: i16) -> String {
    if value < 0 {
        format!("-16'sd{}", i32::from(value).abs())
    } else {
        format!("16'sd{value}")
    }
}

fn build_packed_i16_literal(values: &[i16]) -> String {
    let parts = values
        .iter()
        .rev()
        .map(|value| verilog_i16_literal(*value))
        .collect::<Vec<_>>();
    format!("{{{}}}", parts.join(", "))
}

fn build_packed_i64_literal(values: &[i64]) -> String {
    let parts = values
        .iter()
        .rev()
        .map(|value| verilog_i64_literal(*value))
        .collect::<Vec<_>>();
    format!("{{{}}}", parts.join(", "))
}

fn verilog_i64_literal(value: i64) -> String {
    if value < 0 {
        format!("-64'sd{}", value.unsigned_abs())
    } else {
        format!("64'sd{value}")
    }
}
