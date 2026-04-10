use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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

        fs::create_dir_all(output_dir)?;
        let request_path = output_dir.join("sim_request.json");
        let response_path = output_dir.join("sim_response.json");
        let vector_include_path = output_dir.join("dot_product_vectors.vh");
        let result_path = output_dir.join("dot_product_result.txt");
        let vvp_output_path = output_dir.join("dot_product_i16x8_tb.out");

        fs::write(&request_path, serde_json::to_vec_pretty(request)?)?;
        fs::write(
            &vector_include_path,
            build_vector_include(&request.vector_a, &request.vector_b),
        )?;

        let compile_output = Command::new(&self.iverilog)
            .arg("-g2012")
            .arg("-o")
            .arg(&vvp_output_path)
            .arg("fpga/rtl/dot_product_i16x8.v")
            .arg("fpga/tb/dot_product_i16x8_tb.v")
            .current_dir(&self.project_root)
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
            .current_dir(&self.project_root)
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
                format!(
                    "Waveform: {}",
                    output_dir.join("dot_product_i16x8_tb.vcd").display()
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
        fs::create_dir_all(output_dir)?;
        let request_path = output_dir.join("sim_request.json");
        let response_path = output_dir.join("sim_response.json");
        let vector_include_path = output_dir.join("gemm_tile_vectors.vh");
        let result_path = output_dir.join("gemm_tile_result.txt");
        let vvp_output_path = output_dir.join("gemm_tile_i16x8_tb.out");

        fs::write(&request_path, serde_json::to_vec_pretty(request)?)?;
        fs::write(
            &vector_include_path,
            build_gemm_tile_include(request.shape, &request.lhs_tile, &request.rhs_tile)?,
        )?;

        let compile_output = Command::new(&self.iverilog)
            .arg("-g2012")
            .arg("-o")
            .arg(&vvp_output_path)
            .arg("fpga/rtl/dot_product_i16x8.v")
            .arg("fpga/rtl/gemm_tile_i16x8.v")
            .arg("fpga/tb/gemm_tile_i16x8_tb.v")
            .current_dir(&self.project_root)
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
            .current_dir(&self.project_root)
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
                "Executed real RTL GEMM tile {}x{}x{} through direct Rust -> iverilog/vvp invocation.",
                request.shape.rows, request.shape.cols, request.shape.inner
            ),
            format!("Received audio path: {}", request.audio_path),
            "Operation: gemm-tile".to_owned(),
            format!(
                "Waveform: {}",
                output_dir.join("gemm_tile_i16x8_tb.vcd").display()
            ),
            "This tile primitive is the first realistic insertion point for linear layers."
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
) -> Result<String> {
    shape.as_layout().validate(lhs_tile.len(), rhs_tile.len())?;

    let lhs_bits = shape.rows * shape.inner * 16;
    let rhs_bits = shape.inner * shape.cols * 16;

    Ok(format!(
        "localparam integer TILE_ROWS = {};\nlocalparam integer TILE_COLS = {};\nlocalparam integer TILE_INNER = {};\nlocalparam signed [{}:0] LHS_TILE = {};\nlocalparam signed [{}:0] RHS_TILE = {};\n",
        shape.rows,
        shape.cols,
        shape.inner,
        lhs_bits - 1,
        build_packed_i16_literal(lhs_tile),
        rhs_bits - 1,
        build_packed_i16_literal(rhs_tile),
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
