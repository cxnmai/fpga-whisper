use anyhow::{Result, bail};

use crate::fpga::kernels::gemm::{
    GemmComparison, MatrixI16, MatrixI64, simulate_gemm_tile_with_accumulator,
};
use crate::fpga::quant::FixedPointConfig;
use crate::fpga::transport::FpgaExecutor;

#[derive(Debug, Clone)]
pub struct LinearLayerI16 {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: MatrixI16,
    pub bias: Vec<i16>,
    pub quant: FixedPointConfig,
}

#[derive(Debug, Clone)]
pub struct LinearComparison {
    pub gemm: GemmComparison,
    pub software_output: Vec<i64>,
    pub rtl_output: Vec<i64>,
    pub matched: bool,
    pub notes: Vec<String>,
}

impl LinearLayerI16 {
    pub fn validate(&self) -> Result<()> {
        if self.weights.rows != self.input_dim || self.weights.cols != self.output_dim {
            bail!(
                "weight matrix shape mismatch: expected {}x{}, got {}x{}",
                self.input_dim,
                self.output_dim,
                self.weights.rows,
                self.weights.cols
            );
        }
        if self.bias.len() != self.output_dim {
            bail!(
                "bias length mismatch: expected {}, got {}",
                self.output_dim,
                self.bias.len()
            );
        }
        Ok(())
    }
}

pub fn software_linear(layer: &LinearLayerI16, input: &[i16]) -> Result<Vec<i64>> {
    layer.validate()?;
    if input.len() != layer.input_dim {
        bail!(
            "input length mismatch: expected {}, got {}",
            layer.input_dim,
            input.len()
        );
    }

    let input_matrix = MatrixI16::new(1, layer.input_dim, input.to_vec());
    let bias_accumulator = MatrixI64::new(
        1,
        layer.output_dim,
        layer
            .bias
            .iter()
            .map(|bias| layer.quant.bias_to_accumulator(*bias))
            .collect(),
    );
    let gemm = crate::fpga::kernels::gemm::software_gemm_with_accumulator(
        &input_matrix,
        &layer.weights,
        Some(&bias_accumulator),
    )?;
    Ok(gemm.values)
}

pub fn simulate_linear(
    executor: &dyn FpgaExecutor,
    output_dir: &std::path::Path,
    audio_path: &str,
    layer: &LinearLayerI16,
    input: &[i16],
) -> Result<LinearComparison> {
    layer.validate()?;
    if input.len() != layer.input_dim {
        bail!(
            "input length mismatch: expected {}, got {}",
            layer.input_dim,
            input.len()
        );
    }

    let input_matrix = MatrixI16::new(1, layer.input_dim, input.to_vec());
    let bias_accumulator = MatrixI64::new(
        1,
        layer.output_dim,
        layer
            .bias
            .iter()
            .map(|bias| layer.quant.bias_to_accumulator(*bias))
            .collect(),
    );
    let gemm = simulate_gemm_tile_with_accumulator(
        executor,
        output_dir,
        audio_path,
        &input_matrix,
        &layer.weights,
        Some(&bias_accumulator),
    )?;
    let software_output = software_linear(layer, input)?;
    let rtl_output = gemm.rtl.values.clone();
    let matched = gemm.matched && software_output == rtl_output;

    let mut notes = vec![
        format!(
            "Linear layer quantization contract: {}",
            layer.quant.description()
        ),
        format!("Bias values: {:?}", layer.bias),
    ];
    notes.extend(gemm.notes.clone());

    Ok(LinearComparison {
        gemm,
        software_output,
        rtl_output,
        matched,
        notes,
    })
}

pub fn format_vector_i64(values: &[i64]) -> String {
    let parts = values.iter().map(ToString::to_string).collect::<Vec<_>>();
    format!("[{}]", parts.join(", "))
}
