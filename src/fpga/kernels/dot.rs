use anyhow::Result;

use crate::fpga::transport::{DotProductRequest, FpgaExecutor};

#[derive(Debug, Clone)]
pub struct DotProductResult {
    pub rtl_result: i64,
    pub expected_result: i64,
    pub matched: bool,
    pub notes: Vec<String>,
}

pub fn software_dot_product(lhs: &[i16], rhs: &[i16]) -> i64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| i64::from(*left) * i64::from(*right))
        .sum()
}

pub fn simulate_dot_product(
    executor: &dyn FpgaExecutor,
    output_dir: &std::path::Path,
    audio_path: &str,
    lhs: &[i16],
    rhs: &[i16],
) -> Result<DotProductResult> {
    let expected_result = software_dot_product(lhs, rhs);
    let response = executor.execute_dot_product(
        &DotProductRequest {
            audio_path: audio_path.to_owned(),
            vector_a: lhs.to_vec(),
            vector_b: rhs.to_vec(),
            expected_result,
        },
        output_dir,
    )?;

    Ok(DotProductResult {
        rtl_result: response.rtl_result,
        expected_result: response.expected_result,
        matched: response.matched,
        notes: response.notes,
    })
}
