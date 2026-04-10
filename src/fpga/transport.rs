use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FpgaSimRequest {
    pub operation: String,
    pub audio_path: String,
    pub vector_a: Vec<i16>,
    pub vector_b: Vec<i16>,
    pub expected_result: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FpgaSimResponse {
    pub operation: String,
    pub rtl_result: i64,
    pub expected_result: i64,
    pub matched: bool,
    pub notes: Vec<String>,
}

pub trait FpgaExecutor {
    fn name(&self) -> &'static str;
    fn execute_stage(&self, request: &FpgaSimRequest, output_dir: &Path)
    -> Result<FpgaSimResponse>;
}
