use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::fpga::layout::TileShape;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DotProductRequest {
    pub audio_path: String,
    pub vector_a: Vec<i16>,
    pub vector_b: Vec<i16>,
    pub expected_result: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DotProductResponse {
    pub rtl_result: i64,
    pub expected_result: i64,
    pub matched: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmTileI16Request {
    pub audio_path: String,
    pub shape: GemmTileShape,
    pub lhs_tile: Vec<i16>,
    pub rhs_tile: Vec<i16>,
    pub expected_output: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmTileI64Response {
    pub shape: GemmTileShape,
    pub rtl_output: Vec<i64>,
    pub expected_output: Vec<i64>,
    pub matched: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GemmTileShape {
    pub rows: usize,
    pub cols: usize,
    pub inner: usize,
}

impl GemmTileShape {
    pub fn as_layout(self) -> TileShape {
        TileShape {
            rows: self.rows,
            cols: self.cols,
            inner: self.inner,
        }
    }
}

pub trait FpgaExecutor {
    fn name(&self) -> &'static str;
    fn execute_dot_product(
        &self,
        request: &DotProductRequest,
        output_dir: &Path,
    ) -> Result<DotProductResponse>;
    fn execute_gemm_tile(
        &self,
        request: &GemmTileI16Request,
        output_dir: &Path,
    ) -> Result<GemmTileI64Response>;
}
