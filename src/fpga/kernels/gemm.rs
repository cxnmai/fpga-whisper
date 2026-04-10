use anyhow::{Result, bail};

use crate::fpga::transport::{FpgaExecutor, GemmTileI16Request, GemmTileShape};

#[derive(Debug, Clone)]
pub struct MatrixI16 {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<i16>,
}

impl MatrixI16 {
    pub fn new(rows: usize, cols: usize, values: Vec<i16>) -> Self {
        Self { rows, cols, values }
    }

    pub fn row(&self, index: usize) -> &[i16] {
        let start = index * self.cols;
        let end = start + self.cols;
        &self.values[start..end]
    }
}

#[derive(Debug, Clone)]
pub struct MatrixI64 {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<i64>,
}

impl MatrixI64 {
    pub fn new(rows: usize, cols: usize, values: Vec<i64>) -> Self {
        Self { rows, cols, values }
    }

    pub fn get(&self, row: usize, col: usize) -> i64 {
        self.values[row * self.cols + col]
    }
}

#[derive(Debug, Clone)]
pub struct GemmComparison {
    pub software: MatrixI64,
    pub rtl: MatrixI64,
    pub matched: bool,
    pub notes: Vec<String>,
}

pub fn software_gemm(lhs: &MatrixI16, rhs: &MatrixI16) -> Result<MatrixI64> {
    if lhs.cols != rhs.rows {
        bail!(
            "incompatible GEMM dimensions: lhs {}x{}, rhs {}x{}",
            lhs.rows,
            lhs.cols,
            rhs.rows,
            rhs.cols
        );
    }

    let mut values = Vec::with_capacity(lhs.rows * rhs.cols);
    for row in 0..lhs.rows {
        for col in 0..rhs.cols {
            let mut sum = 0_i64;
            for inner in 0..lhs.cols {
                let lhs_value = i64::from(lhs.values[row * lhs.cols + inner]);
                let rhs_value = i64::from(rhs.values[inner * rhs.cols + col]);
                sum += lhs_value * rhs_value;
            }
            values.push(sum);
        }
    }

    Ok(MatrixI64::new(lhs.rows, rhs.cols, values))
}

pub fn simulate_gemm_tile(
    executor: &dyn FpgaExecutor,
    output_dir: &std::path::Path,
    audio_path: &str,
    lhs: &MatrixI16,
    rhs: &MatrixI16,
) -> Result<GemmComparison> {
    if lhs.cols != rhs.rows {
        bail!(
            "incompatible GEMM dimensions: lhs {}x{}, rhs {}x{}",
            lhs.rows,
            lhs.cols,
            rhs.rows,
            rhs.cols
        );
    }

    let software = software_gemm(lhs, rhs)?;
    let response = executor.execute_gemm_tile(
        &GemmTileI16Request {
            audio_path: audio_path.to_owned(),
            shape: GemmTileShape {
                rows: lhs.rows,
                cols: rhs.cols,
                inner: lhs.cols,
            },
            lhs_tile: lhs.values.clone(),
            rhs_tile: rhs.values.clone(),
            expected_output: software.values.clone(),
        },
        output_dir,
    )?;

    Ok(GemmComparison {
        software,
        rtl: MatrixI64::new(lhs.rows, rhs.cols, response.rtl_output),
        matched: response.matched,
        notes: response.notes,
    })
}
