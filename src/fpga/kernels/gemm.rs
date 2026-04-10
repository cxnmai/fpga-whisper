use anyhow::{Result, bail};

use crate::fpga::kernels::dot::{simulate_dot_product, software_dot_product};
use crate::fpga::transport::FpgaExecutor;

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

    pub fn get(&self, row: usize, col: usize) -> i64 {
        i64::from(self.values[row * self.cols + col])
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
            let rhs_column = (0..rhs.rows)
                .map(|rhs_row| rhs.values[rhs_row * rhs.cols + col])
                .collect::<Vec<_>>();
            values.push(software_dot_product(lhs.row(row), &rhs_column));
        }
    }

    Ok(MatrixI64::new(lhs.rows, rhs.cols, values))
}

pub fn simulate_gemm_via_dot_products(
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
    if lhs.cols != 8 {
        bail!("current dot-product RTL path expects inner dimension of exactly 8");
    }

    let software = software_gemm(lhs, rhs)?;
    let mut rtl_values = Vec::with_capacity(lhs.rows * rhs.cols);
    let mut notes = Vec::new();
    let mut matched = true;

    for row in 0..lhs.rows {
        for col in 0..rhs.cols {
            let rhs_column = (0..rhs.rows)
                .map(|rhs_row| rhs.values[rhs_row * rhs.cols + col])
                .collect::<Vec<_>>();
            let result =
                simulate_dot_product(executor, output_dir, audio_path, lhs.row(row), &rhs_column)?;
            matched &= result.matched;
            rtl_values.push(result.rtl_result);
            notes.extend(result.notes);
        }
    }

    Ok(GemmComparison {
        software,
        rtl: MatrixI64::new(lhs.rows, rhs.cols, rtl_values),
        matched,
        notes,
    })
}
