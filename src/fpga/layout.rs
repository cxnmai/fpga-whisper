use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy)]
pub struct TileShape {
    pub rows: usize,
    pub cols: usize,
    pub inner: usize,
}

impl TileShape {
    pub fn validate(self, lhs_len: usize, rhs_len: usize) -> Result<()> {
        if lhs_len != self.rows * self.inner {
            bail!(
                "lhs tile length mismatch: expected {}, got {}",
                self.rows * self.inner,
                lhs_len
            );
        }
        if rhs_len != self.inner * self.cols {
            bail!(
                "rhs tile length mismatch: expected {}, got {}",
                self.inner * self.cols,
                rhs_len
            );
        }
        Ok(())
    }
}
