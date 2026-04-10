#[derive(Debug, Clone, Copy)]
pub struct FixedPointConfig {
    pub bits: u8,
    pub fractional_bits: u8,
}

impl FixedPointConfig {
    pub const Q8_8: Self = Self {
        bits: 16,
        fractional_bits: 8,
    };

    pub fn description(self) -> String {
        format!(
            "Q{}.{}",
            self.bits - self.fractional_bits,
            self.fractional_bits
        )
    }
}
