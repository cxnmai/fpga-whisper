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

    pub fn scale_factor(self) -> f32 {
        (1_u32 << self.fractional_bits) as f32
    }

    pub fn quantize_scalar(self, value: f32) -> i16 {
        let scaled = (value * self.scale_factor()).round();
        scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }

    pub fn quantize_slice(self, values: &[f32]) -> Vec<i16> {
        values
            .iter()
            .map(|value| self.quantize_scalar(*value))
            .collect()
    }

    pub fn dequantize_scalar(self, value: i16) -> f32 {
        f32::from(value) / self.scale_factor()
    }

    pub fn bias_to_accumulator(self, value: i16) -> i64 {
        i64::from(value) << self.fractional_bits
    }

    pub fn dequantize_accumulator(self, value: i64) -> f32 {
        let scale = 1_u64 << (u64::from(self.fractional_bits) * 2);
        value as f32 / scale as f32
    }
}
