use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FpgaFeatureRequest {
    pub audio_path: String,
    pub requested_stage: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FpgaFeatureResponse {
    pub produced_stage: String,
    pub sample_rate_hz: u32,
    pub frame_count: usize,
    pub bin_count: usize,
    pub notes: Vec<String>,
}

pub trait FpgaExecutor {
    fn name(&self) -> &'static str;
    fn execute_feature_stage(
        &self,
        request: &FpgaFeatureRequest,
        output_dir: &Path,
    ) -> Result<FpgaFeatureResponse>;
}
