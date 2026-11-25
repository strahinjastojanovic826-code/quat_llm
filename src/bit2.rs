use crate::QuatError;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bit2Val {
    Val00 = 0b00, // -1.0
    Val01 = 0b01, // -0.33333333
    Val10 = 0b10, //  0.33333333
    Val11 = 0b11, //  1.0
}

impl Bit2Val {
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        match self {
            Bit2Val::Val00 => -1.0,
            Bit2Val::Val01 => -0.33333333,
            Bit2Val::Val10 => 0.33333333,
            Bit2Val::Val11 => 1.0,
        }
    }

    #[inline(always)]
    pub fn from_u8(val: u8) -> Self {
        match val & 0b11 {
            0b00 => Bit2Val::Val00,
            0b01 => Bit2Val::Val01,
            0b10 => Bit2Val::Val10,
            _ => Bit2Val::Val11,
        }
    }

    /// Quantize a float value to 2-bit representation based on a scale factor
    #[inline(always)]
    pub fn quantize(val: f32, scale: f32) -> Self {
        let norm = if scale == 0.0 { 0.0 } else { val / scale };
        if norm < -0.6666666 {
            Bit2Val::Val00
        } else if norm < 0.0 {
            Bit2Val::Val01
        } else if norm < 0.6666666 {
            Bit2Val::Val10
        } else {
            Bit2Val::Val11
        }
    }

    #[inline(always)]
    pub fn to_f32_affine(self, scale: f32, zero_point: f32) -> f32 {
        let q_idx = match self {
            Bit2Val::Val00 => 0.0_f32,
            Bit2Val::Val01 => 1.0_f32,
            Bit2Val::Val10 => 2.0_f32,
            Bit2Val::Val11 => 3.0_f32,
        };
        (q_idx - zero_point) * scale
    }

    #[inline(always)]
    pub fn quantize_affine(val: f32, scale: f32, zero_point: f32) -> Self {
        if scale == 0.0 || !scale.is_finite() {
            return Bit2Val::Val01;
        }
        let q = ((val / scale) + zero_point).round().clamp(0.0, 3.0) as u8;
        Self::from_u8(q)
    }

    /// Calibrates (scale, zero_point) from a slice using optional percentile/clipping
    /// to reduce outlier distortion in 2-bit quaternary quantization.
    pub fn calibrate_affine(values: &[f32], clip_ratio: f32) -> (f32, f32) {
        if values.is_empty() {
            return (1.0, 1.5);
        }
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        for &v in values {
            if v.is_finite() {
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
        }
        if !min_v.is_finite() || !max_v.is_finite() || (max_v - min_v).abs() < 1e-12 {
            return (1.0, 1.5);
        }
        let clamped_ratio = clip_ratio.clamp(0.0, 0.25);
        let span = max_v - min_v;
        let cal_min = min_v + span * clamped_ratio;
        let cal_max = max_v - span * clamped_ratio;
        let scale = ((cal_max - cal_min) / 3.0).max(1e-8);
        let zero_point = (-cal_min / scale).clamp(0.0, 3.0);
        (scale, zero_point)
    }
    
}

#[derive(Debug, Clone)]
pub struct BlockQuantizedArray {
    pub data: Vec<u8>,          // Packed 2-bit values (4 per byte)
    pub scales: Vec<f32>,       // One scale factor per block
    pub block_size: usize,      // e.g., 32 or 64 weights per block
    pub len: usize,
}

impl BlockQuantizedArray {
    /// Creates a new block-quantized array from f32 slice with group-wise scaling
    pub fn from_f32_slice(input: &[f32], block_size: usize) -> Self {
        let len = input.len();
        let num_blocks = (len + block_size - 1) / block_size;
        let byte_len = (len + 3) / 4;
        
        let mut data = vec![0u8; byte_len];
        let mut scales = vec![1.0f32; num_blocks];

        for b in 0..num_blocks {
            let start = b * block_size;
            let end = std::cmp::min(start + block_size, len);
            let block = &input[start..end];

            // Find max absolute value in this specific block for precise scaling
            let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs };
            scales[b] = scale;

            // Quantize block values
            for (i, &val) in block.iter().enumerate() {
                let global_idx = start + i;
                let q = Bit2Val::quantize(val, scale);
                
                let byte_idx = global_idx / 4;
                let bit_shift = (global_idx % 4) * 2;
                let val_u8 = q as u8;
                
                // Pack into byte safely
                data[byte_idx] |= val_u8 << bit_shift;
            }
        }

        Self {
            data,
            scales,
            block_size,
            len,
        }
    }

    /// Group-wise dot product using per-block scales
    pub fn dot(&self, input: &[f32]) -> crate::error::Result<f32> {
        if self.len != input.len() {
            return Err(QuatError::DimensionMismatch {
                expected: self.len,
                got: input.len(),
            });
        }

        let mut total_sum = 0.0;
        let num_blocks = self.scales.len();

        for b in 0..num_blocks {
            let start = b * self.block_size;
            let end = std::cmp::min(start + self.block_size, self.len);
            let scale = self.scales[b];

            let mut block_sum = 0.0;
            for idx in start..end {
                let byte_idx = idx / 4;
                let bit_shift = (idx % 4) * 2;
                let b_val = (self.data[byte_idx] >> bit_shift) & 0b11;
                let weight = Bit2Val::from_u8(b_val).to_f32();
                
                block_sum += weight * input[idx];
            }
            total_sum += block_sum * scale;
        }

        Ok(total_sum)
    }
}