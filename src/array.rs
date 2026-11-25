use crate::bit2::Bit2Val;
use crate::error::{QuatError, Result};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Bit2Array {
    pub data: Vec<u8>,
    pub len: usize,
    pub scale: f32, // Added quantization scale factor[cite: 6]
}

impl Bit2Array {
    /// Creates a new Bit2Array with given length
    pub fn new(len: usize) -> Self {
        let byte_len = (len + 3) / 4;
        Bit2Array {
            data: vec![0; byte_len],
            len,
            scale: 1.0,
        }
    }

    /// Creates a Bit2Array from an f32 slice using quantization
    pub fn from_f32_slice(input: &[f32]) -> Self {
        let mut arr = Self::new(input.len());
        let max_abs = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        arr.scale = if max_abs == 0.0 { 1.0 } else { max_abs };

        for (i, &val) in input.iter().enumerate() {
            let q = Bit2Val::quantize(val, arr.scale);
            arr.set(i, q).unwrap();
        }
        arr
    }

    #[cfg(target_arch = "aarch64")]
pub fn dot_f32_neon(&self, input: &[f32]) -> Result<f32> {
    use std::arch::aarch64::*;

    if self.len != input.len() {
        return Err(Quaterror::DimensionMismatch {
            expected: self.len,
            got: input.len(),
        });
    }

    // SAFETY: `self.len == input.len()` is checked above, `chunks * 8 <= self.len`, and
    // `unpacked` has 8 elements; NEON loads read within bounds and `target_arch = "aarch64"`.
    unsafe {
        let mut sum_vec = vdupq_n_f32(0.0);
        let chunks = self.len / 8;
        let base_len = chunks * 8;

        for i in 0..chunks {
            let base_idx = i * 8;
            let mut unpacked = [0.0f32; 8];

            for j in 0..8 {
                let idx = base_idx + j;
                let byte_idx = idx / 4;
                let bit_shift = (idx % 4) * 2;
                let b_val = (self.data[byte_idx] >> bit_shift) & 0b11;
                unpacked[j] = Bit2Val::from_u8(b_val).to_f32();
            }

            // Load into NEON registers (two 128-bit vectors of 4 floats)
            let w_reg1 = vld1q_f32(unpacked.as_ptr());
            let w_reg2 = vld1q_f32(unpacked.as_ptr().add(4));
            
            let i_reg1 = vld1q_f32(input.as_ptr().add(base_idx));
            let i_reg2 = vld1q_f32(input.as_ptr().add(base_idx + 4));

            // Fused Multiply-Add (FMA) equivalents in NEON
            sum_vec = vfmaq_f32(sum_vec, w_reg1, i_reg1);
            sum_vec = vfmaq_f32(sum_vec, w_reg2, i_reg2);
        }

        // Horizontal sum of the 4 lanes in the NEON register
        let mut total_sum = vaddvq_f32(sum_vec);

        // Handle remainder elements sequentially
        for idx in base_len..self.len {
            let byte_idx = idx / 4;
            let bit_shift = (idx % 4) * 2;
            let b_val = (self.data[byte_idx] >> bit_shift) & 0b11;
            let weight = Bit2Val::from_u8(b_val).to_f32();
            total_sum += weight * input[idx];
        }

        Ok(total_sum * self.scale)
    }
}

    // Standardna skalarna petlja kao bezbedni fallback
    fn dot_f32_scalar(&self, input: &[f32]) -> Result<f32> {
        let mut total_sum = 0.0f32;
        for idx in 0..self.len {
            let byte_idx = idx / 4;
            let bit_shift = (idx % 4) * 2;
            let b_val = (self.data[byte_idx] >> bit_shift) & 0b11;
            let weight = Bit2Val::from_u8(b_val).to_f32();
            total_sum += weight * input[idx];
        }
        Ok(total_sum * self.scale)
    }


    /// Safely retrieves a 2-bit value at a given index
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<Bit2Val> {
        if index >= self.len {
            return None;
        }
        let byte_idx = index / 4;
        let bit_shift = (index % 4) * 2;
        let byte = self.data[byte_idx];
        Some(Bit2Val::from_u8((byte >> bit_shift) & 0b11))
    }

    /// Safely sets a 2-bit value at a given index
    #[inline(always)]
    pub fn set(&mut self, index: usize, val: Bit2Val) -> Result<()> {
        if index >= self.len {
            return Err(QuatError::DimensionMismatch {
                expected: self.len,
                got: index,
            });
        }
        let byte_idx = index / 4;
        let bit_shift = (index % 4) * 2;
        let mask = !(0b11 << bit_shift);
        let val_u8 = val as u8;
        self.data[byte_idx] = (self.data[byte_idx] & mask) | (val_u8 << bit_shift);
        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn dot_f32_neon(&self, input: &[f32]) -> f32 {
        use std::arch::aarch64::*;
        // SAFETY: Caller guarantees `self.len == input.len()`. `unpacked` and `input`
        // have at least `base_idx + 8` elements for every `i in 0..chunks`.
        unsafe {
            let mut sum_vec = vdupq_n_f32(0.0);
            let chunks = self.len / 8;
            let base_len = chunks * 8;

            for i in 0..chunks {
                let base_idx = i * 8;
                let mut unpacked = [0.0_f32; 8];
                for j in 0..8 {
                    let idx = base_idx + j;
                    let byte_idx = idx / 4;
                    let bit_shift = (idx % 4) * 2;
                    let b_val = (self.data[byte_idx] >> bit_shift) & 0b11;
                    unpacked[j] = Bit2Val::from_u8(b_val).to_f32();
                }

                let w_reg1 = vld1q_f32(unpacked.as_ptr());
                let w_reg2 = vld1q_f32(unpacked.as_ptr().add(4));
                let i_reg1 = vld1q_f32(input.as_ptr().add(base_idx));
                let i_reg2 = vld1q_f32(input.as_ptr().add(base_idx + 4));

                sum_vec = vfmaq_f32(sum_vec, w_reg1, i_reg1);
                sum_vec = vfmaq_f32(sum_vec, w_reg2, i_reg2);
            }

            let mut total_sum = vaddvq_f32(sum_vec);
            for idx in base_len..self.len {
                let byte_idx = idx / 4;
                let bit_shift = (idx % 4) * 2;
                let b_val = (self.data[byte_idx] >> bit_shift) & 0b11;
                let weight = Bit2Val::from_u8(b_val).to_f32();
                total_sum += weight * input[idx];
            }

            total_sum * self.scale
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_f32_avx2(&self, input: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        // SAFETY: Caller only invokes this after verifying `is_x86_feature_detected!("avx2")`
        // and `self.len == input.len()`. Every `i in 0..chunks` reads 8 contiguous f32s.
        unsafe {
            let mut unpacked = [0.0_f32; 8];
            let mut sum_vec = _mm256_setzero_ps();
            let chunks = self.len / 8;

            for i in 0..chunks {
                let base_idx = i * 8;
                for j in 0..8 {
                    let idx = base_idx + j;
                    let byte_idx = idx / 4;
                    let bit_shift = (idx % 4) * 2;
                    let b_val = (self.data[byte_idx] >> bit_shift) & 0b11;
                    unpacked[j] = Bit2Val::from_u8(b_val).to_f32();
                }

                let w_reg = _mm256_loadu_ps(unpacked.as_ptr());
                let i_reg = _mm256_loadu_ps(input.as_ptr().add(base_idx));
                let prod = _mm256_mul_ps(w_reg, i_reg);
                sum_vec = _mm256_add_ps(sum_vec, prod);
            }

            let mut temp = [0.0_f32; 8];
            _mm256_storeu_ps(temp.as_mut_ptr(), sum_vec);
            let mut total_sum: f32 = temp.iter().sum();

            for idx in (chunks * 8)..self.len {
                let byte_idx = idx / 4;
                let bit_shift = (idx % 4) * 2;
                let b_val = (self.data[byte_idx] >> bit_shift) & 0b11;
                let weight = Bit2Val::from_u8(b_val).to_f32();
                total_sum += weight * input[idx];
            }

            total_sum * self.scale
        }
    }

    pub fn dot_f32(&self, input: &[f32]) -> Result<f32> {
        if self.len != input.len() {
            return Err(QuatError::DimensionMismatch {
                expected: self.len,
                got: input.len(),
            });
        }

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: Target architecture is `aarch64` and slice lengths match.
            return Ok(unsafe { self.dot_f32_neon(input) });
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: Runtime CPU check confirmed AVX2 support and slice lengths match.
                return Ok(unsafe { self.dot_f32_avx2(input) });
            }
        }

        self.dot_f32_scalar(input)
    }

}

#[derive(Debug, Clone)]
pub struct BlockQuantizedArray {
    pub data: Vec<u8>,       // 2-bit packed weights
    pub scales: Vec<f32>,    // One scale factor per block
    pub block_size: usize,   // e.g., 32 or 64
    pub len: usize,
}

impl BlockQuantizedArray {
    pub fn new(len: usize, block_size: usize) -> Self {
        let num_blocks = (len + block_size - 1) / block_size;
        let byte_len = (len + 3) / 4;
        Self {
            data: vec![0; byte_len],
            scales: vec![1.0; num_blocks],
            block_size,
            len,
        }
    }

    pub fn set_block_scale(&mut self, block_idx: usize, scale: f32) {
        self.scales[block_idx] = scale;
    }

    /// Creates a block-quantized array using calibrated affine bounds per block.
    pub fn from_f32_slice_calibrated(
        input: &[f32],
        block_size: usize,
        clip_ratio: f32,
    ) -> Self {
        let bs = block_size.max(1);
        let len = input.len();
        let num_blocks = (len + bs - 1) / bs;
        let byte_len = (len + 3) / 4;

        let mut data = vec![0u8; byte_len];
        let mut scales = vec![1.0_f32; num_blocks];

        for b in 0..num_blocks {
            let start = b * bs;
            let end = (start + bs).min(len);
            let block = &input[start..end];
            let max_abs = block
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f32, f32::max);
            let scale = if max_abs == 0.0 {
                1.0
            } else {
                max_abs * (1.0 - clip_ratio.clamp(0.0, 0.25))
            };
            scales[b] = scale.max(1e-8);

            for idx in start..end {
                let q = Bit2Val::quantize(input[idx], scales[b]) as u8;
                let byte_idx = idx / 4;
                let bit_shift = (idx % 4) * 2;
                data[byte_idx] |= (q & 0b11) << bit_shift;
            }
        }

        Self {
            data,
            scales,
            block_size: bs,
            len,
        }
    }

    /// Computes dot product against affine-quantized activations with f32 accumulation.
    pub fn dot_quantized_activations(&self, act: &QuantizedActivations) -> Result<f32> {
        if self.len != act.len {
            return Err(QuatError::DimensionMismatch {
                expected: self.len,
                got: act.len,
            });
        }

        let mut total_sum = 0.0_f32;
        let num_blocks = self.scales.len();
        let act_bs = act.block_size.max(1);

        for b in 0..num_blocks {
            let start = b * self.block_size;
            let end = (start + self.block_size).min(self.len);
            let w_scale = self.scales[b];
            let mut block_sum = 0.0_f32;

            for idx in start..end {
                let byte_idx = idx / 4;
                let bit_shift = (idx % 4) * 2;

                let w_bits = (self.data[byte_idx] >> bit_shift) & 0b11;
                let w_val = Bit2Val::from_u8(w_bits).to_f32() * w_scale;

                let act_blk = idx / act_bs;
                let a_bits = (act.data[byte_idx] >> bit_shift) & 0b11;
                let a_val = Bit2Val::from_u8(a_bits)
                    .to_f32_affine(act.scales[act_blk], act.zero_points[act_blk]);

                block_sum += w_val * a_val;
            }

            total_sum += block_sum;
        }

        Ok(total_sum)
    }

    pub fn dot(&self, input: &[f32]) -> Result<f32> {
        if self.len != input.len() {
            return Err(QuatError::DimensionMismatch {
                expected: self.len,
                got: input.len(),
            });
        }

        let mut total_sum = 0.0_f32;
        let num_blocks = self.scales.len();

        for b in 0..num_blocks {
            let start = b * self.block_size;
            let end = (start + self.block_size).min(self.len);
            let scale = self.scales[b];
            let mut block_sum = 0.0_f32;

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

#[derive(Debug, Clone)]
pub struct QuantizedActivations {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub zero_points: Vec<f32>,
    pub block_size: usize,
    pub len: usize,
}

impl QuantizedActivations {
    pub fn from_f32_slice(input: &[f32], block_size: usize, clip_ratio: f32) -> Self {
        let bs = block_size.max(1);
        let len = input.len();
        let num_blocks = (len + bs - 1) / bs;
        let byte_len = (len + 3) / 4;

        let mut data = vec![0u8; byte_len];
        let mut scales = vec![1.0_f32; num_blocks];
        let mut zero_points = vec![1.5_f32; num_blocks];

        for b in 0..num_blocks {
            let start = b * bs;
            let end = (start + bs).min(len);
            let (scale, zp) = Bit2Val::calibrate_affine(&input[start..end], clip_ratio);
            scales[b] = scale;
            zero_points[b] = zp;

            for idx in start..end {
                let q = Bit2Val::quantize_affine(input[idx], scale, zp) as u8;
                let byte_idx = idx / 4;
                let bit_shift = (idx % 4) * 2;
                data[byte_idx] |= (q & 0b11) << bit_shift;
            }
        }

        Self {
            data,
            scales,
            zero_points,
            block_size: bs,
            len,
        }
    }
}