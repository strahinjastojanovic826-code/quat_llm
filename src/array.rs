use crate::bit2::Bit2Val;
use crate::error::{QuatError, Result};
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

    pub fn dot_f32(&self, input: &[f32]) -> Result<f32> {
        if self.len != input.len() {
            return Err(QuatError::DimensionMismatch {
                expected: self.len,
                got: input.len(),
            });
        }

        // Provera da li procesor podržava AVX2 u realnom vremenu
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: Pozivamo unsafe AVX2 kod samo ako je provereno da procesor podržava AVX2
                return unsafe { Ok(self.dot_f32_avx2(input)) };
            }
        }

        // Fallback (rezervna varijanta) za procesore bez AVX2 podrške
        self.dot_f32_scalar(input)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_f32_avx2(&self, input: &[f32]) -> f32 { unsafe {
        let mut unpacked = [0.0f32; 8];
        let mut sum_vec = _mm256_setzero_ps();
        let chunks = self.len / 8;

        for i in 0..chunks {
            let base_idx = i * 8;
            
            // Otpakivanje 8 elemenata iz 2-bitnih pakovanih bajtova
            for j in 0..8 {
                let idx = base_idx + j;
                let byte_idx = idx / 4;
                let bit_shift = (idx % 4) * 2;
                let b_val = (self.data[byte_idx] >> bit_shift) & 0b11;
                unpacked[j] = Bit2Val::from_u8(b_val).to_f32();
            }

            // Učitavanje u AVX registre (8 x f32 odjednom)
            let w_reg = _mm256_loadu_ps(unpacked.as_ptr());
            let i_reg = _mm256_loadu_ps(input[base_idx..].as_ptr());

            // FMA / Množenje i sabiranje u jednom ciklusu
            let prod = _mm256_mul_ps(w_reg, i_reg);
            sum_vec = _mm256_add_ps(sum_vec, prod);
        }

        // Horizontalno sakupljanje rezultata iz 256-bitnog registra
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), sum_vec);
        let mut total_sum: f32 = temp.iter().sum();

        // Obrada ostatka elemenata ako dužina nije deljiva sa 8
        for idx in (chunks * 8)..self.len {
            let byte_idx = idx / 4;
            let bit_shift = (idx % 4) * 2;
            let b_val = (self.data[byte_idx] >> bit_shift) & 0b11;
            let weight = Bit2Val::from_u8(b_val).to_f32();
            total_sum += weight * input[idx];
        }

        total_sum * self.scale
    }}

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
}