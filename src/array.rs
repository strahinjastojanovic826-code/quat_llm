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

    /// Fast parallelized dot product using Rayon (Production-Ready & Cache-Optimized)
    pub fn dot_f32(&self, input: &[f32]) -> Result<f32> {
        if self.len != input.len() {
            return Err(QuatError::DimensionMismatch {
                expected: self.len,
                got: input.len(),
            });
        }

        let chunk_size = 256;
        let sum: f32 = self.data
            .par_chunks(chunk_size / 4)
            .enumerate()
            .map(|(chunk_idx, byte_chunk)| {
                let mut local_sum = 0.0f64;
                let base_i = chunk_idx * chunk_size;

                for (b_i, &byte) in byte_chunk.iter().enumerate() {
                    let byte_offset = base_i + (b_i * 4);
                    for shift in (0..8).step_by(2) {
                        let val_idx = byte_offset + (shift / 2);
                        if val_idx < self.len {
                            let b_val = Bit2Val::from_u8((byte >> shift) & 0b11);
                            local_sum += (b_val.to_f32() as f64) * (input[val_idx] as f64);
                        }
                    }
                }
                local_sum as f32
            })
            .sum::<f32>();

        Ok(sum * self.scale)
    }
}