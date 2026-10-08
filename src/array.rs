use crate::bit2::Bit2Val;
use crate::error::{QuatError, Result};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Bit2Array {
    pub data: Vec<u8>,
    pub len: usize,
    pub scale: f32, // Added quantization scale factor
}

impl Bit2Array {
    pub fn new(len: usize) -> Self {
        let byte_len = (len + 3) / 4;
        Bit2Array {
            data: vec![0; byte_len],
            len,
            scale: 1.0,
        }
    }

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
        self.data[byte_idx] = (self.data[byte_idx] & mask) | ((val as u8) << bit_shift);
        Ok(())
    }

    /// Fast parallelized dot product using Rayon
    pub fn dot_f32(&self, input: &[f32]) -> Result<f32> {
        if self.len != input.len() {
            return Err(QuatError::DimensionMismatch {
                expected: self.len,
                got: input.len(),
            });
        }

        let sum: f32 = (0..self.len)
            .into_par_iter()
            .map(|i| {
                let b_val = unsafe { self.get(i).unwrap_unchecked() };
                b_val.to_f32() * input[i]
            })
            .sum();

        Ok(sum * self.scale)
    }
}