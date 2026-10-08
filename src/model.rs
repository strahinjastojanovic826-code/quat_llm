use crate::array::Bit2Array;
use crate::bit2::Bit2Val;
use crate::error::{QuatError, Result};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Bit2Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Vec<Bit2Array>,
}

impl Bit2Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut weights = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            weights.push(Bit2Array::new(in_features));
        }
        Bit2Linear {
            in_features,
            out_features,
            weights,
        }
    }

    pub fn set_weight(&mut self, row: usize, col: usize, val: Bit2Val) -> Result<()> {
        if row >= self.out_features {
            return Err(QuatError::DimensionMismatch {
                expected: self.out_features,
                got: row,
            });
        }
        self.weights[row].set(col, val)
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.in_features || output.len() != self.out_features {
            return Err(QuatError::DimensionMismatch {
                expected: self.in_features,
                got: input.len(),
            });
        }

        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, out_val)| {
                *out_val = self.weights[row_idx].dot_f32(input).unwrap_or(0.0);
            });

        Ok(())
    }
}