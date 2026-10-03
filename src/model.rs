use crate::array::Bit2Array;
use crate::bit2::Bit2Val;

/// 2-bit Linear Layer (Weight matrix + forward pass)
pub struct Bit2Linear {
    pub in_features: usize,
    pub out_features: usize,
    /// Vector of 2-bit packed rows (one Bit2Array per output feature)
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

    /// Sets a weight value in the linear layer matrix
    pub fn set_weight(&mut self, row: usize, col: usize, val: Bit2Val) {
        if row < self.out_features {
            self.weights[row].set(col, val);
        }
    }

    /// Forward pass: multiplies input vector with 2-bit weight matrix
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.in_features);
        assert_eq!(output.len(), self.out_features);

        for (row_idx, row_weights) in self.weights.iter().enumerate() {
            output[row_idx] = row_weights.dot_f32(input);
        }
    }
}