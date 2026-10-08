use crate::error::{QuatError, Result};

#[derive(Debug, Clone)]
pub struct Embedding {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub weights: Vec<f32>, // Spljoštena matrica dimenzija (vocab_size * hidden_dim)
}

impl Embedding {
    pub fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        Embedding {
            vocab_size,
            hidden_dim,
            weights: vec![0.1f32; vocab_size * hidden_dim], // Podrazumevana inicijalizacija
        }
    }

    /// Čitanje embedding vektora za dati token ID uz proveru granica
    pub fn forward(&self, token_id: usize, output: &mut [f32]) -> Result<()> {
        if token_id >= self.vocab_size {
            return Err(QuatError::DimensionMismatch {
                expected: self.vocab_size,
                got: token_id,
            });
        }

        if output.len() != self.hidden_dim {
            return Err(QuatError::DimensionMismatch {
                expected: self.hidden_dim,
                got: output.len(),
            });
        }

        let start_idx = token_id * self.hidden_dim;
        let end_idx = start_idx + self.hidden_dim;
        output.copy_from_slice(&self.weights[start_idx..end_idx]);

        Ok(())
    }

    /// Postavljanje težina za specifičan token ID (korisno pri učitavanju modela)
    pub fn set_embedding(&mut self, token_id: usize, values: &[f32]) -> Result<()> {
        if token_id >= self.vocab_size {
            return Err(QuatError::DimensionMismatch {
                expected: self.vocab_size,
                got: token_id,
            });
        }
        if values.len() != self.hidden_dim {
            return Err(QuatError::DimensionMismatch {
                expected: self.hidden_dim,
                got: values.len(),
            });
        }

        let start_idx = token_id * self.hidden_dim;
        let end_idx = start_idx + self.hidden_dim;
        self.weights[start_idx..end_idx].copy_from_slice(values);

        Ok(())
    }
}