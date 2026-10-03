/// Embedding layer mapping token IDs to hidden_dim float vectors
pub struct Embedding {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub weights: Vec<f32>, // Flatted matrix of size (vocab_size * hidden_dim)
}

impl Embedding {
    pub fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        Embedding {
            vocab_size,
            hidden_dim,
            weights: vec![0.1f32; vocab_size * hidden_dim], // Default init
        }
    }

    /// Look up vector representation for a token ID
    pub fn forward(&self, token_id: usize, output: &mut [f32]) {
        assert_eq!(output.len(), self.hidden_dim);
        let start_idx = token_id * self.hidden_dim;
        let end_idx = start_idx + self.hidden_dim;
        output.copy_from_slice(&self.weights[start_idx..end_idx]);
    }
}