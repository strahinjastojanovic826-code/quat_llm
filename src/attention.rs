use crate::model::Bit2Linear;
use crate::ops;

/// 2-bit Quantized Self-Attention Head
pub struct Bit2Attention {
    pub hidden_dim: usize,
    pub q_proj: Bit2Linear,
    pub k_proj: Bit2Linear,
    pub v_proj: Bit2Linear,
    pub out_proj: Bit2Linear,
}

impl Bit2Attention {
    pub fn new(hidden_dim: usize) -> Self {
        Bit2Attention {
            hidden_dim,
            q_proj: Bit2Linear::new(hidden_dim, hidden_dim),
            k_proj: Bit2Linear::new(hidden_dim, hidden_dim),
            v_proj: Bit2Linear::new(hidden_dim, hidden_dim),
            out_proj: Bit2Linear::new(hidden_dim, hidden_dim),
        }
    }

    /// Single sequence step forward pass
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        let mut q = vec![0.0f32; self.hidden_dim];
        let mut k = vec![0.0f32; self.hidden_dim];
        let mut v = vec![0.0f32; self.hidden_dim];

        // 1. Project input with 2-bit weights to Q, K, V
        self.q_proj.forward(input, &mut q);
        self.k_proj.forward(input, &mut k);
        self.v_proj.forward(input, &mut v);

        // 2. Compute attention score (dot product Q * K)
        let mut score = 0.0f32;
        for i in 0..self.hidden_dim {
            score += q[i] * k[i];
        }
        score /= (self.hidden_dim as f32).sqrt();

        // 3. Apply Softmax and scale Value projection
        let mut attn_weights = vec![score];
        ops::softmax(&mut attn_weights);

        let mut attn_out = vec![0.0f32; self.hidden_dim];
        for i in 0..self.hidden_dim {
            attn_out[i] = v[i] * attn_weights[0];
        }

        // 4. Final output projection
        self.out_proj.forward(&attn_out, output);
    }
}