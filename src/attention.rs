use crate::error::{QuatError, Result};
use crate::model::Bit2Linear;
use crate::ops::{apply_rope, softmax};

#[derive(Debug, Clone)]
pub struct KVCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl KVCache {
    pub fn new() -> Self {
        KVCache {
            k: Vec::new(),
            v: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Bit2MultiHeadAttention {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_proj: Bit2Linear,
    pub k_proj: Bit2Linear,
    pub v_proj: Bit2Linear,
    pub out_proj: Bit2Linear,
}

impl Bit2MultiHeadAttention {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        Bit2MultiHeadAttention {
            hidden_dim,
            num_heads,
            head_dim,
            q_proj: Bit2Linear::new(hidden_dim, hidden_dim),
            k_proj: Bit2Linear::new(hidden_dim, hidden_dim),
            v_proj: Bit2Linear::new(hidden_dim, hidden_dim),
            out_proj: Bit2Linear::new(hidden_dim, hidden_dim),
        }
    }

    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        cache: &mut KVCache,
        pos: usize,
    ) -> Result<()> {
        let mut q = vec![0.0f32; self.hidden_dim];
        let mut k = vec![0.0f32; self.hidden_dim];
        let mut v = vec![0.0f32; self.hidden_dim];

        self.q_proj.forward(input, &mut q)?;
        self.k_proj.forward(input, &mut k)?;
        self.v_proj.forward(input, &mut v)?;

        apply_rope(&mut q, pos, self.head_dim)?;
        apply_rope(&mut k, pos, self.head_dim)?;

        cache.k.push(k);
        cache.v.push(v);

        let seq_len = cache.k.len();
        let mut attn_out = vec![0.0f32; self.hidden_dim];

        for h in 0..self.num_heads {
            let h_off = h * self.head_dim;
            let q_head = &q[h_off..h_off + self.head_dim];

            let mut scores = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let k_head = &cache.k[t][h_off..h_off + self.head_dim];
                let score: f32 = q_head.iter().zip(k_head.iter()).map(|(a, b)| a * b).sum();
                scores[t] = score / (self.head_dim as f32).sqrt();
            }

            softmax(&mut scores)?;

            for t in 0..seq_len {
                let v_head = &cache.v[t][h_off..h_off + self.head_dim];
                for d in 0..self.head_dim {
                    attn_out[h_off + d] += scores[t] * v_head[d];
                }
            }
        }

        self.out_proj.forward(&attn_out, output)?;
        Ok(())
    }
}