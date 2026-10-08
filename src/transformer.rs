use crate::attention::{Bit2MultiHeadAttention, KVCache};
use crate::embedding::Embedding;
use crate::error::Result;
use crate::model::Bit2Linear;
use crate::ops::{rms_norm, silu, softmax};

#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub attention: Bit2MultiHeadAttention,
    pub ffn_gate: Bit2Linear,
    pub ffn_up: Bit2Linear,
    pub ffn_down: Bit2Linear,
}

impl TransformerLayer {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        TransformerLayer {
            attention: Bit2MultiHeadAttention::new(hidden_dim, num_heads),
            ffn_gate: Bit2Linear::new(hidden_dim, hidden_dim * 2),
            ffn_up: Bit2Linear::new(hidden_dim, hidden_dim * 2),
            ffn_down: Bit2Linear::new(hidden_dim * 2, hidden_dim),
        }
    }
}

pub struct Bit2Transformer {
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub embedding: Embedding,
    pub layers: Vec<TransformerLayer>,
    pub lm_head: Bit2Linear,
    pub norm_weight: Vec<f32>,
}

impl Bit2Transformer {
    pub fn new(
        vocab_size: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerLayer::new(hidden_dim, num_heads))
            .collect();

        Bit2Transformer {
            hidden_dim,
            vocab_size,
            embedding: Embedding::new(vocab_size, hidden_dim),
            layers,
            lm_head: Bit2Linear::new(hidden_dim, vocab_size),
            norm_weight: vec![1.0f32; hidden_dim],
        }
    }

    pub fn forward(
        &self,
        token_id: usize,
        pos: usize,
        caches: &mut [KVCache],
    ) -> Result<Vec<f32>> {
        let mut x = vec![0.0f32; self.hidden_dim];
        self.embedding.forward(token_id, &mut x)?;

        let mut norm_buf = vec![0.0f32; self.hidden_dim];

        for (i, layer) in self.layers.iter().enumerate() {
            // Attention Layer
            rms_norm(&x, &self.norm_weight, &mut norm_buf, 1e-5)?;
            let mut attn_out = vec![0.0f32; self.hidden_dim];
            layer
                .attention
                .forward(&norm_buf, &mut attn_out, &mut caches[i], pos)?;

            for j in 0..self.hidden_dim {
                x[j] += attn_out[j];
            }

            // FFN Layer (SwiGLU)
            rms_norm(&x, &self.norm_weight, &mut norm_buf, 1e-5)?;
            let mut gate = vec![0.0f32; self.hidden_dim * 2];
            let mut up = vec![0.0f32; self.hidden_dim * 2];
            layer.ffn_gate.forward(&norm_buf, &mut gate)?;
            layer.ffn_up.forward(&norm_buf, &mut up)?;

            let mut ffn_hidden = vec![0.0f32; self.hidden_dim * 2];
            for j in 0..gate.len() {
                ffn_hidden[j] = silu(gate[j]) * up[j];
            }

            let mut ffn_out = vec![0.0f32; self.hidden_dim];
            layer.ffn_down.forward(&ffn_hidden, &mut ffn_out)?;

            for j in 0..self.hidden_dim {
                x[j] += ffn_out[j];
            }
        }

        rms_norm(&x, &self.norm_weight, &mut norm_buf, 1e-5)?;
        let mut logits = vec![0.0f32; self.vocab_size];
        self.lm_head.forward(&norm_buf, &mut logits)?;
        softmax(&mut logits)?;

        Ok(logits)
    }
}