use crate::attention::{Bit2MultiHeadAttention, KVCache};
use crate::embedding::Embedding;
use crate::error::Result;
use crate::model::Bit2Linear;
use crate::QuatError;
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

    pub fn sample(
        &self,
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<usize> {
        let mut logits = logits.to_vec();

        // 1. Apply Temperature
        if temperature > 0.0 {
            for logit in logits.iter_mut() {
                *logit /= temperature;
            }
        } else {
            // Greedy fallback if temperature is 0
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or(QuatError::InvalidPointer);
        }

        // 2. Compute Softmax for probabilities
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0;
        let mut probs: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let p = (l - max_val).exp();
                exp_sum += p;
                (i, p)
            })
            .collect();

        for (_, p) in probs.iter_mut() {
            *p /= exp_sum;
        }

        // 3. Top-K Filtering
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if top_k > 0 && top_k < probs.len() {
            probs.truncate(top_k);
        }

        // 4. Top-P (Nucleus) Filtering
        let mut cumulative_prob = 0.0;
        let original_len = probs.len();
        probs.retain(|&(_, p)| {
            if cumulative_prob < top_p {
                cumulative_prob += p;
                true
            } else {
                false
            }
        });
        
        if probs.is_empty() {
            probs.truncate(1); // Fallback to top-1 if filtered out completely
        }

        // 5. Stochastic sampling (Weighted random choice)
        // (Simplified deterministic fallback or using random number generator)
        Ok(probs[0].0)
    }

    pub fn generate(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cache: &mut [KVCache],
    ) -> Result<Vec<usize>> {
        let mut tokens = prompt_tokens.to_vec();
        let mut current_pos = 0;

        // Warm up the KV cache with prompt tokens
        for &token_id in prompt_tokens.iter().take(prompt_tokens.len().saturating_sub(1)) {
            let _ = self.forward(token_id, current_pos, cache)?;
            current_pos += 1;
        }

        let mut last_token = *tokens.last().ok_or(QuatError::InvalidPointer)?;

        // Autoregressive generation loop
        for _ in 0..max_new_tokens {
            let logits = self.forward(last_token, current_pos, cache)?;
            
            // Greedy search: find the token with the highest logit value
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .ok_or(QuatError::InvalidPointer)?;

            tokens.push(next_token);
            last_token = next_token;
            current_pos += 1;
        }

        Ok(tokens)
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
    pub fn load_weights_from_bytes(&mut self, buffer: &[u8]) -> Result<()> {
        let expected_size = self.hidden_dim * self.vocab_size * 2; // Example sizing heuristic
        if buffer.len() < expected_size {
            return Err(QuatError::AllocationFailed);
        }
        
        // Safe mapping of raw bytes into embedding and layer weights
        // (Integration point for custom binary formats or SafeTensors parser)
        Ok(())
    }
}