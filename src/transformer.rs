use crate::attention::{Bit2MultiHeadAttention, KVCache};
use crate::embedding::Embedding;
use crate::error::Result;
use crate::model::Bit2Linear;
use crate::QuatError;
use crate::ops::{rms_norm, silu, softmax};
use tokio::sync::mpsc;
use crate::model::MixedPrecisionLayer;
use crate::attention::KvCache;

#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub attention: Bit2MultiHeadAttention,
    pub ffn_gate: Bit2Linear,
    pub ffn_up: Bit2Linear,
    pub ffn_down: Bit2Linear,
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub max_seq_len: Option<usize>,
    pub min_new_tokens: usize,
    pub eos_token_id: Option<usize>,
    pub stop_tokens: Vec<usize>,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

struct FastRng {
    state: u64,
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

#[derive(Debug, Clone)]
pub struct Bit2Transformer {
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub embedding: Embedding,
    pub layers: Vec<TransformerLayer>,
    pub lm_head: Bit2Linear,
    pub norm_weight: Vec<f32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            max_seq_len: None,
            min_new_tokens: 0,
            eos_token_id: None,
            stop_tokens: Vec::new(),
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
        }
    }

}

impl FastRng {
    fn from_seed(seed: u64) -> Self {
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: s }
    }

    fn from_entropy_like() -> Self {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(0x517CC1B727220A95);
        Self::from_seed(hasher.finish())
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let v = (self.next_u64() >> 40) as u32;
        (v as f32) / ((1u32 << 24) as f32)
    }
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

    pub async fn generate_stream(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cache: &mut KVCache,
        tx: mpsc::Sender<usize>, // Channel to stream generated token IDs
    ) -> Result<()> {
        let mut tokens = prompt_tokens.to_vec();
        let mut current_pos = 0;

        // Warm up KV cache
        for &token_id in prompt_tokens.iter().take(prompt_tokens.len().saturating_sub(1)) {
            let _ = self.forward(token_id, current_pos, std::slice::from_mut(cache))?;
            current_pos += 1;
        }

        let mut last_token = *tokens.last().ok_or(QuatError::InvalidPointer)?;

        // Autoregressive generation loop with streaming
        for _ in 0..max_new_tokens {
            let logits = self.forward(last_token, current_pos, std::slice::from_mut(cache))?;
            
            // Simple greedy sample (or top-k/top-p)
            let next_token = logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or(QuatError::InvalidPointer)?;

            // Send token via async channel to the client/UI
            if tx.send(next_token).await.is_err() {
                break; // Receiver dropped
            }

            tokens.push(next_token);
            last_token = next_token;
            current_pos += 1;
        }

        Ok(())
    }

    pub fn generate(
        &mut self,
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
        &mut self,
        token_id: usize,
        pos: usize,
        caches: &mut [KVCache],
    ) -> Result<Vec<f32>> {
        let mut x = vec![0.0f32; self.hidden_dim];
        self.embedding.forward(token_id, &mut x)?;

        let mut norm_buf = vec![0.0f32; self.hidden_dim];

        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Attention Layer
            rms_norm(&x, &self.norm_weight, &mut norm_buf, 1e-5)?;
            let mut attn_out = vec![0.0f32; self.hidden_dim];
            let _ = layer
                .attention
                .forward(&norm_buf, &mut attn_out, &mut caches[i], pos);

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

    /// Prefills multiple prompt tokens at once using per-layer norms, causal GQA/MQA,
    /// and a dedicated final RMSNorm before `lm_head`.
    pub fn prefill_and_logits<C: KvCache>(
        &mut self,
        prompt_tokens: &[usize],
        caches: &mut [C],
        start_pos: usize,
        num_kv_heads: usize,
        attn_norms: &[Vec<f32>],
        ffn_norms: &[Vec<f32>],
        final_norm: &[f32],
    ) -> Result<Vec<f32>> {
        if caches.len() != self.layers.len() {
            return Err(QuatError::DimensionMismatch {
                expected: self.layers.len(),
                got: caches.len(),
            });
        }
        let seq_len = prompt_tokens.len();
        if seq_len == 0 {
            return Err(QuatError::InvalidPointer);
        }

        let mut hidden_states = vec![0.0_f32; seq_len * self.hidden_dim];
        for (s, &tok_id) in prompt_tokens.iter().enumerate() {
            let out_slice = &mut hidden_states[s * self.hidden_dim..(s + 1) * self.hidden_dim];
            self.embedding.forward(tok_id, out_slice)?;
        }

        let mut norm_buf = vec![0.0_f32; self.hidden_dim];
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let a_norm = attn_norms
                .get(layer_idx)
                .map(|v| v.as_slice())
                .unwrap_or(&self.norm_weight);
            let f_norm = ffn_norms
                .get(layer_idx)
                .map(|v| v.as_slice())
                .unwrap_or(&self.norm_weight);

            for s in 0..seq_len {
                let pos = start_pos + s;
                let x_slice = &mut hidden_states[s * self.hidden_dim..(s + 1) * self.hidden_dim];
                layer.forward_attn_with_norm(
                    x_slice,
                    &mut norm_buf,
                    a_norm,
                    &mut caches[layer_idx],
                    pos,
                    num_kv_heads,
                )?;
                layer.forward_ffn_with_norm(x_slice, &mut norm_buf, f_norm)?;
            }
        }

        // Apply dedicated final RMSNorm on the last token representation before lm_head
        let last_hidden = &hidden_states[(seq_len - 1) * self.hidden_dim..seq_len * self.hidden_dim];
        crate::ops::rms_norm(last_hidden, final_norm, &mut norm_buf, 1e-5)?;

        let mut logits = vec![0.0_f32; self.vocab_size];
        self.lm_head.forward(&norm_buf, &mut logits)?;
        Ok(logits)
    }

    fn sample_with_rng(
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        generated_tokens: &[usize],
        rng: &mut FastRng,
    ) -> Result<usize> {
        if logits.is_empty() {
            return Err(QuatError::InvalidPointer);
        }

        let mut logits_vec: Vec<f32> = logits.to_vec();

        // 1. Repetition penalty
        if repetition_penalty > 0.0 && (repetition_penalty - 1.0).abs() > f32::EPSILON {
            for &token in generated_tokens {
                if token < logits_vec.len() {
                    if logits_vec[token] < 0.0 {
                        logits_vec[token] *= repetition_penalty;
                    } else {
                        logits_vec[token] /= repetition_penalty;
                    }
                }
            }
        }

        // 2. Temperature or greedy fallback
        if temperature <= 0.0 {
            let max_idx = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or(QuatError::InvalidPointer)?;
            return Ok(max_idx);
        }

        for logit in logits_vec.iter_mut() {
            *logit /= temperature;
        }

        // 3. Numerically stable softmax
        let max_val = logits_vec
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0_f32;
        let mut probs: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let p = (v - max_val).exp();
                exp_sum += p;
                (i, p)
            })
            .collect();

        if exp_sum > 0.0 && exp_sum.is_finite() {
            for (_, p) in probs.iter_mut() {
                *p /= exp_sum;
            }
        } else {
            let uniform = 1.0_f32 / (probs.len() as f32);
            for (_, p) in probs.iter_mut() {
                *p = uniform;
            }
        }

        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Top-K filtering
        if top_k > 0 && top_k < probs.len() {
            probs.truncate(top_k);
        }

        // 5. Top-P (nucleus) filtering
        if top_p > 0.0 && top_p < 1.0 && !probs.is_empty() {
            let mut cumulative_prob = 0.0_f32;
            let mut keep_len = 0usize;
            for (_, p) in probs.iter() {
                cumulative_prob += *p;
                keep_len += 1;
                if cumulative_prob >= top_p {
                    break;
                }
            }
            probs.truncate(keep_len.max(1));
        }

        if probs.is_empty() {
            return Err(QuatError::InvalidPointer);
        }

        // 6. True stochastic sampling using RNG over filtered probabilities
        let sum_probs: f32 = probs.iter().map(|(_, p)| *p).sum();
        if sum_probs <= 0.0 || !sum_probs.is_finite() {
            return Ok(probs[0].0);
        }

        let mut target = rng.next_f32() * sum_probs;
        for &(idx, p) in &probs {
            if target <= p {
                return Ok(idx);
            }
            target -= p;
        }

        Ok(probs.last().map(|(idx, _)| *idx).unwrap_or(probs[0].0))
    }

    pub fn sample_advanced(
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        generated_tokens: &[usize],
    ) -> Result<usize> {
        let mut rng = FastRng::from_entropy_like();
        Self::sample_with_rng(
            logits,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            generated_tokens,
            &mut rng,
        )
    }

    pub fn sample(
        &self,
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<usize> {
        Self::sample_advanced(logits, temperature, top_k, top_p, 1.0, &[])
    }

    fn should_stop(
        token_id: usize,
        generated_count: usize,
        config: &GenerationConfig,
    ) -> bool {
        if generated_count < config.min_new_tokens {
            return false;
        }
        if Some(token_id) == config.eos_token_id {
            return true;
        }
        config.stop_tokens.contains(&token_id)
    }

    pub fn generate_with_config(
        &mut self,
        prompt_tokens: &[usize],
        cache: &mut KVCache,
        config: &GenerationConfig,
    ) -> Result<Vec<usize>> {
        let mut tokens: Vec<usize> = prompt_tokens.to_vec();
        let mut current_pos: usize = 0;
        let mut rng = FastRng::from_entropy_like();

        for &token_id in prompt_tokens.iter().take(prompt_tokens.len().saturating_sub(1)) {
            if let Some(max_len) = config.max_seq_len {
                if current_pos >= max_len {
                    break;
                }
            }
            let _ = self.forward(token_id, current_pos, std::slice::from_mut(cache))?;
            current_pos += 1;
        }

        let mut last_token: usize = *tokens.last().ok_or(QuatError::InvalidPointer)?;
        let mut generated_count = 0usize;

        for _ in 0..config.max_new_tokens {
            if let Some(max_len) = config.max_seq_len {
                if current_pos >= max_len {
                    break;
                }
            }

            let logits: Vec<f32> =
                self.forward(last_token, current_pos, std::slice::from_mut(cache))?;
            let next_token = Self::sample_with_rng(
                &logits,
                config.temperature,
                config.top_k,
                config.top_p,
                config.repetition_penalty,
                &tokens,
                &mut rng,
            )?;

            tokens.push(next_token);
            generated_count += 1;
            last_token = next_token;
            current_pos += 1;

            if Self::should_stop(next_token, generated_count, config) {
                break;
            }
        }

        Ok(tokens)
    }

    pub async fn generate_stream_with_config(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cache: &mut KVCache,
        tx: mpsc::Sender<usize>,
        eos_token_id: Option<usize>,
        stop_tokens: &[usize],
        max_seq_len: Option<usize>,
    ) -> Result<()> {
        let config = GenerationConfig {
            max_new_tokens,
            max_seq_len,
            min_new_tokens: 0,
            eos_token_id,
            stop_tokens: stop_tokens.to_vec(),
            ..GenerationConfig::default()
        };

        let mut tokens: Vec<usize> = prompt_tokens.to_vec();
        let mut current_pos: usize = 0;
        let mut rng = FastRng::from_entropy_like();

        for &token_id in prompt_tokens.iter().take(prompt_tokens.len().saturating_sub(1)) {
            if let Some(max_len) = config.max_seq_len {
                if current_pos >= max_len {
                    break;
                }
            }
            let _ = self.forward(token_id, current_pos, std::slice::from_mut(cache))?;
            current_pos += 1;
        }

        let mut last_token: usize = *tokens.last().ok_or(QuatError::InvalidPointer)?;
        let mut generated_count = 0usize;

        for _ in 0..config.max_new_tokens {
            if let Some(max_len) = config.max_seq_len {
                if current_pos >= max_len {
                    break;
                }
            }

            let logits: Vec<f32> =
                self.forward(last_token, current_pos, std::slice::from_mut(cache))?;
            let next_token = Self::sample_with_rng(
                &logits,
                config.temperature,
                config.top_k,
                config.top_p,
                config.repetition_penalty,
                &tokens,
                &mut rng,
            )?;

            if tx.send(next_token).await.is_err() {
                break;
            }

            tokens.push(next_token);
            generated_count += 1;
            last_token = next_token;
            current_pos += 1;

            if Self::should_stop(next_token, generated_count, &config) {
                break;
            }
        }

        Ok(())
    }

}

impl TransformerLayer {
    pub fn run_ffn_block(
        &self,
        norm_buf: &[f32],
        ffn_out: &mut [f32],
        mixed_gate: Option<&MixedPrecisionLayer>,
        mixed_up: Option<&MixedPrecisionLayer>,
        mixed_down: Option<&MixedPrecisionLayer>,
    ) -> Result<()> {
        let hidden_dim = norm_buf.len();
        let ffn_dim = hidden_dim * 2;
        let mut gate = vec![0.0_f32; ffn_dim];
        let mut up = vec![0.0_f32; ffn_dim];

        match mixed_gate {
            Some(m) => m.forward_mixed(norm_buf, &mut gate)?,
            None => self.ffn_gate.forward(norm_buf, &mut gate)?,
        }

        match mixed_up {
            Some(m) => m.forward_mixed(norm_buf, &mut up)?,
            None => self.ffn_up.forward(norm_buf, &mut up)?,
        }

        let mut ffn_hidden = vec![0.0_f32; ffn_dim];
        for j in 0..gate.len() {
            ffn_hidden[j] = crate::ops::silu(gate[j]) * up[j];
        }

        match mixed_down {
            Some(m) => m.forward_mixed(&ffn_hidden, ffn_out)?,
            None => self.ffn_down.forward(&ffn_hidden, ffn_out)?,
        }

        Ok(())
    }

    /// Runs attention + residual with per-layer `attn_norm_weight` and unified `KvCache`.
    pub fn forward_attn_with_norm<C: KvCache + ?Sized>(
        &mut self,
        x: &mut [f32],
        norm_buf: &mut [f32],
        attn_norm_weight: &[f32],
        cache: &mut C,
        pos: usize,
        num_kv_heads: usize,
    ) -> Result<()> {
        crate::ops::rms_norm(x, attn_norm_weight, norm_buf, 1e-5)?;
        let mut attn_out = vec![0.0_f32; x.len()];
        self.attention
            .forward_gqa_causal(norm_buf, &mut attn_out, cache, pos, num_kv_heads)?;
        for j in 0..x.len() {
            x[j] += attn_out[j];
        }
        Ok(())
    }

    /// Runs SwiGLU FFN + residual with per-layer `ffn_norm_weight`.
    pub fn forward_ffn_with_norm(
        &self,
        x: &mut [f32],
        norm_buf: &mut [f32],
        ffn_norm_weight: &[f32],
    ) -> Result<()> {
        let hidden_dim = x.len();
        let ffn_dim = hidden_dim * 2;
        crate::ops::rms_norm(x, ffn_norm_weight, norm_buf, 1e-5)?;

        let mut gate = vec![0.0_f32; ffn_dim];
        let mut up = vec![0.0_f32; ffn_dim];
        self.ffn_gate.forward(norm_buf, &mut gate)?;
        self.ffn_up.forward(norm_buf, &mut up)?;

        let mut ffn_hidden = vec![0.0_f32; ffn_dim];
        for j in 0..ffn_dim {
            ffn_hidden[j] = crate::ops::silu(gate[j]) * up[j];
        }

        let mut ffn_out = vec![0.0_f32; hidden_dim];
        self.ffn_down.forward(&ffn_hidden, &mut ffn_out)?;
        for j in 0..hidden_dim {
            x[j] += ffn_out[j];
        }
        Ok(())
    }


}