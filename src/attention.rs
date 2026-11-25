use crate::error::{QuatError, Result};
use crate::model::Bit2Linear;

pub trait KvCache {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn reset(&mut self);
    fn append_kv(&mut self, k_data: &[f32], v_data: &[f32], pos: usize) -> Result<()>;
    fn get_k_at(&self, t: usize, hidden_or_kv_dim: usize) -> Result<Vec<f32>>;
    fn get_v_at(&self, t: usize, hidden_or_kv_dim: usize) -> Result<Vec<f32>>;
}

impl KvCache for KVCache {
    fn len(&self) -> usize {
        self.k.len()
    }

    fn reset(&mut self) {
        self.k.clear();
        self.v.clear();
    }

    fn append_kv(&mut self, k_data: &[f32], v_data: &[f32], pos: usize) -> Result<()> {
        if pos < self.k.len() {
            self.k[pos] = k_data.to_vec();
            self.v[pos] = v_data.to_vec();
        } else {
            self.k.push(k_data.to_vec());
            self.v.push(v_data.to_vec());
        }
        Ok(())
    }

    fn get_k_at(&self, t: usize, _dim: usize) -> Result<Vec<f32>> {
        self.k
            .get(t)
            .cloned()
            .ok_or(QuatError::DimensionMismatch {
                expected: self.k.len(),
                got: t,
            })
    }

    fn get_v_at(&self, t: usize, _dim: usize) -> Result<Vec<f32>> {
        self.v
            .get(t)
            .cloned()
            .ok_or(QuatError::DimensionMismatch {
                expected: self.v.len(),
                got: t,
            })
    }
}

impl KvCache for StaticKVCache {
    fn len(&self) -> usize {
        self.current_len
    }

    fn reset(&mut self) {
        self.current_len = 0;
    }

    fn append_kv(&mut self, k_data: &[f32], v_data: &[f32], pos: usize) -> Result<()> {
        self.append(k_data, v_data, pos)
    }

    fn get_k_at(&self, t: usize, dim: usize) -> Result<Vec<f32>> {
        if t >= self.current_len {
            return Err(QuatError::DimensionMismatch {
                expected: self.current_len,
                got: t,
            });
        }
        let start = t * dim;
        let end = start + dim;
        if end > self.k_cache.len() {
            return Err(QuatError::DimensionMismatch {
                expected: self.k_cache.len(),
                got: end,
            });
        }
        Ok(self.k_cache[start..end].to_vec())
    }

    fn get_v_at(&self, t: usize, dim: usize) -> Result<Vec<f32>> {
        if t >= self.current_len {
            return Err(QuatError::DimensionMismatch {
                expected: self.current_len,
                got: t,
            });
        }
        let start = t * dim;
        let end = start + dim;
        if end > self.v_cache.len() {
            return Err(QuatError::DimensionMismatch {
                expected: self.v_cache.len(),
                got: end,
            });
        }
        Ok(self.v_cache[start..end].to_vec())
    }
}

impl KvCache for EfficientKVCache {
    fn len(&self) -> usize {
        self.current_len
    }

    fn reset(&mut self) {
        self.current_len = 0;
    }

    fn append_kv(&mut self, k_data: &[f32], v_data: &[f32], pos: usize) -> Result<()> {
        self.append(k_data, v_data, pos)
    }

    fn get_k_at(&self, t: usize, dim: usize) -> Result<Vec<f32>> {
        if t >= self.current_len {
            return Err(QuatError::DimensionMismatch {
                expected: self.current_len,
                got: t,
            });
        }
        let start = t * dim;
        let end = start + dim;
        if end > self.k_buffer.len() {
            return Err(QuatError::DimensionMismatch {
                expected: self.k_buffer.len(),
                got: end,
            });
        }
        Ok(self.k_buffer[start..end].to_vec())
    }

    fn get_v_at(&self, t: usize, dim: usize) -> Result<Vec<f32>> {
        if t >= self.current_len {
            return Err(QuatError::DimensionMismatch {
                expected: self.current_len,
                got: t,
            });
        }
        let start = t * dim;
        let end = start + dim;
        if end > self.v_buffer.len() {
            return Err(QuatError::DimensionMismatch {
                expected: self.v_buffer.len(),
                got: end,
            });
        }
        Ok(self.v_buffer[start..end].to_vec())
    }
}

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
        Self {
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
    &mut self,
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

    // Apply Rotary Position Embedding (RoPE)
    crate::ops::apply_rope(&mut q, pos, self.head_dim)?;
    crate::ops::apply_rope(&mut k, pos, self.head_dim)?;

    // Push to KV Cache
    cache.k.push(k.clone());
    cache.v.push(v.clone());

    let seq_len = cache.k.len();
    output.fill(0.0);

    // Tiled FlashAttention loop to improve memory locality and cache hits
    const TILE_SIZE: usize = 16;
    let mut attn_out = vec![0.0f32; self.hidden_dim];

    for h in 0..self.num_heads {
        let h_off = h * self.head_dim;
        let q_chunk = &q[h_off..h_off + self.head_dim];

        // Process sequence in blocks (tiles) for FlashAttention optimization
        for t_block in (0..seq_len).step_by(TILE_SIZE) {
            let block_end = std::cmp::min(t_block + TILE_SIZE, seq_len);
            let mut tile_scores = vec![0.0f32; block_end - t_block];

            for (local_idx, t) in (t_block..block_end).enumerate() {
                let k_chunk = &cache.k[t][h_off..h_off + self.head_dim];
                
                let dot_prod: f32 = q_chunk.iter()
                    .zip(k_chunk.iter())
                    .map(|(a, b)| a * b)
                    .sum();

                tile_scores[local_idx] = dot_prod / (self.head_dim as f32).sqrt();
            }

            // Apply local softmax stabilization over the tile
            crate::ops::softmax(&mut tile_scores)?;

            // Accumulate weighted values into attention output
            for (local_idx, t) in (t_block..block_end).enumerate() {
                let score = tile_scores[local_idx];
                let v_chunk = &cache.v[t][h_off..h_off + self.head_dim];
                
                for d in 0..self.head_dim {
                    attn_out[h_off + d] += score * v_chunk[d];
                }
            }
        }
    }

    self.out_proj.forward(&attn_out, output)?;
    Ok(())
  }

  /// Creates a quaternary attention module with Grouped-Query / Multi-Query support.
    /// Set `num_kv_heads == num_heads` for standard MHA, `1` for MQA, or a divisor for GQA.
    pub fn new_gqa(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<Self> {
        let kv_heads = num_kv_heads.max(1);
        if num_heads == 0 || hidden_dim % num_heads != 0 || num_heads % kv_heads != 0 {
            return Err(QuatError::DimensionMismatch {
                expected: num_heads,
                got: kv_heads,
            });
        }
        let head_dim = hidden_dim / num_heads;
        let kv_dim = head_dim * kv_heads;
        Ok(Self {
            hidden_dim,
            num_heads,
            head_dim,
            q_proj: Bit2Linear::new(hidden_dim, hidden_dim),
            k_proj: Bit2Linear::new(hidden_dim, kv_dim),
            v_proj: Bit2Linear::new(hidden_dim, kv_dim),
            out_proj: Bit2Linear::new(hidden_dim, hidden_dim),
        })
    }

    /// Single-token forward pass with GQA/MQA head mapping, causal masking up to `pos`,
    /// and unified `KvCache` trait support.
    pub fn forward_gqa_causal<C: KvCache + ?Sized>(
        &self,
        input: &[f32],
        output: &mut [f32],
        cache: &mut C,
        pos: usize,
        num_kv_heads: usize,
    ) -> Result<()> {
        if input.len() != self.hidden_dim || output.len() != self.hidden_dim {
            return Err(QuatError::DimensionMismatch {
                expected: self.hidden_dim,
                got: input.len(),
            });
        }
        let kv_heads = num_kv_heads.max(1);
        let kv_dim = self.head_dim * kv_heads;
        let group_size = (self.num_heads / kv_heads).max(1);

        let mut q = vec![0.0_f32; self.hidden_dim];
        let mut k = vec![0.0_f32; kv_dim];
        let mut v = vec![0.0_f32; kv_dim];

        self.q_proj.forward(input, &mut q)?;
        self.k_proj.forward(input, &mut k)?;
        self.v_proj.forward(input, &mut v)?;

        crate::ops::apply_rope(&mut q, pos, self.head_dim)?;
        crate::ops::apply_rope(&mut k, pos, self.head_dim)?;

        cache.append_kv(&k, &v, pos)?;

        let active_len = (cache.len()).min(pos + 1);
        output.fill(0.0);
        let mut attn_out = vec![0.0_f32; self.hidden_dim];
        let scale = 1.0_f32 / (self.head_dim as f32).sqrt();

        for h in 0..self.num_heads {
            let h_off = h * self.head_dim;
            let q_chunk = &q[h_off..h_off + self.head_dim];
            let kv_head_idx = h / group_size;
            let kv_off = kv_head_idx * self.head_dim;

            let mut scores = vec![f32::NEG_INFINITY; active_len];
            for (local_idx, t) in (0..active_len).enumerate() {
                if t > pos {
                    continue; // Causal mask: block future positions
                }
                let k_vec = cache.get_k_at(t, kv_dim)?;
                let k_chunk = &k_vec[kv_off..kv_off + self.head_dim];
                let dot_prod: f32 = q_chunk
                    .iter()
                    .zip(k_chunk.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                scores[local_idx] = dot_prod * scale;
            }

            crate::ops::softmax(&mut scores)?;

            for (local_idx, t) in (0..active_len).enumerate() {
                if t > pos {
                    continue;
                }
                let score = scores[local_idx];
                let v_vec = cache.get_v_at(t, kv_dim)?;
                let v_chunk = &v_vec[kv_off..kv_off + self.head_dim];
                for d in 0..self.head_dim {
                    attn_out[h_off + d] += score * v_chunk[d];
                }
            }
        }

        self.out_proj.forward(&attn_out, output)
    }

    /// Batch / prefill forward pass over multiple tokens (`seq_len * hidden_dim`),
    /// applying causal masking (`k_pos <= q_pos`) and populating the KV cache.
    pub fn forward_prefill<C: KvCache + ?Sized>(
        &self,
        tokens_hidden: &[f32],
        seq_len: usize,
        output_hidden: &mut [f32],
        cache: &mut C,
        start_pos: usize,
        num_kv_heads: usize,
    ) -> Result<()> {
        if tokens_hidden.len() != seq_len * self.hidden_dim
            || output_hidden.len() != seq_len * self.hidden_dim
        {
            return Err(QuatError::DimensionMismatch {
                expected: seq_len * self.hidden_dim,
                got: tokens_hidden.len(),
            });
        }

        let kv_heads = num_kv_heads.max(1);
        let kv_dim = self.head_dim * kv_heads;
        let group_size = (self.num_heads / kv_heads).max(1);
        let scale = 1.0_f32 / (self.head_dim as f32).sqrt();

        // 1. Project & RoPE all prompt tokens into KV cache
        let mut all_q = vec![0.0_f32; seq_len * self.hidden_dim];
        for s in 0..seq_len {
            let pos = start_pos + s;
            let in_slice = &tokens_hidden[s * self.hidden_dim..(s + 1) * self.hidden_dim];
            let q_slice = &mut all_q[s * self.hidden_dim..(s + 1) * self.hidden_dim];
            let mut k = vec![0.0_f32; kv_dim];
            let mut v = vec![0.0_f32; kv_dim];

            self.q_proj.forward(in_slice, q_slice)?;
            self.k_proj.forward(in_slice, &mut k)?;
            self.v_proj.forward(in_slice, &mut v)?;

            crate::ops::apply_rope(q_slice, pos, self.head_dim)?;
            crate::ops::apply_rope(&mut k, pos, self.head_dim)?;
            cache.append_kv(&k, &v, pos)?;
        }

        // 2. Causal attention per query token in the prefill chunk
        for s in 0..seq_len {
            let q_pos = start_pos + s;
            let q_base = s * self.hidden_dim;
            let mut attn_out = vec![0.0_f32; self.hidden_dim];
            let visible_len = (cache.len()).min(q_pos + 1);

            for h in 0..self.num_heads {
                let h_off = h * self.head_dim;
                let q_chunk = &all_q[q_base + h_off..q_base + h_off + self.head_dim];
                let kv_head_idx = h / group_size;
                let kv_off = kv_head_idx * self.head_dim;

                let mut scores = vec![f32::NEG_INFINITY; visible_len];
                for t in 0..visible_len {
                    let k_vec = cache.get_k_at(t, kv_dim)?;
                    let k_chunk = &k_vec[kv_off..kv_off + self.head_dim];
                    let dot_prod: f32 = q_chunk
                        .iter()
                        .zip(k_chunk.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    scores[t] = dot_prod * scale;
                }

                crate::ops::softmax(&mut scores)?;

                for t in 0..visible_len {
                    let score = scores[t];
                    let v_vec = cache.get_v_at(t, kv_dim)?;
                    let v_chunk = &v_vec[kv_off..kv_off + self.head_dim];
                    for d in 0..self.head_dim {
                        attn_out[h_off + d] += score * v_chunk[d];
                    }
                }
            }

            let out_slice = &mut output_hidden[s * self.hidden_dim..(s + 1) * self.hidden_dim];
            self.out_proj.forward(&attn_out, out_slice)?;
        }

        Ok(())
    }

}

#[derive(Debug, Clone)]
pub struct StaticKVCache {
    pub k_cache: Vec<f32>, // Flat contiguous buffer: [max_seq_len, num_heads, head_dim]
    pub v_cache: Vec<f32>,
    pub max_seq_len: usize,
    pub hidden_dim: usize,
    pub current_len: usize,
}

impl StaticKVCache {
    pub fn new(max_seq_len: usize, hidden_dim: usize) -> Self {
        Self {
            k_cache: vec![0.0; max_seq_len * hidden_dim],
            v_cache: vec![0.0; max_seq_len * hidden_dim],
            max_seq_len,
            hidden_dim,
            current_len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    pub fn append(&mut self, k_data: &[f32], v_data: &[f32], pos: usize) -> Result<()> {
        if pos >= self.max_seq_len {
            return Err(QuatError::DimensionMismatch {
                expected: self.max_seq_len,
                got: pos,
            });
        }
        let start_idx = pos * self.hidden_dim;
        let end_idx = start_idx + self.hidden_dim;
        
        self.k_cache[start_idx..end_idx].copy_from_slice(k_data);
        self.v_cache[start_idx..end_idx].copy_from_slice(v_data);
        self.current_len = self.current_len.max(pos + 1);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct EfficientKVCache {
    pub k_buffer: Vec<f32>, // Flat contiguous memory block
    pub v_buffer: Vec<f32>, // Flat contiguous memory block
    pub max_seq_len: usize,
    pub hidden_dim: usize,
    pub current_len: usize,
}

impl EfficientKVCache {
    pub fn new(max_seq_len: usize, hidden_dim: usize) -> Self {
        Self {
            k_buffer: vec![0.0; max_seq_len * hidden_dim],
            v_buffer: vec![0.0; max_seq_len * hidden_dim],
            max_seq_len,
            hidden_dim,
            current_len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    /// Appends new key-value data into pre-allocated memory without reallocations
    pub fn append(&mut self, k_data: &[f32], v_data: &[f32], pos: usize) -> Result<()> {
        if pos >= self.max_seq_len {
            return Err(QuatError::DimensionMismatch {
                expected: self.max_seq_len,
                got: pos,
            });
        }

        let start_idx = pos * self.hidden_dim;
        let end_idx = start_idx + self.hidden_dim;

        self.k_buffer[start_idx..end_idx].copy_from_slice(k_data);
        self.v_buffer[start_idx..end_idx].copy_from_slice(v_data);

        self.current_len = std::cmp::max(self.current_len, pos + 1);
        Ok(())
    }

    #[inline(always)]
    pub fn get_k(&self, pos: usize) -> &[f32] {
        let start = pos * self.hidden_dim;
        &self.k_buffer[start..start + self.hidden_dim]
    }

    #[inline(always)]
    pub fn get_v(&self, pos: usize) -> &[f32] {
        let start = pos * self.hidden_dim;
        &self.v_buffer[start..start + self.hidden_dim]
    }
}