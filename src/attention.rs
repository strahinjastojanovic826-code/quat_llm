use crate::error::{QuatError, Result};
use crate::model::Bit2Linear;

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