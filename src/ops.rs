use crate::error::{QuatError, Result};

pub fn softmax(logits: &mut [f32]) -> Result<()> {
    if logits.is_empty() {
        return Ok(());
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for val in logits.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }
    if sum > 0.0 {
        for val in logits.iter_mut() {
            *val /= sum;
        }
    }
    Ok(())
}

pub fn rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) -> Result<()> {
    if input.len() != weight.len() || input.len() != output.len() {
        return Err(QuatError::DimensionMismatch {
            expected: input.len(),
            got: weight.len(),
        });
    }

    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = (sum_sq / input.len() as f32 + eps).sqrt();

    for i in 0..input.len() {
        output[i] = (input[i] / rms) * weight[i];
    }
    Ok(())
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Rotary Position Embedding (RoPE)
pub fn apply_rope(vec: &mut [f32], pos: usize, head_dim: usize) -> Result<()> {
    if vec.len() % head_dim != 0 {
        return Err(QuatError::DimensionMismatch {
            expected: head_dim,
            got: vec.len(),
        });
    }

    for chunk in vec.chunks_mut(head_dim) {
        for i in 0..(head_dim / 2) {
            let freq = 1.0 / 10000.0f32.powf((2 * i) as f32 / head_dim as f32);
            let val = pos as f32 * freq;
            let cos = val.cos();
            let sin = val.sin();

            let x0 = chunk[i];
            let x1 = chunk[i + head_dim / 2];

            chunk[i] = x0 * cos - x1 * sin;
            chunk[i + head_dim / 2] = x0 * sin + x1 * cos;
        }
    }
    Ok(())
}