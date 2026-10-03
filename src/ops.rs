/// Calculates Softmax probabilities over a slice of float logits.
/// Converts raw output numbers into probabilities that sum to 1.0 (100%).
pub fn softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }

    // Find max value for numerical stability (prevents overflow in exp)
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
}

/// Root Mean Square Normalization (RMSNorm) used in modern LLMs (e.g. Llama, BitNet)
pub fn rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    assert_eq!(input.len(), weight.len());
    assert_eq!(input.len(), output.len());

    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = (sum_sq / input.len() as f32 + eps).sqrt();

    for i in 0..input.len() {
        output[i] = (input[i] / rms) * weight[i];
    }
}

/// Sigmoid Linear Unit (SiLU / Swish) activation function
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}