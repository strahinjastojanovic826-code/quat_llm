use crate::attention::Bit2Attention;
use crate::embedding::Embedding;
use crate::model::Bit2Linear;
use crate::ops;

/// Full 2-Bit Transformer Model Architecture
pub struct Bit2Transformer {
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub embedding: Embedding,
    pub attention: Bit2Attention,
    pub ffn_gate: Bit2Linear,
    pub ffn_up: Bit2Linear,
    pub lm_head: Bit2Linear, // Projects hidden vector to vocabulary logits
}

impl Bit2Transformer {
    pub fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        Bit2Transformer {
            hidden_dim,
            vocab_size,
            embedding: Embedding::new(vocab_size, hidden_dim),
            attention: Bit2Attention::new(hidden_dim),
            ffn_gate: Bit2Linear::new(hidden_dim, hidden_dim * 2),
            ffn_up: Bit2Linear::new(hidden_dim * 2, hidden_dim),
            lm_head: Bit2Linear::new(hidden_dim, vocab_size),
        }
    }

    /// Predicts probabilities for the next token given a token_id
    pub fn forward(&self, token_id: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; self.hidden_dim];
        let mut attn_out = vec![0.0f32; self.hidden_dim];
        let mut ffn_hidden = vec![0.0f32; self.hidden_dim * 2];
        let mut ffn_out = vec![0.0f32; self.hidden_dim];
        let mut logits = vec![0.0f32; self.vocab_size];

        // 1. Embedding Lookup
        self.embedding.forward(token_id, &mut x);

        // 2. 2-Bit Attention Layer
        self.attention.forward(&x, &mut attn_out);

        // Residual connection
        for i in 0..self.hidden_dim {
            x[i] += attn_out[i];
        }

        // 3. 2-Bit Feed Forward Network (FFN) with SiLU activation
        self.ffn_gate.forward(&x, &mut ffn_hidden);
        for val in ffn_hidden.iter_mut() {
            *val = ops::silu(*val);
        }
        self.ffn_up.forward(&ffn_hidden, &mut ffn_out);

        // Residual connection
        for i in 0..self.hidden_dim {
            x[i] += ffn_out[i];
        }

        // 4. LM Head (Predict logits for next vocabulary word)
        self.lm_head.forward(&x, &mut logits);

        // 5. Convert Logits to Probabilities
        ops::softmax(&mut logits);

        logits
    }
}