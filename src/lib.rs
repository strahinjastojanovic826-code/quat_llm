pub mod array;
pub mod attention;
pub mod bit2;
pub mod c_api;
pub mod embedding;
pub mod error;
pub mod model;
pub mod ops;
pub mod py_api;
pub mod tokenizer;
pub mod transformer;
pub mod loader;

// Eksportujemo primarne tipove sa nivoa biblioteke
pub use array::*;
pub use attention::*;
pub use bit2::*;
pub use embedding::Embedding;
pub use error::{QuatError, Result};
pub use model::*;
pub use tokenizer::Tokenizer;
pub use transformer::*;
pub use attention::Bit2MultiHeadAttention;

#[cfg(test)]
mod tests {
    use crate::array::Bit2Array;
    use crate::bit2::Bit2Val;
    use crate::tokenizer::Tokenizer;
    use crate::embedding::Embedding;
    use crate::model::Bit2Linear;
    use crate::ops::{rms_norm, softmax};
    use crate::Bit2Transformer;
    use crate::attention::KVCache;
    use rayon::prelude::*;

    // --- BASIC FUNCTIONAL TESTS ---

    #[test]
    fn test_bit2_array_and_quantization() {
        let input = vec![1.0, -1.0, 0.333333, 0.0];
        let arr = Bit2Array::from_f32_slice(&input);
        assert_eq!(arr.len, 4);

        let val = arr.get(0).unwrap();
        assert_eq!(val, Bit2Val::Val11);

        let dot_res = arr.dot_f32(&[1.0, 1.0, 1.0, 1.0]);
        assert!(dot_res.is_ok());
    }

    #[test]
fn test_tokenizer() {
    let mut tokenizer = Tokenizer::new();
    tokenizer.add_token("hello", 0);
    tokenizer.add_token("world", 1);

    let encoded = tokenizer.encode("hello world");
    assert!(!encoded.is_empty());

    let decoded = tokenizer.decode(&encoded);
    assert!(decoded.contains("hello") || decoded.contains("world"));
}

    #[test]
    fn test_embedding() {
        let vocab_size = 10;
        let hidden_dim = 4;
        let emb = Embedding::new(vocab_size, hidden_dim);
        
        let mut output = vec![0.0; hidden_dim];
        let res = emb.forward(2, &mut output);
        assert!(res.is_ok());

        let res_err = emb.forward(15, &mut output);
        assert!(res_err.is_err());
    }

    #[test]
    fn test_bit2_linear() {
        let in_features = 8;
        let out_features = 4;
        let mut layer = Bit2Linear::new(in_features, out_features);
        
        let input = vec![1.0; in_features];
        let mut output = vec![0.0; out_features];
        
        let res = layer.forward(&input, &mut output);
        assert!(res.is_ok());
        assert_eq!(output.len(), out_features);
    }

    #[test]
    fn test_ops_math() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        let norm_res = rms_norm(&input, &weight, &mut output, 1e-5);
        assert!(norm_res.is_ok());

        let mut logits = vec![1.0, 2.0, 3.0];
        let softmax_res = softmax(&mut logits);
        assert!(softmax_res.is_ok());
        
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_transformer_forward_pass() {
        let vocab_size = 50;
        let hidden_dim = 16;
        let num_layers = 1;
        let num_heads = 2;

        let transformer = Bit2Transformer::new(
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
        );

        let mut cache = vec![KVCache::new(); num_layers];
        let token_id = 5;
        let pos = 0;

        let logits_result = transformer.forward(token_id, pos, &mut cache);
        
        assert!(logits_result.is_ok(), "Forward pass failed");
        let logits = logits_result.unwrap();
        assert_eq!(logits.len(), vocab_size, "Logits size must match vocabulary size");
    }

    // --- PRODUCTION STRESS TESTS ---

    #[test]
    fn stress_test_large_dot_product() {
        let dim = 4096;
        let input: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let arr = Bit2Array::from_f32_slice(&input);

        let dot_res = arr.dot_f32(&input);
        assert!(dot_res.is_ok(), "Stress test for dot_f32 with 4096 dimensions failed");
        let val = dot_res.unwrap();
        assert!(val.is_finite(), "Dot product result must be a finite number (no NaN/Inf)");
    }

    #[test]
    fn stress_test_long_autoregressive_loop() {
        let vocab_size = 256;
        let hidden_dim = 64;
        let num_layers = 2;
        let num_heads = 4;

        let transformer = Bit2Transformer::new(
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
        );

        let mut cache = vec![KVCache::new(); num_layers];
        let mut current_token = 42;

        for pos in 0..200 {
            let logits_result = transformer.forward(current_token, pos, &mut cache);
            assert!(logits_result.is_ok(), "Autoregressive generation failed at position: {}", pos);
            
            let logits = logits_result.unwrap();
            current_token = (logits[0].abs() as usize) % vocab_size;
        }
    }

    #[test]
    fn stress_test_concurrent_parallel_inference() {
        let vocab_size = 128;
        let hidden_dim = 32;
        let num_layers = 1;
        let num_heads = 2;

        let transformer = Bit2Transformer::new(
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
        );

        let results: Vec<Result<Vec<f32>, _>> = (0..16)
            .into_par_iter()
            .map(|_| {
                let mut cache = vec![KVCache::new(); num_layers];
                transformer.forward(10, 0, &mut cache)
            })
            .collect();

        for res in results {
            assert!(res.is_ok(), "Concurrent inference stress test failed under thread load");
        }
    }
}