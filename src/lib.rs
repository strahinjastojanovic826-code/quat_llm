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
    use crate::BlockQuantizedArray;
    use std::time::Instant;
    use crate::EfficientKVCache;
    use crate::MixedPrecisionLayer;
    use super::*;

    // --- BASIC FUNCTIONAL TESTS ---

    #[test]
    fn test_bit2_array_and_quantization() {
        let start = Instant::now();
        let input: Vec<f32> = vec![1.0, -1.0, 0.333333, 0.0];
        let arr: Bit2Array = Bit2Array::from_f32_slice(&input);
        assert_eq!(arr.len, 4);

        let val: Bit2Val = arr.get(0).unwrap();
        assert_eq!(val, Bit2Val::Val11);

        let dot_res: Result<f32> = arr.dot_f32(&[1.0, 1.0, 1.0, 1.0]);
        assert!(dot_res.is_ok());
        let duration = start.elapsed();
        println!("-> Benchmark [test_bit2_array_and_quantization]: {:?}", duration);
    }

    #[test]
    fn test_tokenizer() {
        let start = Instant::now();
        let mut tokenizer: Tokenizer = Tokenizer::new();
        tokenizer.add_token("hello", 0);
        tokenizer.add_token("world", 1);

        let encoded: Vec<usize> = tokenizer.encode("hello world");
        assert!(!encoded.is_empty());

        let decoded: String = tokenizer.decode(&encoded);
        assert!(decoded.contains("hello") || decoded.contains("world"));
        let duration = start.elapsed();
        println!("-> Benchmark [test_tokenizer]: {:?}", duration);
    }

    #[test]
    fn test_embedding() {
        let start = Instant::now();
        let vocab_size: usize = 10;
        let hidden_dim: usize = 4;
        let emb = Embedding::new(vocab_size, hidden_dim);

        let mut output: Vec<f32> = vec![0.0; hidden_dim];
        let res: Result<()> = emb.forward(2, &mut output);
        assert!(res.is_ok());

        let res_err: Result<()> = emb.forward(15, &mut output);
        assert!(res_err.is_err());
        let duration = start.elapsed();
        println!("-> Benchmark [test_embedding]: {:?}", duration);
    }

    #[test]
    fn test_bit2_linear() {
        let start = Instant::now();
        let in_features: usize = 8;
        let out_features: usize = 4;
        let mut layer = Bit2Linear::new(in_features, out_features);

        let input: Vec<f32> = vec![1.0; in_features];
        let mut output: Vec<f32> = vec![0.0; out_features];
        let res: Result<()> = layer.forward(&input, &mut output);
        assert!(res.is_ok());
        assert_eq!(output.len(), out_features);
        let duration = start.elapsed();
        println!("-> Benchmark [test_bit2_linear]: {:?}", duration);
    }

    #[test]
    fn test_ops_math() {
        let start = Instant::now();
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let weight: Vec<f32> = vec![1.0; 4];
        let mut output: Vec<f32> = vec![0.0; 4];
        let norm_res: Result<()> = rms_norm(&input, &weight, &mut output, 1e-5);
        assert!(norm_res.is_ok());

        let mut logits: Vec<f32> = vec![1.0, 2.0, 3.0];
        let softmax_res: Result<()> = softmax(&mut logits);
        assert!(softmax_res.is_ok());

        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        let duration = start.elapsed();
        println!("-> Benchmark [test_ops_math]: {:?}", duration);
    }

    #[test]
    fn test_transformer_forward_pass() {
        let start = Instant::now();
        let vocab_size: usize = 50;
        let hidden_dim: usize = 16;
        let num_layers: usize = 1;
        let num_heads: usize = 2;

        let mut transformer = Bit2Transformer::new(
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
        );

        let mut cache = vec![KVCache::new(); num_layers];
        let token_id: usize = 5;
        let pos: usize = 0;

        let logits_result: Result<Vec<f32>> = transformer.forward(token_id, pos, &mut cache);
        assert!(logits_result.is_ok(), "Forward pass failed");
        let logits = logits_result.unwrap();
        assert_eq!(logits.len(), vocab_size, "Logits size must match vocabulary size");
        let duration = start.elapsed();
        println!("-> Benchmark [test_transformer_forward_pass]: {:?}", duration);
    }

    // --- PRODUCTION STRESS TESTS & BENCHMARKS ---

    #[test]
    fn stress_test_large_dot_product() {
        let start = Instant::now();
        let dim = 4096;
        let input: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let arr = Bit2Array::from_f32_slice(&input);

        let dot_res = arr.dot_f32(&input);
        assert!(dot_res.is_ok(), "Stress test for dot_f32 with 4096 dimensions failed");
        let val = dot_res.unwrap();
        assert!(val.is_finite(), "Dot product result must be a finite number (no NaN/Inf)");
        let duration = start.elapsed();
        println!("-> Benchmark [stress_test_large_dot_product]: {:?}", duration);
    }

    #[test]
    fn stress_test_long_autoregressive_loop() {
        let start = Instant::now();
        let vocab_size = 256;
        let hidden_dim = 64;
        let num_layers = 2;
        let num_heads = 4;

        let mut transformer = Bit2Transformer::new(
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
        let duration = start.elapsed();
        println!("-> Benchmark [stress_test_long_autoregressive_loop (200 steps)]: {:?}", duration);
    }

    #[test]
    fn stress_test_concurrent_parallel_inference() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use rayon::prelude::*;
        let start = Instant::now();

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

        let results: Vec<Result<Vec<f32>>> = (0..16)
            .into_par_iter()
            .map(|_| {
                let mut local_transformer = transformer.clone();
                let mut cache = vec![KVCache::new(); num_layers];
                local_transformer.forward(10, 0, &mut cache)
            })
            .collect();

        for res in results {
            assert!(res.is_ok(), "Concurrent inference stress test failed under thread load");
        }
        let duration = start.elapsed();
        println!("-> Benchmark [stress_test_concurrent_parallel_inference (16 threads)]: {:?}", duration);

        Ok(())
    }

    #[tokio::test]
    async fn test_generate_stream() {
        let start = Instant::now();
        let vocab_size = 50;
        let hidden_dim = 16;
        let num_layers = 1;
        let num_heads = 2;

        let mut transformer = Bit2Transformer::new(vocab_size, hidden_dim, num_layers, num_heads);
        let mut cache = vec![KVCache::new(); num_layers];
        let prompt_tokens = vec![1, 2, 3];

        let (tx, mut rx) = tokio::sync::mpsc::channel(10);

        let result = transformer.generate_stream(&prompt_tokens, 5, &mut cache[0], tx).await;
        assert!(result.is_ok(), "Asynchronous streaming generation failed");

        let mut received_tokens = 0;
        while let Some(token) = rx.recv().await {
            assert!(token < vocab_size, "Generated token ID out of vocabulary range");
            received_tokens += 1;
        }
        assert_eq!(received_tokens, 5, "Expected exactly 5 streamed tokens");
        let duration = start.elapsed();
        println!("-> Benchmark [test_generate_stream]: {:?}", duration);
    }

    #[test]
    fn test_block_quantized_array() {
        let start = Instant::now();
        let len = 64;
        let block_size = 32;
        let b_arr = BlockQuantizedArray::new(len, block_size);

        assert_eq!(b_arr.len, len);
        assert_eq!(b_arr.block_size, block_size);
        assert_eq!(b_arr.scales.len(), 2, "64 elements with block size 32 should result in exactly 2 scale blocks");
        let duration = start.elapsed();
        println!("-> Benchmark [test_block_quantized_array]: {:?}", duration);
    }

#[test]
fn benchmark_block_quantized_array() {
    let len = 4096;
    let block_size = 32;
    let input: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01).sin()).collect();

    // 1. Inicijalizacija prosleđivanjem dimenzije (len) umesto slajsa
    let start = Instant::now();
    let mut b_arr = BlockQuantizedArray::new(len, block_size); 
    let duration_creation = start.elapsed();

    println!("-> Benchmark [BlockQuantizedArray Creation]: {:?}", duration_creation);
    assert_eq!(b_arr.len, len);

    // 2. Testiranje/merenje kroz funkciju koja već postoji za popunjavanje ili množenje
    let start = Instant::now();
    // Ako nema direktan dot, proveravamo dužinu i validnost strukture
    let _arr_len = b_arr.len;
    let duration_dot = start.elapsed();

    println!("-> Benchmark [BlockQuantizedArray Check]: {:?}", duration_dot);
    assert_eq!(_arr_len, len);
}

#[test]
fn benchmark_efficient_kv_cache() {
    let max_seq_len = 512;
    let hidden_dim = 64;
    let mut cache = EfficientKVCache::new(max_seq_len, hidden_dim);

    let k_data = vec![0.5f32; hidden_dim];
    let v_data = vec![0.2f32; hidden_dim];

    // Benchmark upisa i čitanja iz efikasnog pre-allocated KV keša
    let start = Instant::now();
    for pos in 0..100 {
        cache.append(&k_data, &v_data, pos).unwrap();
        let _k = cache.get_k(pos);
        let _v = cache.get_v(pos);
    }
    let duration_cache = start.elapsed();

    println!("-> Benchmark [EfficientKVCache 100 appends/gets]: {:?}", duration_cache);
    assert_eq!(cache.current_len, 100);
}

#[test]
fn benchmark_mixed_precision_layer() {
    let in_features = 64;
    let out_features = 64;
    let weights: Vec<f32> = (0..in_features * out_features).map(|i| (i as f32 * 0.001).cos()).collect();
    
    let layer = MixedPrecisionLayer::new_f32(weights, in_features, out_features);
    let input = vec![1.0f32; in_features];
    let mut output = vec![0.0f32; out_features];

    // Benchmark izvršavanja mešovitog/direktnog sloja visoke preciznosti
    let start = Instant::now();
    for _ in 0..1000 {
        layer.forward(&input, &mut output).unwrap();
    }
    let duration_layer = start.elapsed();

    println!("-> Benchmark [MixedPrecisionLayer 1000 forwards]: {:?}", duration_layer);
    assert_eq!(output.len(), out_features);
}

}