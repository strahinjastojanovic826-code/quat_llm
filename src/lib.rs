pub mod bit2;
pub mod array;
pub mod ops;
pub mod model;
pub mod tokenizer;
pub mod embedding;
pub mod attention;
pub mod transformer;
pub mod c_api;

#[cfg(feature = "python")]
pub mod py_api;

// Re-exports
pub use bit2::Bit2Val;
pub use array::Bit2Array;
pub use model::Bit2Linear;
pub use tokenizer::Tokenizer;
pub use transformer::Bit2Transformer;

#[cfg(test)]
mod tests {
    use super::array::Bit2Array;
    use super::attention::Bit2Attention;
    use super::bit2::Bit2Val;
    use super::model::Bit2Linear;
    use super::ops;
    use super::tokenizer::Tokenizer;
    use super::transformer::Bit2Transformer;
    use std::time::Instant;

    // ==========================================
    // 1. UNIT TESTS
    // ==========================================

    #[test]
    fn test_bit2_encoding_decoding() {
        assert_eq!(Bit2Val::from_u8(0b00), Bit2Val::Val00);
        assert_eq!(Bit2Val::from_u8(0b01), Bit2Val::Val01);
        assert_eq!(Bit2Val::from_u8(0b10), Bit2Val::Val10);
        assert_eq!(Bit2Val::from_u8(0b11), Bit2Val::Val11);

        assert_eq!(Bit2Val::Val00.to_f32(), -1.0);
        assert_eq!(Bit2Val::Val11.to_f32(), 1.0);
    }

    #[test]
    fn test_bit2_array_packing() {
        let mut arr = Bit2Array::new(10);

        arr.set(0, Bit2Val::Val00);
        arr.set(1, Bit2Val::Val01);
        arr.set(2, Bit2Val::Val10);
        arr.set(3, Bit2Val::Val11);
        arr.set(4, Bit2Val::Val01);

        assert_eq!(arr.get(0), Some(Bit2Val::Val00));
        assert_eq!(arr.get(1), Some(Bit2Val::Val01));
        assert_eq!(arr.get(2), Some(Bit2Val::Val10));
        assert_eq!(arr.get(3), Some(Bit2Val::Val11));
        assert_eq!(arr.get(4), Some(Bit2Val::Val01));
        assert_eq!(arr.get(10), None); // Out of bounds
    }

    #[test]
    fn test_dot_product() {
        let mut arr = Bit2Array::new(4);
        arr.set(0, Bit2Val::Val00); // -1.0
        arr.set(1, Bit2Val::Val11); //  1.0
        arr.set(2, Bit2Val::Val00); // -1.0
        arr.set(3, Bit2Val::Val11); //  1.0

        let input = vec![1.0, 2.0, 3.0, 4.0];
        // (-1.0 * 1.0) + (1.0 * 2.0) + (-1.0 * 3.0) + (1.0 * 4.0) = -1 + 2 - 3 + 4 = 2.0
        let result = arr.dot_f32(&input);
        assert!((result - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_probabilities() {
        let mut logits = vec![1.0, 2.0, 3.0];
        ops::softmax(&mut logits);

        // Sum of probabilities must equal 1.0
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Larger logit must have higher probability
        assert!(logits[2] > logits[1]);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_tokenizer() {
        let mut tok = Tokenizer::new();
        tok.add_token("hello", 1);
        tok.add_token("world", 2);

        let encoded = tok.encode("hello world unknown");
        assert_eq!(encoded, vec![1, 2, 0]);

        assert_eq!(tok.decode(1), "hello");
        assert_eq!(tok.decode(99), "<UNK>");
    }

    #[test]
    fn test_attention_forward() {
        let hidden_dim = 16;
        let attn = Bit2Attention::new(hidden_dim);
        let input = vec![0.5f32; hidden_dim];
        let mut output = vec![0.0f32; hidden_dim];

        attn.forward(&input, &mut output);
        assert_eq!(output.len(), hidden_dim);
    }

    #[test]
    fn test_transformer_forward() {
        let vocab_size = 10;
        let hidden_dim = 16;
        let model = Bit2Transformer::new(vocab_size, hidden_dim);

        let probs = model.forward(1);

        assert_eq!(probs.len(), vocab_size);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    // ==========================================
    // 2. STRESS & PERFORMANCE TESTS
    // ==========================================

    #[test]
    fn stress_test_large_bit2_array() {
        // 10 Million 2-bit elements
        let len = 10_000_000;
        let mut arr = Bit2Array::new(len);

        // Fill elements
        for i in 0..100_000 {
            arr.set(i, Bit2Val::from_u8((i % 4) as u8));
        }

        // Verify memory footprint (10M * 2 bits = 20M bits = 2.5 MB)
        let expected_bytes = (len + 3) / 4;
        assert_eq!(arr.data.len(), expected_bytes);
        println!(
            "\n[STRESS] 10M 2-bit elements memory size: {:.2} MB",
            arr.data.len() as f32 / (1024.0 * 1024.0)
        );
    }

    #[test]
    fn stress_test_dot_product_throughput() {
        let len = 1_000_000; // 1 Million elements
        let mut arr = Bit2Array::new(len);
        let input = vec![0.5f32; len];

        for i in 0..len {
            arr.set(i, Bit2Val::Val11);
        }

        let start = Instant::now();
        let iterations = 100;

        for _ in 0..iterations {
            let _ = arr.dot_f32(&input);
        }

        let duration = start.elapsed();
        let avg_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;
        println!(
            "\n[PERF] 1M element dot-product average time: {:.3} ms across {} runs",
            avg_ms, iterations
        );
    }

    #[test]
    fn stress_test_large_linear_layer() {
        // Large Layer: 4096 in_features -> 4096 out_features (16.7M weights)
        let in_features = 4096;
        let out_features = 4096;

        let start = Instant::now();
        let layer = Bit2Linear::new(in_features, out_features);
        let init_time = start.elapsed();

        let input = vec![1.0f32; in_features];
        let mut output = vec![0.0f32; out_features];

        let forward_start = Instant::now();
        layer.forward(&input, &mut output);
        let forward_time = forward_start.elapsed();

        println!(
            "\n[STRESS] Large Layer (4096x4096 = 16.7M weights):"
        );
        println!("  - Creation time: {:?}", init_time);
        println!("  - Forward pass time: {:?}", forward_time);
        println!(
            "  - Memory footprint of weights: {:.2} MB (vs {:.2} MB in FP32)",
            (in_features * out_features / 4) as f32 / (1024.0 * 1024.0),
            (in_features * out_features * 4) as f32 / (1024.0 * 1024.0)
        );
    }

    #[test]
    fn stress_test_transformer_inference_loop() {
        let vocab_size = 1000;
        let hidden_dim = 128;
        let model = Bit2Transformer::new(vocab_size, hidden_dim);

        let start = Instant::now();
        let tokens_to_generate = 50;
        let mut current_token = 1;

        for _ in 0..tokens_to_generate {
            let probs = model.forward(current_token);
            // Argmax selection
            current_token = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
        }

        let total_time = start.elapsed();
        let tokens_per_sec = tokens_to_generate as f64 / total_time.as_secs_f64();

        println!(
            "\n[STRESS] Transformer generated {} tokens in {:?}",
            tokens_to_generate, total_time
        );
        println!("[PERF] Speed: {:.2} tokens/sec", tokens_per_sec);
    }
}