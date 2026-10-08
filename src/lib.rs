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
    use super::*;
    use std::time::{Duration, Instant};

    // ------------------------------------------------------------------------
    // TEST REPORTING UTILITY
    // ------------------------------------------------------------------------

    pub struct TestReport {
        pub test_name: &'static str,
        pub duration: Duration,
        pub status: &'static str,
        pub details: String,
    }

    impl TestReport {
        pub fn start(name: &'static str) -> TestRunner {
            println!("\n==================================================");
            println!("🚀 STARTING TEST: {}", name);
            println!("==================================================");
            TestRunner {
                name,
                start_time: Instant::now(),
            }
        }
    }

    pub struct TestRunner {
        name: &'static str,
        start_time: Instant,
    }

    impl TestRunner {
        pub fn finish(self, success: bool, details: String) {
            let duration = self.start_time.elapsed();
            let status = if success { "✅ PASSED" } else { "❌ FAILED" };

            println!("\n--------------------------------------------------");
            println!("📊 TEST REPORT: {}", self.name);
            println!("Status:     {}", status);
            println!("Duration:   {:?}", duration);
            println!("Details:    {}", details);
            println!("--------------------------------------------------\n");

            if !success {
                panic!("Test '{}' failed!", self.name);
            }
        }
    }

    // ------------------------------------------------------------------------
    // LEVEL 1: BASIC UNIT TESTS
    // ------------------------------------------------------------------------

    #[test]
    fn test_quantization_and_array_packing() {
        let runner = TestReport::start("L1: Quantization & Bit2Array Packing");

        let mut arr = array::Bit2Array::new(8);
        arr.set(0, bit2::Bit2Val::Val00).unwrap();
        arr.set(1, bit2::Bit2Val::Val01).unwrap();
        arr.set(2, bit2::Bit2Val::Val10).unwrap();
        arr.set(3, bit2::Bit2Val::Val11).unwrap();

        let mut success = true;
        success &= arr.get(0) == Some(bit2::Bit2Val::Val00);
        success &= arr.get(1) == Some(bit2::Bit2Val::Val01);
        success &= arr.get(2) == Some(bit2::Bit2Val::Val10);
        success &= arr.get(3) == Some(bit2::Bit2Val::Val11);

        let details = format!(
            "Byte 0 in memory: {:#010b} | Expected 4 x 2-bit states packed into 1 byte.",
            arr.data[0]
        );
        runner.finish(success, details);
    }

    #[test]
    fn test_tokenizer_roundtrip() {
        let runner = TestReport::start("L1: Tokenizer Encode/Decode Roundtrip");

        let mut tok = tokenizer::Tokenizer::new();
        tok.add_token("hello", 1);
        tok.add_token("world", 2);

        let encoded = tok.encode("hello world test");
        let decoded = tok.decode(&encoded);

        let success = encoded == vec![1, 2, 0]; // 0 is UNK
        let details = format!(
            "Input: 'hello world test' -> Tokens: {:?} -> Output: '{}'",
            encoded, decoded
        );
        runner.finish(success, details);
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let runner = TestReport::start("L1: Softmax Verification (Sum = 1.0)");

        let mut logits = vec![2.0, 1.0, 0.1, -1.0];
        ops::softmax(&mut logits).unwrap();

        let sum: f32 = logits.iter().sum();
        let success = (sum - 1.0).abs() < 1e-5;

        let details = format!("Calculated Softmax: {:?} | Sum = {}", logits, sum);
        runner.finish(success, details);
    }

    // ------------------------------------------------------------------------
    // LEVEL 2: MEDIUM INTEGRITY & ERROR HANDLING TESTS
    // ------------------------------------------------------------------------

    #[test]
    fn test_dimension_mismatch_handling() {
        let runner = TestReport::start("L2: Dimension Mismatch Error Handling");

        let layer = model::Bit2Linear::new(128, 64);
        let bad_input = vec![0.5f32; 100]; // Should be 128
        let mut output = vec![0.0f32; 64];

        let result = layer.forward(&bad_input, &mut output);

        let success = match result {
            Err(error::QuatError::DimensionMismatch { expected, got }) => {
                expected == 128 && got == 100
            }
            _ => false,
        };

        let details = format!("Mismatch call result: {:?}", result);
        runner.finish(success, details);
    }

    #[test]
    fn test_embedding_out_of_bounds() {
        let runner = TestReport::start("L2: Embedding Out-of-Bounds Protection");

        let emb = embedding::Embedding::new(100, 64); // Vocab size = 100
        let mut output = vec![0.0f32; 64];

        let result = emb.forward(150, &mut output); // Token ID 150 does not exist

        let success = result.is_err();
        let details = format!("Requested Token ID 150 in Vocab size 100. Error: {:?}", result);
        runner.finish(success, details);
    }

    // ------------------------------------------------------------------------
    // LEVEL 3: HEAVY AUTOREGRESSIVE & KV-CACHE TESTS
    // ------------------------------------------------------------------------

    #[test]
    fn test_autoregressive_generation_loop() {
        let runner = TestReport::start("L3: Autoregressive Generation Loop (10 steps with KV-Cache)");

        let vocab_size = 500;
        let hidden_dim = 128;
        let num_layers = 4;
        let num_heads = 4;

        let model = transformer::Bit2Transformer::new(vocab_size, hidden_dim, num_layers, num_heads);
        let mut caches: Vec<attention::KVCache> = (0..num_layers)
            .map(|_| attention::KVCache::new())
            .collect();

        let mut current_token = 12usize;
        let mut generated_tokens = vec![current_token];

        for pos in 0..10 {
            let logits = model.forward(current_token, pos, &mut caches).unwrap();

            // Argmax selection for the next token
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            generated_tokens.push(next_token);
            current_token = next_token;
        }

        let success = generated_tokens.len() == 11 && caches[0].k.len() == 10;
        let details = format!(
            "Generated Token Sequence: {:?} | KV-Cache size per layer: {} steps",
            generated_tokens,
            caches[0].k.len()
        );
        runner.finish(success, details);
    }

    // ------------------------------------------------------------------------
    // LEVEL 4: EXTREME STRESS & PARALLEL PERFORMANCE TESTS
    // ------------------------------------------------------------------------

    #[test]
    fn test_extreme_model_stress_and_concurrency() {
        let runner = TestReport::start("💥 L4: EXTREME STRESS - Large Model & Long Sequence");

        let vocab_size = 32000;
        let hidden_dim = 1024;
        let num_layers = 8;
        let num_heads = 8;

        println!("--> Initializing large 2-bit model (Hidden: {}, Layers: {})...", hidden_dim, num_layers);
        let init_start = Instant::now();
        let model = transformer::Bit2Transformer::new(vocab_size, hidden_dim, num_layers, num_heads);
        println!("--> Model initialized in {:?}", init_start.elapsed());

        let mut caches: Vec<attention::KVCache> = (0..num_layers)
            .map(|_| attention::KVCache::new())
            .collect();

        let num_steps = 100;
        let mut total_forward_time = Duration::ZERO;

        println!("--> Running {} inference steps with Rayon matrix multiplication...", num_steps);
        for pos in 0..num_steps {
            let step_start = Instant::now();
            let logits = model.forward(1, pos, &mut caches);

            assert!(logits.is_ok(), "Forward pass failed at position {}", pos);
            total_forward_time += step_start.elapsed();
        }

        let avg_step_time = total_forward_time / num_steps as u32;
        let tokens_per_second = 1.0 / avg_step_time.as_secs_f32();

        let details = format!(
            "Total time for {} steps: {:?} | Avg time per token: {:?} | Throughput: {:.2} tok/s | Allocated KV-Cache Memory: {:.2} MB",
            num_steps,
            total_forward_time,
            avg_step_time,
            tokens_per_second,
            (num_layers * num_steps * hidden_dim * 4) as f32 / (1024.0 * 1024.0)
        );

        runner.finish(true, details);
    }
}