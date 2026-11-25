# quat_llm 🚀

`quat_llm` is a high-performance, lightweight 2-bit quantized LLM inference engine built from scratch in **Rust**, with seamless bindings for **Python (PyO3)**, **C**, **C++**, and **C#**. It drastically reduces memory footprint and maximizes CPU throughput through custom bit-packing and low-level optimization.

---

## ⚡ Key Features

- **Extreme Memory Efficiency**: Packs 4 values into a single byte using custom 2-bit quantization (`Bit2Array`), drastically reducing RAM usage compared to FP32/FP16[cite: 6].
- **Blazing Fast Inference**: Cache-optimized parallel execution engine leveraging Rayon for multi-threaded matrix-vector dot products and chunk-based processing[cite: 6].
- **Cross-Language Support**: Seamlessly integrate the engine into Rust, Python, C, C++, and C#[cite: 6].
- **Modular Architecture**: Includes clean, robust implementations of Tokenizer, Embedding, RMSNorm, Softmax, RoPE, and Attention with KV-Caching[cite: 6].
- **Production-Grade Robustness**: Built-in comprehensive test suites covering functional behavior, long autoregressive generation loops, and high-concurrency Rayon stress tests[cite: 6].
- **Stable SIMD Optimization**: Hardware-accelerated AVX2 intrinsics running on stable Rust for maximum execution speed without requiring nightly features.
- **Asynchronous Streaming**: Non-blocking token generation and real-time streaming capabilities powered by Tokio channels.
- **Standardized Weight Loading**: Seamless integration with `safetensors` for secure and efficient model loading.
- **Group-Wise Block Quantization**: Advanced block-level scaling for 2-bit arrays to significantly mitigate precision loss during low-bit weight quantization.
- **Pre-Allocated & Efficient KV Caching**: Optimized, memory-safe sliding-window/pre-allocated caching mechanisms designed to accelerate long autoregressive inference loops.
- **Mixed-Precision Layer Support**: Seamless integration supporting both ultra-dense 2-bit quaternary weights and high-precision direct FP32 layers for critical network paths.
- **Cross-Platform SIMD Acceleration**: Native hardware-accelerated AVX2 and ARM NEON instruction sets for maximum mathematical throughput.
- **High-Concurrency Rayon Stress Testing**: Robust parallel inference verification ensuring thread-safe operations under intense concurrent workloads.
- **Advanced Sampling & Generation Config**: Configurable sampling pipeline featuring temperature scaling, top-k, top-p (nucleus) filtering, repetition penalty, and fast deterministic/pseudo-random selection via `FastRng`.
- **Activation Quantization (`QuantAct`)**: Optional block-wise activation quantization (`act_block_size` and `act_clip_ratio`) inside `MixedPrecisionLayer` for `Bit2` and `Block2` layers to further cut memory bandwidth overhead.
- **Complete Transformer Primitives & Prefill**: Full autoregressive generation loop with prompt prefill, causal masking, RMSNorm, and Grouped-Query / Multi-Query Attention (GQA/MQA).
- **Safe C ABI & Multi-Layer Cache Support**: Validated raw-pointer C API (`QuatStatus` error codes, `quat_transformer_create`, `quat_transformer_generate`, `quat_transformer_free`) managing per-layer KV caches cleanly across FFI boundaries.

---

## 📊 Performance & Testing (Stress Tests)

| Test / Benchmark Name | Category | Description / What It Verifies & Measures | Debug Time (`cargo test`) | Release Time (`cargo test --release`) |
| :--- | :--- | :--- | :--- | :--- |
| `benchmark_efficient_kv_cache` | KV Cache | Benchmarks 100 sequential append and get operations on pre-allocated `EfficientKVCache` and verifies final length. | `34.8µs` | *TBD* |
| `benchmark_block_quantized_array` | Block Quantization | Measures `BlockQuantizedArray` creation (`len=4096`, `block_size=32`) and block-scale layout verification. | `14.6µs` (create) / `0ns` (check) | *TBD* |
| `test_block_quantized_array` | Block Quantization | Verifies `BlockQuantizedArray::new` dimensions (`len=64`, `block_size=32`) and scale block count (`2` blocks). | `600ns` | *TBD* |
| `test_bit2_array_and_quantization` | 2-Bit Array | Tests `Bit2Array::from_f32_slice` packing, element retrieval (`Bit2Val`), and `dot_f32` over 4 elements. | `167.5µs` | *TBD* |
| `stress_test_large_dot_product` | Stress / Dot Product | Stress-tests `Bit2Array::dot_f32` on a 4096-dimensional sinusoidal vector and asserts finiteness (no `NaN`/`Inf`). | `616.9µs` | *TBD* |
| `test_ops_math` | Core Ops | Verifies `rms_norm` and `softmax` numerical stability and probability normalization (`sum ≈ 1.0`). | `2.6µs` | *TBD* |
| `test_embedding` | Embeddings | Validates `Embedding::forward` output dimensions and out-of-range token ID error handling. | `2.1µs` | *TBD* |
| `test_tokenizer` | Tokenizer | Verifies token insertion, text encoding, and decoding round-trip behavior. | `98.3µs` | *TBD* |
| `test_block_quantized_array_dot_accuracy` | Block Quantization | Runs 1000 iterations of `BlockQuantizedArray::dot` (`dim=128`, `block_size=32`) and checks accuracy against `f32` reference. | `5.7263ms` | *TBD* |
| `test_advanced_sampling_penalties_and_rng` | Sampling | Runs 500 calls of `Bit2Transformer::sample_advanced` with temperature, `top_k`, `top_p`, and `repetition_penalty` via `FastRng`. | `6.1071ms` | *TBD* |
| `test_transformer_forward_pass` | Transformer | Verifies a single `Bit2Transformer::forward` pass with per-layer `KVCache` and output logit dimensions. | `1.1808ms` | *TBD* |
| `test_bit2_linear` | Linear Layer | Tests `Bit2Linear::new` and `forward` pass output dimensions for an 8x4 layer. | `14.0158ms` | *TBD* |
| `stress_test_concurrent_parallel_inference` | Concurrency (Rayon) | Runs parallel inference across 16 threads with per-thread `Bit2Transformer` clones and `KVCache` vectors. | `16.0266ms` | *TBD* |
| `test_generate_stream` | Async Streaming | Tests async token streaming (`generate_stream`) over a Tokio `mpsc` channel and verifies token count and vocab bounds. | `9.2756ms` | *TBD* |
| `test_c_api_end_to_end_generation` | C ABI / FFI | Tests end-to-end `quat_transformer_create`, `quat_transformer_generate` with per-layer caches, and `quat_transformer_free`. | `22.7175ms` | *TBD* |
| `test_mixed_precision_block2_and_act_quantization` | Mixed Precision | Benchmarks 500 `MixedPrecisionLayer` (`Block2` + `quantize_activations=true`) passes and checks deviation vs `f32` activations. | `49.0889ms` | *TBD* |
| `benchmark_mixed_precision_layer` | Mixed Precision | Benchmarks 1000 forward passes of `MixedPrecisionLayer` (`64 -> 64`) and verifies output size. | `67.7634ms` | *TBD* |
| `stress_test_long_autoregressive_loop` | Autoregressive Loop | Runs a 200-step autoregressive loop on a 2-layer, 4-head `Bit2Transformer` with KV cache updates. | `370.9101ms` | *TBD* |

---

## 🛠️ Project Structure

- `src/bit2.rs` / `array.rs` — 2-bit value definitions and ultra-dense memory packing.
- `src/ops.rs` — Mathematical primitives (Softmax, custom quantized dot-products).
- `src/attention.rs` — Attention mechanism implementation.
- `src/model.rs` — Linear layers and model primitives (`Bit2Linear`).
- `src/transformer.rs` — Core transformer inference loop.
- `src/tokenizer.rs` — Tokenizer logic.
- `src/lib.rs` — Main crate root & comprehensive test suites.

---

## 🔌 Multi-Language Usage Examples

### 1. Rust
Add this to your `Cargo.toml`:
```toml
[dependencies]
quat_llm = "1.3.0" 
```

Usage in Rust code:
```rust
use quat_llm::model::Bit2Transformer;
use quat_llm::attention::KVCache;

fn main() {
    let vocab_size = 1000;
    let hidden_dim = 128;
    let num_layers = 2;
    let num_heads = 4;

    // Inicijalizacija modela sa svim potrebnim parametrima (bez suvišnog ? operatora)
    let model = Bit2Transformer::new(vocab_size, hidden_dim, num_layers, num_heads);
    let mut cache = vec![KVCache::new(); num_layers];

    let token_id = 5;
    let pos = 0;

    // Forward prolaz sa KV kešom i obradom grešaka
    let logits = model.forward(token_id, pos, &mut cache).unwrap();
    println!("Output logits length: {}", logits.len());
}
```

### Advanced Sampling & Mixed-Precision Layer Example:
```rust
use quat_llm::array::BlockQuantizedArray;
use quat_llm::model::{LayerWeightType, MixedPrecisionLayer};
use quat_llm::transformer::Bit2Transformer;

fn main() {
    // 1. Advanced sampling with repetition penalty, top-k, and top-p
    let logits = vec![0.2_f32, 4.8, 4.3, 1.1];
    let generated_history = vec![1, 1];
    let next_token = Bit2Transformer::sample_advanced(
        &logits,
        0.7,
        3,
        0.9,
        1.5,
        &generated_history,
    )
    .unwrap();
    println!("Sampled token ID: {}", next_token);

    // 2. MixedPrecisionLayer with Block2 weights and activation quantization
    let rows = vec![BlockQuantizedArray::new(64, 32); 16];
    let mut layer = MixedPrecisionLayer {
        weight_type: LayerWeightType::Block2(rows),
        quantize_activations: true,
        act_block_size: 32,
        act_clip_ratio: 0.95,
    };

    let input = vec![0.5_f32; 64];
    let mut output = vec![0.0_f32; 16];
    layer.forward(&input, &mut output).unwrap();
    println!("Layer output len: {}", output.len());
}

---

### Asynchronous Streaming Example:
```rust
use quat_llm::model::Bit2Transformer;
use quat_llm::attention::KVCache;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocab_size = 50;
    let hidden_dim = 16;
    let num_layers = 1;
    let num_heads = 2;

    let transformer = Bit2Transformer::new(vocab_size, hidden_dim, num_layers, num_heads);
    let mut cache = vec![KVCache::new(); num_layers];
    let prompt_tokens = vec![1, 2, 3];

    let (tx, mut rx) = mpsc::channel(10);

    // Asinhrono generisanje i strimovanje tokena
    transformer.generate_stream(&prompt_tokens, 5, &mut cache, tx).await?;

    // Čitanje generisanih tokena u realnom vremenu
    while let Some(token) = rx.recv().await {
        assert!(token < vocab_size, "Generated token ID out of vocabulary range");
        println!("Strimovani token ID: {}", token);
    }

    Ok(())
}
```

---

### 2. Python (via PyO3)
After building the Python wheel, you can import and use it directly:

```python
import quat_llm

model = quat_llm.PyBit2Transformer(vocab_size=1000, hidden_dim=128)
probabilities = model.forward(1)
print("Top token probabilities:", probabilities[:5])
```

---

### 3. C (FFI / C-API)
If exposed via C-compatible FFI bindings (`extern "C"`):

```c
#include <stdio.h>
#include <stddef.h>

extern void* quat_transformer_create(size_t vocab_size, size_t hidden_dim, size_t num_layers, size_t num_heads);
extern int quat_transformer_generate(
    void* model_ptr,
    const size_t* prompt_ptr,
    size_t prompt_len,
    size_t max_new_tokens,
    size_t* out_ptr,
    size_t out_cap,
    size_t* out_len
);
extern void quat_transformer_free(void* model_ptr);

int main(void) {
    void* model = quat_transformer_create(1000, 128, 2, 4);
    if (!model) {
        return 1;
    }

    size_t prompt[3] = {1, 2, 3};
    size_t out_tokens[16] = {0};
    size_t out_len = 0;

    int status = quat_transformer_generate(model, prompt, 3, 8, out_tokens, 16, &out_len);
    printf("Status: %d, generated tokens: %zu\n", status, out_len);

    quat_transformer_free(model);
    return 0;
}
```

---

### 4. C++
Using the C-compatible export inside C++:

```cpp
#include <iostream>
#include <vector>
#include <cstddef>

extern "C" {
    void* quat_transformer_create(size_t vocab_size, size_t hidden_dim, size_t num_layers, size_t num_heads);
    int quat_transformer_generate(
        void* model_ptr,
        const size_t* prompt_ptr,
        size_t prompt_len,
        size_t max_new_tokens,
        size_t* out_ptr,
        size_t out_cap,
        size_t* out_len
    );
    void quat_transformer_free(void* model_ptr);
}

int main() {
    void* model = quat_transformer_create(1000, 128, 2, 4);
    if (!model) {
        return 1;
    }

    std::vector<size_t> prompt = {1, 2, 3};
    std::vector<size_t> out_tokens(16, 0);
    size_t out_len = 0;

    int status = quat_transformer_generate(
        model,
        prompt.data(),
        prompt.size(),
        8,
        out_tokens.data(),
        out_tokens.size(),
        &out_len
    );

    std::cout << "Status: " << status << ", generated tokens: " << out_len << std::endl;
    quat_transformer_free(model);
    return 0;
}
```

---

### 5. C# (.NET P/Invoke)
Interop with the compiled dynamic library (`.dll` / `.so`):

```csharp
using System;
using System.Runtime.InteropServices;

class Program
{
    private const string LibName = "quat_llm.dll";

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr quat_transformer_create(UIntPtr vocabSize, UIntPtr hiddenDim, UIntPtr numLayers, UIntPtr numHeads);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int quat_transformer_generate(
        IntPtr modelPtr,
        [In] UIntPtr[] promptPtr,
        UIntPtr promptLen,
        UIntPtr maxNewTokens,
        [Out] UIntPtr[] outPtr,
        UIntPtr outCap,
        out UIntPtr outLen
    );

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void quat_transformer_free(IntPtr modelPtr);

    static void Main()
    {
        IntPtr model = quat_transformer_create((UIntPtr)1000, (UIntPtr)128, (UIntPtr)2, (UIntPtr)4);
        if (model == IntPtr.Zero) return;

        UIntPtr[] prompt = { (UIntPtr)1, (UIntPtr)2, (UIntPtr)3 };
        UIntPtr[] outTokens = new UIntPtr[16];
        int status = quat_transformer_generate(
            model,
            prompt,
            (UIntPtr)prompt.Length,
            (UIntPtr)8,
            outTokens,
            (UIntPtr)outTokens.Length,
            out UIntPtr outLen
        );

        Console.WriteLine($"Status: {status}, generated tokens: {outLen}");
        quat_transformer_free(model);
    }
}
```

---

## 🚀 Getting Started & Testing (Rust)

Run tests (including stress and performance benchmarks):

```bash
cargo test -- --nocapture
```

```bash
cargo test --release -- --nocapture
```

---

## 📄 License & Commercial Use Notice

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 

### What this means:
- You are free to use, modify, and share this software under the terms of the AGPL-3.0.
- **Copyleft clause**: If you modify the code and run it as a network service (e.g., via a cloud API), you must make the source code of your modified version available to your users under the same license.

### 💼 Commercial Licensing
If you intend to use `qllm` in a proprietary commercial product, SaaS, or closed-source environment without being bound by the copyleft requirements of AGPL-3.0, **dual-licensing options are available**. 

Please reach out via email at **strahinjastojanovic826@gmail.com** to discuss a commercial license.