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

---

## 📊 Performance & Testing (Stress Tests)

| Test Name | Description / What the Test Does | Execution Time |
| :--- | :--- | :--- |
| **EfficientKVCache** | Benchmarks 100 append and get operations on the efficient KV cache | 22.5 µs |
| **BlockQuantizedArray Creation** | Measures the allocation and setup time of a block-quantized array | 26.1 µs |
| **Embedding Forward Pass** | Evaluates the execution speed of the embedding layer forward pass | 6.8 µs |
| **Block Quantized Array Check** | Benchmarks verification and integrity checks on the quantized array | 3.0 µs (Check: 100 ns) |
| **Bit2 Array & Quantization** | Tests 2-bit array conversions and data quantization logic | 27.5 µs |
| **Ops Math (RMS/Softmax)** | Benchmarks foundational math operations (RMS normalization & Softmax) | 4.2 µs |
| **Bit2 Linear Layer** | Evaluates the compute speed of the 2-bit linear transformation layer | 926.4 µs |
| **Tokenizer** | Measures encoding and decoding performance for text tokens | 134.2 µs |
| **Transformer Forward Pass** | Tests the execution time of a single full transformer forward pass | 5.84 ms |
| **Large Dot Product** | Stress tests high-dimensional dot product calculations (4096 dimensions) | 569.2 µs |
| **Concurrent Parallel Inference** | Stress tests multi-threaded inference performance across 16 Rayon threads | 12.18 ms |
| **Generate Stream** | Benchmarks asynchronous streaming token generation using Tokio channels | 11.56 ms |
| **Mixed Precision Layer** | Benchmarks performance across 1,000 continuous forward passes | 72.94 ms |
| **Long Autoregressive Loop** | Stress tests long-form autoregressive sequence generation over 200 steps | 324.94 ms |

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
quat_llm = "1.2.0" 
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

# Initialize and run forward pass
model = quat_llm.PyBit2Transformer(vocab_size=1000, hidden_dim=128)
probabilities = model.forward(1)
print("Top token probabilities:", probabilities[:5])
```

---

### 3. C (FFI / C-API)
If exposed via C-compatible FFI bindings (`extern "C"`):

```c
#include <stdio.h>

// Declare external Rust function
extern void* quat_transformer_new(size_t vocab_size, size_t hidden_dim);
extern void quat_transformer_free(void* model);

int main() {
    void* model = quat_transformer_new(1000, 128);
    printf("Model loaded successfully in C!\n");
    quat_transformer_free(model);
    return 0;
}
```

---

### 4. C++
Using the C-compatible export inside C++:

```cpp
#include <iostream>

extern "C" {
    void* quat_transformer_new(size_t vocab_size, size_t hidden_dim);
    void quat_transformer_free(void* model);
}

int main() {
    void* model = quat_transformer_new(1000, 128);
    std::cout << "Model initialized via C++!" << std::endl;
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
    [DllImport("quat_llm.dll", CallingConvention = CallingConvention.Cdecl)]
    extern static IntPtr quat_transformer_new(IntPtr vocabSize, IntPtr hiddenDim);

    [DllImport("quat_llm.dll", CallingConvention = CallingConvention.Cdecl)]
    extern static void quat_transformer_free(IntPtr model);

    static void Main()
    {
        IntPtr model = quat_transformer_new((IntPtr)1000, (IntPtr)128);
        Console.WriteLine("Model loaded in C# successfully!");
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