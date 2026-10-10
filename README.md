# quat_llm 🚀

`quat_llm` is a high-performance, lightweight 2-bit quantized LLM inference engine built from scratch in **Rust**, with seamless bindings for **Python (PyO3)**, **C**, **C++**, and **C#**. It drastically reduces memory footprint and maximizes CPU throughput through custom bit-packing and low-level optimization.

---

## ⚡ Key Features

- **Extreme Memory Efficiency**: Packs 4 values into a single byte using custom 2-bit quantization (`Bit2Array`), drastically reducing RAM usage compared to FP32/FP16[cite: 6].
- **Blazing Fast Inference**: Cache-optimized parallel execution engine leveraging Rayon for multi-threaded matrix-vector dot products and chunk-based processing[cite: 6].
- **Cross-Language Support**: Seamlessly integrate the engine into Rust, Python, C, C++, and C#[cite: 6].
- **Modular Architecture**: Includes clean, robust implementations of Tokenizer, Embedding, RMSNorm, Softmax, RoPE, and Attention with KV-Caching[cite: 6].
- **Production-Grade Robustness**: Built-in comprehensive test suites covering functional behavior, long autoregressive generation loops, and high-concurrency Rayon stress tests[cite: 6].

---

## 📊 Performance & Testing (Stress Tests)

The library includes multi-tiered automated test suites ensuring safety and stability under extreme loads:
- **Functional Unit Tests**: Validation of quantization, tokenization, embeddings, and mathematical primitives[cite: 6].
- **Large-Scale Dot Product Stress Tests**: Verified stability with 4096+ dimensions[cite: 6].
- **Autoregressive Generation Loops**: Long-sequence generation stress tests for KV-cache stability[cite: 6].
- **Concurrent Multi-Threading Stress Tests**: Parallel inference loads verified under Rayon task scheduling[cite: 6].

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
quat_llm = "1.0.0" 
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

    // Inicijalizacija modela sa svim potrebnim parametrima
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