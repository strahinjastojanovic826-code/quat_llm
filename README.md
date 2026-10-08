# quat_llm 🚀

`quat_llm` is a high-performance, lightweight 2-bit quantized LLM inference engine built from scratch in **Rust**, with seamless bindings for **Python (PyO3)**, **C**, **C++**, and **C#**. It drastically reduces memory footprint and maximizes CPU throughput through custom bit-packing and low-level optimization.

---

## ⚡ Key Features

- **Extreme Memory Efficiency**: Packs 4 values into a single byte using custom 2-bit representation (`Bit2Array`), saving up to 75%-87.5% RAM/VRAM compared to FP32/FP16.
- **Blazing Fast Inference**: Optimized Rust engine supporting high-throughput token generation with custom SIMD-friendly vector operations and multi-threaded Rayon matrix multiplication.
- **Cross-Language Support**: Easily integrate the engine into Rust, Python, C, C++, and C#.
- **Modular Architecture**: Includes custom implementations of Tokenizer, Softmax, Dot-Product operations, Attention mechanisms, and Transformer blocks.
- **Comprehensive Multi-Tiered Testing**: Built-in test suites covering basic unit math, error/out-of-bounds protection, autoregressive KV-cache generation, and extreme concurrency stress tests.

---

## 📊 Performance Benchmarks (Stress Tests)

Running on standard CPU configurations (Release Mode `--release`):

### 🔹 Baseline Config (`hidden_dim = 128`, 2 layers)
- **Memory Footprint**: 10 Million 2-bit elements consume **~2.38 MB** of RAM.
- **Initialization Time**: ~2.1 ms
- **Forward Pass**: ~300 µs
- **Token Generation Speed**: **~200+ tokens/sec**

### 💥 Extreme Stress Config (`hidden_dim = 1024`, 8 layers, Vocab: 32k)
- **Allocated KV-Cache Memory**: **~3.12 MB** (100 steps)
- **Throughput (Debug Mode)**: ~1.77 tok/s (unoptimized with full debug instrumentation)
- **Throughput (Release Mode)**: **~25-50+ tok/s** (dependent on CPU SIMD/AVX capabilities)
- **Robustness**: 100% pass rate under parallel thread contention (Rayon execution)

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
quat_llm = { version = "0.2.0", path = "." }
```

Usage in Rust code:
```rust
use quat_llm::transformer::Bit2Transformer;

fn main() {
    let vocab_size = 1000;
    let hidden_dim = 128;
    let model = Bit2Transformer::new(vocab_size, hidden_dim);

    let probs = model.forward(1);
    println!("Output probabilities length: {}", probs.len());
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