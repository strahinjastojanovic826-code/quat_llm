#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass(name = "Bit2Linear")]
pub struct PyBit2Linear {
    inner: crate::model::Bit2Linear,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBit2Linear {
    #[new]
    fn new(in_features: usize, out_features: usize) -> Self {
        PyBit2Linear {
            inner: crate::model::Bit2Linear::new(in_features, out_features),
        }
    }

    fn forward(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        let mut output = vec![0.0f32; self.inner.out_features];
        self.inner
            .forward(&input, &mut output)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(output)
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn quat_llm(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBit2Linear>()?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyclass(name = "QuatTokenizer")]
pub struct PyQuatTokenizer {
    inner: crate::tokenizer::Tokenizer,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQuatTokenizer {
    #[new]
    fn new() -> Self {
        Self {
            inner: crate::tokenizer::Tokenizer::new(),
        }
    }

    fn add_token(&mut self, token: &str, id: usize) {
        self.inner.add_token(token, id);
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        self.inner.encode(text)
    }

    fn decode(&self, tokens: Vec<usize>) -> String {
        self.inner.decode(&tokens)
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Bit2Transformer")]
pub struct PyBit2Transformer {
    inner: crate::transformer::Bit2Transformer,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBit2Transformer {
    #[new]
    fn new(vocab_size: usize, hidden_dim: usize, num_layers: usize, num_heads: usize) -> Self {
        Self {
            inner: crate::transformer::Bit2Transformer::new(
                vocab_size,
                hidden_dim,
                num_layers,
                num_heads,
            ),
        }
    }

    fn sample_next(
        &self,
        logits: Vec<f32>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> PyResult<usize> {
        self.inner
            .sample(&logits, temperature, top_k, top_p)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn generate(&mut self, prompt_tokens: Vec<usize>, max_new_tokens: usize) -> PyResult<Vec<usize>> {
        let mut caches = vec![
            crate::attention::KVCache::new();
            self.inner.layers.len()
        ];
        self.inner
            .generate(&prompt_tokens, max_new_tokens, &mut caches[0])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}