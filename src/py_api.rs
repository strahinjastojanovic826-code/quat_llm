#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use crate::array::Bit2Array;
#[cfg(feature = "python")]
use crate::bit2::Bit2Val;
#[cfg(feature = "python")]
use crate::model::Bit2Linear;
#[cfg(feature = "python")]
use crate::ops;

#[cfg(feature = "python")]
#[pyclass(name = "Bit2Array")]
pub struct PyBit2Array {
    inner: Bit2Array,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBit2Array {
    #[new]
    fn new(len: usize) -> Self {
        PyBit2Array {
            inner: Bit2Array::new(len),
        }
    }

    fn set(&mut self, index: usize, val: u8) {
        self.inner.set(index, Bit2Val::from_u8(val));
    }

    fn get(&self, index: usize) -> Option<u8> {
        self.inner.get(index).map(|v| v as u8)
    }

    fn dot_f32(&self, input: Vec<f32>) -> f32 {
        self.inner.dot_f32(&input)
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Bit2Linear")]
pub struct PyBit2Linear {
    inner: Bit2Linear,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBit2Linear {
    #[new]
    fn new(in_features: usize, out_features: usize) -> Self {
        PyBit2Linear {
            inner: Bit2Linear::new(in_features, out_features),
        }
    }

    fn set_weight(&mut self, row: usize, col: usize, val: u8) {
        self.inner.set_weight(row, col, Bit2Val::from_u8(val));
    }

    fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        let mut output = vec![0.0f32; self.inner.out_features];
        self.inner.forward(&input, &mut output);
        output
    }
}

#[cfg(feature = "python")]
#[pyfunction]
fn softmax(mut logits: Vec<f32>) -> Vec<f32> {
    ops::softmax(&mut logits);
    logits
}

#[cfg(feature = "python")]
#[pymodule]
pub fn bit2_llm(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBit2Array>()?;
    m.add_class::<PyBit2Linear>()?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    Ok(())
}