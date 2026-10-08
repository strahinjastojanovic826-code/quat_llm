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