use crate::error::{QuatError, Result};
use crate::model::Bit2Linear;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub struct ModelLoader {
    mmap: Mmap,
}

impl ModelLoader {
    /// Opens a binary model weights file safely using memory mapping
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|_| QuatError::AllocationFailed)?;
        let mmap = unsafe { Mmap::map(&file).map_err(|_| QuatError::AllocationFailed)? };
        Ok(Self { mmap })
    }

    /// Parses and loads weights into a Bit2Linear layer from the memory map
    pub fn load_linear_layer(&self, mut offset: usize, layer: &mut Bit2Linear) -> Result<usize> {
        for row_weights in &mut layer.weights {
            let weights_size = row_weights.data.len();
            if offset + weights_size > self.mmap.len() {
                return Err(QuatError::AllocationFailed);
            }

            let slice = &self.mmap[offset..offset + weights_size];
            row_weights.data.copy_from_slice(slice);
            offset += weights_size;
        }

        Ok(offset)
    }

}

use safetensors::SafeTensors;

pub struct SafeModelLoader {
    safetensors: SafeTensors<'static>,
}

impl SafeModelLoader {
    pub fn load_from_file(path: &str) -> Result<Self> {
        let buffer = std::fs::read(path).map_err(|_| QuatError::AllocationFailed)?;
        // SAFETY: mmap or safe owning reference handling in production
        let safetensors = SafeTensors::deserialize(Box::leak(buffer.into_boxed_slice()))
            .map_err(|_| QuatError::AllocationFailed)?;
        
        Ok(Self { safetensors })
    }

    pub fn get_tensor_data(&self, tensor_name: &str) -> Result<&[u8]> {
        let tensor = self.safetensors.tensor(tensor_name)
            .map_err(|_| QuatError::InvalidPointer)?;
        Ok(tensor.data())
    }

    pub fn convert_and_load_linear_layer(
        &self, 
        tensor_name: &str, 
        in_features: usize, 
        out_features: usize
    ) -> Result<Vec<crate::array::Bit2Array>> {
        // Fetch raw tensor data from safetensors via your existing loader methods
        let raw_data = self.get_tensor_data(tensor_name)?;
        
        // Ensure the byte length matches f32 matrix size (4 bytes per f32)
        let expected_bytes = in_features * out_features * std::mem::size_of::<f32>();
        if raw_data.len() != expected_bytes {
            return Err(QuatError::DimensionMismatch {
                expected: expected_bytes,
                got: raw_data.len(),
            });
        }

        // Reinterpret raw bytes safely into an f32 slice
        let f32_weights: &[f32] = unsafe {
            std::slice::from_raw_parts(
                raw_data.as_ptr() as *const f32,
                in_features * out_features,
            )
        };

        // Create 2-bit quaternary arrays for each output row
        let mut bit2_rows = Vec::with_capacity(out_features);
        for row in 0..out_features {
            let start = row * in_features;
            let end = start + in_features;
            let row_slice = &f32_weights[start..end];

            // Uses your existing Bit2Array quantization from array.rs
            let bit2_array = crate::array::Bit2Array::from_f32_slice(row_slice);
            bit2_rows.push(bit2_array);
        }

        Ok(bit2_rows)
    }

}