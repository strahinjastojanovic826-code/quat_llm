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
        // SAFETY: `file` is opened and held until `Mmap::map` completes; caller/environment ensures
        // the underlying file is not concurrently truncated or modified in an undefined manner while mapped.
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

    /// Reads `count` f32 values starting at `byte_offset` from the memory map
    /// using unaligned/safe byte-to-f32 conversion.
    pub fn read_f32_vec(&self, byte_offset: usize, count: usize) -> Result<(Vec<f32>, usize)> {
        let byte_len = count.checked_mul(std::mem::size_of::<f32>()).ok_or(QuatError::AllocationFailed)?;
        let end = byte_offset.checked_add(byte_len).ok_or(QuatError::AllocationFailed)?;
        if end > self.mmap.len() {
            return Err(QuatError::AllocationFailed);
        }

        let mut out = Vec::with_capacity(count);
        for chunk in self.mmap[byte_offset..end].chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok((out, end))
    }

    /// Loads a linear layer from raw `f32` bytes in `Mmap` into quaternary `BlockQuantizedArray` rows.
    pub fn load_blocked_linear_from_mmap(
        &self,
        mut byte_offset: usize,
        in_features: usize,
        out_features: usize,
        block_size: usize,
        clip_ratio: f32,
    ) -> Result<(Vec<crate::array::BlockQuantizedArray>, usize)> {
        let mut rows = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            let (row_f32, next_offset) = self.read_f32_vec(byte_offset, in_features)?;
            byte_offset = next_offset;
            rows.push(crate::array::BlockQuantizedArray::from_f32_slice_calibrated(
                &row_f32,
                block_size,
                clip_ratio,
            ));
        }
        Ok((rows, byte_offset))
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
        // SAFETY: `raw_data.len()` was verified to equal `in_features * out_features * size_of::<f32>()`
        // above; `raw_data` remains borrowed for the lifetime of `f32_weights` and is not mutated while read.
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

/// Owns the raw SafeTensors byte buffer so no memory is leaked via `Box::leak`.
#[derive(Debug)]
pub struct OwnedSafeModelLoader {
    raw_bytes: Vec<u8>,
}

impl OwnedSafeModelLoader {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw_bytes = std::fs::read(path).map_err(|_| QuatError::AllocationFailed)?;
        // Validate header/tensor layout eagerly
        let _ = SafeTensors::deserialize(&raw_bytes).map_err(|_| QuatError::AllocationFailed)?;
        Ok(Self { raw_bytes })
    }

    pub fn tensor_f32_vec(&self, tensor_name: &str) -> Result<Vec<f32>> {
        let st = SafeTensors::deserialize(&self.raw_bytes).map_err(|_| QuatError::InvalidPointer)?;
        let tensor = st.tensor(tensor_name).map_err(|_| QuatError::InvalidPointer)?;
        let data = tensor.data();
        if data.len() % 4 != 0 {
            return Err(QuatError::DimensionMismatch {
                expected: (data.len() / 4) * 4,
                got: data.len(),
            });
        }
        let mut vals = Vec::with_capacity(data.len() / 4);
        for chunk in data.chunks_exact(4) {
            vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(vals)
    }

    pub fn convert_to_block2_rows(
        &self,
        tensor_name: &str,
        in_features: usize,
        out_features: usize,
        block_size: usize,
        clip_ratio: f32,
    ) -> Result<Vec<crate::array::BlockQuantizedArray>> {
        let expected_len = in_features
            .checked_mul(out_features)
            .ok_or(QuatError::AllocationFailed)?;
        let weights = self.tensor_f32_vec(tensor_name)?;
        if weights.len() != expected_len {
            return Err(QuatError::DimensionMismatch {
                expected: expected_len,
                got: weights.len(),
            });
        }

        let mut rows = Vec::with_capacity(out_features);
        for row in 0..out_features {
            let start = row * in_features;
            let end = start + in_features;
            rows.push(crate::array::BlockQuantizedArray::from_f32_slice_calibrated(
                &weights[start..end],
                block_size,
                clip_ratio,
            ));
        }
        Ok(rows)
    }
}