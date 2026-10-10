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