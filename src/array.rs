use crate::bit2::Bit2Val;

/// Container holding 4 2-bit values per byte
#[derive(Debug, Clone)]
pub struct Bit2Array {
    pub data: Vec<u8>,
    pub len: usize,
}

impl Bit2Array {
    pub fn new(len: usize) -> Self {
        let byte_len = (len + 3) / 4;
        Bit2Array {
            data: vec![0; byte_len],
            len,
        }
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<Bit2Val> {
        if index >= self.len {
            return None;
        }
        let byte_idx = index / 4;
        let bit_shift = (index % 4) * 2;
        let byte = self.data[byte_idx];
        Some(Bit2Val::from_u8((byte >> bit_shift) & 0b11))
    }

    #[inline(always)]
    pub fn set(&mut self, index: usize, val: Bit2Val) {
        if index >= self.len {
            return;
        }
        let byte_idx = index / 4;
        let bit_shift = (index % 4) * 2;
        let mask = !(0b11 << bit_shift);
        self.data[byte_idx] = (self.data[byte_idx] & mask) | ((val as u8) << bit_shift);
    }

    /// Fast quantized dot product between 2-bit array and f32 slice
    pub fn dot_f32(&self, input: &[f32]) -> f32 {
        assert_eq!(self.len, input.len());
        let mut sum = 0.0f32;

        for (i, &val) in input.iter().enumerate() {
            if let Some(b_val) = self.get(i) {
                sum += b_val.to_f32() * val;
            }
        }
        sum
    }
}