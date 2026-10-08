use crate::error::{QuatError, Result};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bit2Val {
    Val00 = 0b00, // -1.0
    Val01 = 0b01, // -0.33333333
    Val10 = 0b10, //  0.33333333
    Val11 = 0b11, //  1.0
}

impl Bit2Val {
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        match self {
            Bit2Val::Val00 => -1.0,
            Bit2Val::Val01 => -0.33333333,
            Bit2Val::Val10 => 0.33333333,
            Bit2Val::Val11 => 1.0,
        }
    }

    #[inline(always)]
    pub fn from_u8(val: u8) -> Self {
        match val & 0b11 {
            0b00 => Bit2Val::Val00,
            0b01 => Bit2Val::Val01,
            0b10 => Bit2Val::Val10,
            _ => Bit2Val::Val11,
        }
    }

    /// Quantize a float value to 2-bit representation based on a scale factor
    #[inline(always)]
    pub fn quantize(val: f32, scale: f32) -> Self {
        let norm = if scale == 0.0 { 0.0 } else { val / scale };
        if norm < -0.6666666 {
            Bit2Val::Val00
        } else if norm < 0.0 {
            Bit2Val::Val01
        } else if norm < 0.6666666 {
            Bit2Val::Val10
        } else {
            Bit2Val::Val11
        }
    }
}