/// Represents a 2-bit quantized value: 00, 01, 10, 11
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bit2Val {
    Val00 = 0b00, // Maps to -1.0
    Val01 = 0b01, // Maps to -0.33
    Val10 = 0b10, // Maps to  0.33
    Val11 = 0b11, // Maps to  1.0
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
            0b11 => Bit2Val::Val11,
            _ => unreachable!(),
        }
    }
}