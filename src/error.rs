use std::fmt;

#[derive(Debug)]
pub enum QuatError {
    DimensionMismatch { expected: usize, got: usize },
    InvalidPointer,
    NullPointer,
    AllocationFailed,
}

impl fmt::Display for QuatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuatError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            QuatError::InvalidPointer => write!(f, "Invalid pointer provided"),
            QuatError::NullPointer => write!(f, "Null pointer provided"),
            QuatError::AllocationFailed => write!(f, "Allocation failed"),
        }
    }
}

impl std::error::Error for QuatError {}
pub type Result<T> = std::result::Result<T, QuatError>;