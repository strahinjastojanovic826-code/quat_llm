use crate::array::Bit2Array;
use crate::bit2::Bit2Val;
use crate::model::Bit2Linear;
use std::slice;

#[repr(C)]
pub enum QuatStatus {
    Success = 0,
    ErrNullPointer = 1,
    ErrDimMismatch = 2,
    ErrUnknown = 3,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bit2_linear_create(
    in_features: usize,
    out_features: usize,
) -> *mut Bit2Linear {
    Box::into_raw(Box::new(Bit2Linear::new(in_features, out_features)))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bit2_linear_free(ptr: *mut Bit2Linear) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bit2_linear_forward(
    ptr: *const Bit2Linear,
    input_ptr: *const f32,
    input_len: usize,
    output_ptr: *mut f32,
    output_len: usize,
) -> QuatStatus {
    if ptr.is_null() || input_ptr.is_null() || output_ptr.is_null() {
        return QuatStatus::ErrNullPointer;
    }

    let layer = match ptr.as_ref() {
        Some(l) => l,
        None => return QuatStatus::ErrNullPointer,
    };

    let input = slice::from_raw_parts(input_ptr, input_len);
    let output = slice::from_raw_parts_mut(output_ptr, output_len);

    match layer.forward(input, output) {
        Ok(_) => QuatStatus::Success,
        Err(_) => QuatStatus::ErrDimMismatch,
    }
}