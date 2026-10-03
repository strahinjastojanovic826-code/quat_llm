use std::slice;
use crate::array::Bit2Array;
use crate::bit2::Bit2Val;
use crate::model::Bit2Linear;
use crate::ops;

#[unsafe(no_mangle)]
pub extern "C" fn bit2_array_create(len: usize) -> *mut Bit2Array {
    Box::into_raw(Box::new(Bit2Array::new(len)))
}

#[unsafe(no_mangle)]
pub extern "C" fn bit2_array_free(ptr: *mut Bit2Array) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn bit2_array_set(ptr: *mut Bit2Array, index: usize, val: u8) {
    if let Some(arr) = unsafe { ptr.as_mut() } {
        arr.set(index, Bit2Val::from_u8(val));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn bit2_array_dot_f32(
    ptr: *const Bit2Array,
    input_ptr: *const f32,
    input_len: usize,
) -> f32 {
    let arr = match unsafe { ptr.as_ref() } {
        Some(a) => a,
        None => return 0.0,
    };

    if input_ptr.is_null() || input_len != arr.len {
        return 0.0;
    }

    let input_slice = unsafe { slice::from_raw_parts(input_ptr, input_len) };
    arr.dot_f32(input_slice)
}

// ==========================================
// C API for Softmax & Linear Layer
// ==========================================

#[unsafe(no_mangle)]
pub extern "C" fn bit2_softmax(logits_ptr: *mut f32, len: usize) {
    if logits_ptr.is_null() || len == 0 {
        return;
    }
    let logits = unsafe { slice::from_raw_parts_mut(logits_ptr, len) };
    ops::softmax(logits);
}

#[unsafe(no_mangle)]
pub extern "C" fn bit2_linear_create(in_features: usize, out_features: usize) -> *mut Bit2Linear {
    Box::into_raw(Box::new(Bit2Linear::new(in_features, out_features)))
}

#[unsafe(no_mangle)]
pub extern "C" fn bit2_linear_free(ptr: *mut Bit2Linear) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn bit2_linear_forward(
    ptr: *const Bit2Linear,
    input_ptr: *const f32,
    input_len: usize,
    output_ptr: *mut f32,
    output_len: usize,
) {
    let layer = match unsafe { ptr.as_ref() } {
        Some(l) => l,
        None => return,
    };

    if input_ptr.is_null() || output_ptr.is_null() {
        return;
    }

    let input = unsafe { slice::from_raw_parts(input_ptr, input_len) };
    let output = unsafe { slice::from_raw_parts_mut(output_ptr, output_len) };

    layer.forward(input, output);
}