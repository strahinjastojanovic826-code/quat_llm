// SAFETY: All `extern "C"` FFI entry points (`bit2_linear_*`, `quat_transformer_*`) expect
// valid/aligned pointers and matching slice lengths from the caller, validate null pointers
// before dereferencing or constructing slices, and pair `Box::into_raw` allocations with a
// single `Box::from_raw` free on non-null pointers without double-free or overlapping mutable aliases.

use crate::model::Bit2Linear;
use std::slice;

#[repr(C)]
#[derive(Debug, PartialEq)]
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
pub unsafe extern "C" fn bit2_linear_free(ptr: *mut Bit2Linear) { unsafe {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bit2_linear_forward(
    ptr: *const Bit2Linear,
    input_ptr: *const f32,
    input_len: usize,
    output_ptr: *mut f32,
    output_len: usize,
) -> QuatStatus { unsafe {
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
}}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn quat_transformer_create(
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
) -> *mut crate::transformer::Bit2Transformer {
    let model = crate::transformer::Bit2Transformer::new(
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
    );
    Box::into_raw(Box::new(model))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn quat_transformer_free(ptr: *mut crate::transformer::Bit2Transformer) {
    if !ptr.is_null() {
        // SAFETY: `ptr` was allocated via `Box::into_raw(Box::new(...))` in `quat_transformer_create`
        // and is freed once here.
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn quat_transformer_generate(
    model_ptr: *mut crate::transformer::Bit2Transformer,
    prompt_ptr: *const usize,
    prompt_len: usize,
    max_new_tokens: usize,
    out_ptr: *mut usize,
    out_cap: usize,
    out_len: *mut usize,
) -> QuatStatus {
    // SAFETY: Validate raw pointers and slice bounds before constructing Rust slices.
    unsafe {
        if model_ptr.is_null() || prompt_ptr.is_null() || out_ptr.is_null() || out_len.is_null() {
            return QuatStatus::ErrNullPointer;
        }
        let Some(model) = model_ptr.as_mut() else {
            return QuatStatus::ErrNullPointer;
        };
        let prompt = slice::from_raw_parts(prompt_ptr, prompt_len);
        let mut caches = vec![crate::attention::KVCache::new(); model.layers.len()];
        let tokens = match model.generate(prompt, max_new_tokens, &mut caches) {
            Ok(t) => t,
            Err(_) => return QuatStatus::ErrUnknown,
        };
        if tokens.len() > out_cap {
            *out_len = tokens.len();
            return QuatStatus::ErrDimMismatch;
        }
        let out_slice = slice::from_raw_parts_mut(out_ptr, tokens.len());
        out_slice.copy_from_slice(&tokens);
        *out_len = tokens.len();
        QuatStatus::Success
    }
}