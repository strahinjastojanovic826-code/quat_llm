use crate::array::Bit2Array;
use crate::bit2::Bit2Val;
use crate::error::{QuatError, Result};
use rayon::prelude::*;
use crate::array::{BlockQuantizedArray, QuantizedActivations};

#[derive(Debug, Clone)]
pub struct Bit2Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Vec<Bit2Array>,
    pub block_weights: Option<Vec<BlockQuantizedArray>>,
}

impl Bit2Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut weights = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            weights.push(Bit2Array::new(in_features));
        }
        Bit2Linear {
            in_features,
            out_features,
            weights,
            block_weights: Some(Vec::new()),
        }
    }

    pub fn set_weight(&mut self, row: usize, col: usize, val: Bit2Val) -> Result<()> {
        if row >= self.out_features {
            return Err(QuatError::DimensionMismatch {
                expected: self.out_features,
                got: row,
            });
        }
        self.weights[row].set(col, val)
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.in_features || output.len() != self.out_features {
            return Err(QuatError::DimensionMismatch {
                expected: self.in_features,
                got: input.len(),
            });
        }

        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, out_val)| {
                *out_val = self.weights[row_idx].dot_f32(input).unwrap_or(0.0);
            });

        Ok(())
    }

     pub fn new_blocked(
        in_features: usize,
        out_features: usize,
        block_size: usize,
    ) -> Self {
        let bs = block_size.max(1);
        let mut block_weights = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            block_weights.push(BlockQuantizedArray::from_f32_slice_calibrated(
                &vec![0.0_f32; in_features],
                bs,
                0.0,
            ));
        }
        Self {
            in_features,
            out_features,
            weights: Vec::new(),
            block_weights: Some(block_weights),
        }
    }

    pub fn from_f32_rows_blocked(
        rows: &[&[f32]],
        in_features: usize,
        block_size: usize,
        clip_ratio: f32,
    ) -> Result<Self> {
        let out_features = rows.len();
        let bs = block_size.max(1);
        let mut block_weights = Vec::with_capacity(out_features);

        for row in rows {
            if row.len() != in_features {
                return Err(QuatError::DimensionMismatch {
                    expected: in_features,
                    got: row.len(),
                });
            }
            block_weights.push(BlockQuantizedArray::from_f32_slice_calibrated(
                row,
                bs,
                clip_ratio,
            ));
        }

        Ok(Self {
            in_features,
            out_features,
            weights: Vec::new(),
            block_weights: Some(block_weights),
        })
    }

    pub fn forward_blocked(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.in_features || output.len() != self.out_features {
            return Err(QuatError::DimensionMismatch {
                expected: self.in_features,
                got: input.len(),
            });
        }
        let blocks = self
            .block_weights
            .as_ref()
            .ok_or(QuatError::InvalidPointer)?;

        output
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(row_idx, out_val)| {
                *out_val = blocks[row_idx].dot(input)?;
                Ok::<(), QuatError>(())
            })?;

        Ok(())
    }

    pub fn forward_quantized_act(
        &self,
        q_act: &QuantizedActivations,
        output: &mut [f32],
    ) -> Result<()> {
        if q_act.len != self.in_features || output.len() != self.out_features {
            return Err(QuatError::DimensionMismatch {
                expected: self.in_features,
                got: q_act.len,
            });
        }
        let blocks = self
            .block_weights
            .as_ref()
            .ok_or(QuatError::InvalidPointer)?;

        output
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(row_idx, out_val)| {
                *out_val = blocks[row_idx].dot_quantized_activations(q_act)?;
                Ok::<(), QuatError>(())
            })?;

        Ok(())
    }

}

#[derive(Debug, Clone)]
pub enum LayerWeightType {
    Bit2(Bit2Linear),
    // Ovde možeš lako dodati F32Linear ili Bit4Linear kasnije po potrebi
    F32Direct { weights: Vec<f32>, in_features: usize, out_features: usize },
    Block2(Vec<BlockQuantizedArray>),
}

#[derive(Debug, Clone)]
pub struct MixedPrecisionLayer {
    pub weight_type: LayerWeightType,
    pub quantize_activations: bool,
    pub act_block_size: usize,
    pub act_clip_ratio: f32,
}

impl MixedPrecisionLayer {
    pub fn new_bit2(in_features: usize, out_features: usize) -> Self {
        Self {
            weight_type: LayerWeightType::Bit2(Bit2Linear::new(in_features, out_features)),
            quantize_activations: false,
            act_block_size: 32,
            act_clip_ratio: 0.0,
        }
    }

    pub fn new_f32(weights: Vec<f32>, in_features: usize, out_features: usize) -> Self {
        Self {
            weight_type: LayerWeightType::F32Direct { weights, in_features, out_features },
            quantize_activations: false,
            act_block_size: 32,
            act_clip_ratio: 0.0,
        }
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        match &self.weight_type {
            LayerWeightType::Bit2(layer) => {
    let mut res = vec![0.0f32; output.len()]; // koristi dužinu prosleđenog output-a
    layer.forward(input, &mut res)?;
    output.copy_from_slice(&res);
    Ok(())
}

LayerWeightType::Block2(rows) => {
                if rows.is_empty() || output.len() != rows.len() {
                    return Err(QuatError::DimensionMismatch {
                        expected: rows.len(),
                        got: output.len(),
                    });
                }
                if self.quantize_activations {
                    let q_act = QuantizedActivations::from_f32_slice(
                        input,
                        self.act_block_size,
                        self.act_clip_ratio,
                    );
                    output
                        .par_iter_mut()
                        .enumerate()
                        .try_for_each(|(i, out_val)| {
                            *out_val = rows[i].dot_quantized_activations(&q_act)?;
                            Ok::<(), QuatError>(())
                        })?;
                } else {
                    output
                        .par_iter_mut()
                        .enumerate()
                        .try_for_each(|(i, out_val)| {
                            *out_val = rows[i].dot(input)?;
                            Ok::<(), QuatError>(())
                        })?;
                }
                Ok(())
            }

            LayerWeightType::F32Direct { weights, in_features, out_features } => {
                // Standardno množenje matrica za f32 preciznost (kritični slojevi)
                if input.len() != *in_features || output.len() != *out_features {
                    return Err(QuatError::DimensionMismatch { expected: *in_features, got: input.len() });
                }
                for i in 0..*out_features {
                    let mut sum = 0.0;
                    for j in 0..*in_features {
                        sum += weights[i * *in_features + j] * input[j];
                    }
                    output[i] = sum;
                }
                Ok(())
            }
        }
    }

    pub fn new_block2(
        rows: &[&[f32]],
        in_features: usize,
        block_size: usize,
        clip_ratio: f32,
        quantize_activations: bool,
    ) -> Result<Self> {
        let bs = block_size.max(1);
        let mut blk_rows = Vec::with_capacity(rows.len());
        for row in rows {
            if row.len() != in_features {
                return Err(QuatError::DimensionMismatch {
                    expected: in_features,
                    got: row.len(),
                });
            }
            blk_rows.push(BlockQuantizedArray::from_f32_slice_calibrated(
                row,
                bs,
                clip_ratio,
            ));
        }

        Ok(Self {
            weight_type: LayerWeightType::Block2(blk_rows),
            quantize_activations,
            act_block_size: bs,
            act_clip_ratio: clip_ratio,
        })
    }

    pub fn forward_mixed(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        match &self.weight_type {
            LayerWeightType::Bit2(layer) => layer.forward(input, output),
            LayerWeightType::Block2(rows) => {
                if input.len() % rows.first().map(|r| r.len).unwrap_or(0) != 0
                    || output.len() != rows.len()
                {
                    return Err(QuatError::DimensionMismatch {
                        expected: rows.len(),
                        got: output.len(),
                    });
                }
                if self.quantize_activations {
                    let q_act = QuantizedActivations::from_f32_slice(
                        input,
                        self.act_block_size,
                        self.act_clip_ratio,
                    );
                    output
                        .par_iter_mut()
                        .enumerate()
                        .try_for_each(|(i, out_val)| {
                            *out_val = rows[i].dot_quantized_activations(&q_act)?;
                            Ok::<(), QuatError>(())
                        })?;
                } else {
                    output
                        .par_iter_mut()
                        .enumerate()
                        .try_for_each(|(i, out_val)| {
                            *out_val = rows[i].dot(input)?;
                            Ok::<(), QuatError>(())
                        })?;
                }
                Ok(())
            }
            LayerWeightType::F32Direct {
                weights,
                in_features,
                out_features,
            } => {
                if input.len() != *in_features || output.len() != *out_features {
                    return Err(QuatError::DimensionMismatch {
                        expected: *in_features,
                        got: input.len(),
                    });
                }
                for i in 0..*out_features {
                    let mut sum = 0.0_f32;
                    let row_offset = i * (*in_features);
                    for j in 0..*in_features {
                        sum += weights[row_offset + j] * input[j];
                    }
                    output[i] = sum;
                }
                Ok(())
            }
        }
    }

}