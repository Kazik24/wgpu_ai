use super::{GpuNum, GpuTensor, GpuVec};

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum QuantType {
    I4,
    I8,
    I16,
}
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum Dim {
    Rows,
    Cols,
}

#[derive(Clone)]
pub struct QuantizedTensor {
    data: GpuVec<u32>,
    shape: [usize; 2],
    params: QuantizationParams,
    quant_type: QuantType,
}

// https://pytorch.org/blog/quantization-in-practice/#per-tensor-and-per-channel-quantization-schemes
#[derive(Clone, Debug)]
enum QuantizationParams {
    /// Symetric: real_val = scale[x or y] * quantized_val
    SymetricPerStripe {
        scales: GpuVec<f32>,
        scales_dim: Dim, //which dimension is quantized, each scale will apply to eather column or row
    },
    /// Symetric: real_val = scale * quantized_val
    SymetricGlobal { scale: f32 },
    /// todo, not implemented
    AsymetricGlobal {
        scale: f32,
        zero_point: f32, // for all scales
    },
}

impl QuantizedTensor {
    pub fn quantize(tensor: &GpuTensor<f32>, quant_type: QuantType, scales_dim: Option<Dim>) -> Self {
        todo!()
    }

    pub fn dequantize(&self) -> GpuTensor<f32> {
        todo!()
    }
}
