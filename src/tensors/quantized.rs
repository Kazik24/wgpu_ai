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
    scales: GpuVec<f32>,
    zero_point: f32, // for all scales
    scales_dim: Dim, //which dimension is quantized, each scale will aply to eather column or row
    quant_type: QuantType,
}

impl QuantizedTensor {
    pub fn quantize(tensor: &GpuTensor<f32>, quant_type: QuantType, scales_dim: Dim) -> Self {
        todo!()
    }

    pub fn dequantize(&self) -> GpuTensor<f32> {
        todo!()
    }
}
