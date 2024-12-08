use super::{BytesView, CpuTensor, GpuNum, GpuTensor};

pub struct Tensor<T: GpuNum> {
    repr: TensorRepr<T>,
}

enum TensorRepr<T: GpuNum> {
    Cpu(CpuTensor<T>),
    Gpu(GpuTensor<T>),
}

impl<T: GpuNum> Tensor<T> {
    pub fn from_bytes_view(bytes_view: BytesView<T>, shape: [usize; 2]) -> Self {
        Self {
            repr: TensorRepr::Gpu(GpuTensor::from_shape(&bytes_view, shape)),
        }
    }
}
