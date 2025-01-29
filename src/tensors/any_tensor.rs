use std::borrow::Cow;

use super::{ActivationType, BytesView, CpuTensor, FlowFunc, GpuNum, GpuTensor};

#[derive(Debug, Clone)]
pub enum Tensor<T: GpuNum> {
    Cpu(CpuTensor<T>),
    Gpu(GpuTensor<T>),
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum TensorType {
    Cpu,
    Gpu,
}

impl<T: GpuNum> Tensor<T> {
    pub fn from_shape_in(ty: TensorType, data: Cow<[T]>, shape: [usize; 2]) -> Self {
        match ty {
            TensorType::Cpu => Self::Cpu(CpuTensor::from_shape(data, shape)),
            TensorType::Gpu => Self::Gpu(GpuTensor::from_shape(&*data, shape)),
        }
    }

    pub fn new_view(data: BytesView<T>, shape: [usize; 2]) -> Self {
        Self::Cpu(CpuTensor::new_view(data, shape))
    }
    pub fn new(data: &[impl AsRef<[T]>]) -> Self {
        Self::new_in(TensorType::Cpu, data)
    }
    pub fn new_in(ty: TensorType, data: &[impl AsRef<[T]>]) -> Self {
        match ty {
            TensorType::Cpu => Self::Cpu(CpuTensor::new(data)),
            TensorType::Gpu => Self::Gpu(GpuTensor::new(data)),
        }
    }
    pub fn empty(shape: [usize; 2]) -> Self {
        Self::empty_in(TensorType::Cpu, shape)
    }
    pub fn empty_in(ty: TensorType, shape: [usize; 2]) -> Self {
        match ty {
            TensorType::Cpu => Self::Cpu(CpuTensor::empty(shape)),
            TensorType::Gpu => Self::Gpu(GpuTensor::empty(shape)),
        }
    }
    pub const fn get_type(&self) -> TensorType {
        match self {
            Self::Cpu(_) => TensorType::Cpu,
            Self::Gpu(_) => TensorType::Gpu,
        }
    }
    pub const fn shape(&self) -> [usize; 2] {
        match self {
            Self::Cpu(t) => t.shape(),
            Self::Gpu(t) => t.shape(),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(t) => t.len(),
            Self::Gpu(t) => t.len(),
        }
    }
    pub fn get_array(&self) -> Cow<[T]> {
        match self {
            Self::Cpu(t) => Cow::Borrowed(t.get_array()),
            Self::Gpu(t) => Cow::Owned(t.get_array()),
        }
    }
    pub fn get_rows(&self) -> impl Iterator<Item = Cow<[T]>> {
        let [rows, cols] = self.shape();
        struct Iter<'a, T: GpuNum> {
            array: Cow<'a, [T]>,
            rows: std::ops::Range<usize>,
            cols: usize,
        }
        impl<'a, T: GpuNum> Iterator for Iter<'a, T>
        where
            Self: 'a,
        {
            type Item = Cow<'a, [T]>;
            fn next(&mut self) -> Option<Self::Item> {
                let i = self.rows.next()?;
                let start = i * self.cols;
                let range = start..(start + self.cols);
                Some(match &self.array {
                    Cow::Borrowed(array) => Cow::Borrowed(&array[range]),
                    Cow::Owned(array) => Cow::Owned(array[range].to_vec()),
                })
            }
        }
        Iter {
            array: self.get_array(),
            rows: 0..rows,
            cols,
        }
    }

    pub fn to_cpu(&mut self) -> &mut CpuTensor<T> {
        match self {
            Self::Cpu(t) => t,
            Self::Gpu(t) => {
                let array = t.get_array();
                *self = Self::Cpu(CpuTensor::from_shape(Cow::Owned(array), t.shape()));
                match self {
                    Self::Cpu(t) => t,
                    _ => unreachable!(),
                }
            }
        }
    }
    pub fn to_cpu_from(src: &GpuTensor<T>) -> Self {
        let array = src.get_array();
        Self::Cpu(CpuTensor::from_shape(Cow::Owned(array), src.shape()))
    }
    pub fn to_gpu(&mut self) -> &mut GpuTensor<T> {
        match self {
            Self::Gpu(t) => t,
            Self::Cpu(t) => {
                let array = t.get_array();
                *self = Self::Gpu(GpuTensor::from_shape(array, t.shape()));
                match self {
                    Self::Gpu(t) => t,
                    _ => unreachable!(),
                }
            }
        }
    }
    pub fn to_gpu_from(src: &CpuTensor<T>) -> Self {
        let array = src.get_array();
        Self::Gpu(GpuTensor::from_shape(array, src.shape()))
    }

    pub fn copy_to(&self, dst: &mut Self) {
        assert!(self.shape() == dst.shape(), "tensors must be the same shape");
        match (self, dst) {
            (Self::Cpu(src), Self::Cpu(dst)) => src.copy_to(dst),
            (Self::Gpu(src), Self::Gpu(dst)) => src.copy_to(dst),
            (Self::Gpu(src), Self::Cpu(dst)) => {
                let src = CpuTensor::from_gpu_tensor(src);
                src.copy_to(dst);
            }
            (Self::Cpu(src), Self::Gpu(dst)) => {
                let src = src.to_gpu_tensor();
                src.copy_to(dst);
            }
        }
    }
    pub fn copy_from(&mut self, src: &Self) {
        src.copy_to(self);
    }

    pub fn elementwise_add_assign(&mut self, other: &Self) {
        match (self, other) {
            (Self::Cpu(t1), Self::Cpu(t2)) => t1.elementwise_add_assign(t2),
            (Self::Gpu(t1), Self::Gpu(t2)) => t1.elementwise_add_assign(t2),
            (Self::Gpu(t1), Self::Cpu(t2)) => {
                let t2 = t2.to_gpu_tensor();
                t1.elementwise_add_assign(&t2);
            }
            (Self::Cpu(t1), Self::Gpu(t2)) => {
                let t2 = CpuTensor::from_gpu_tensor(t2);
                t1.elementwise_add_assign(&t2);
            }
        }
    }

    pub fn activation_assign(&mut self, activation: ActivationType) {
        match self {
            Self::Cpu(t) => t.activation_assign(activation),
            Self::Gpu(t) => t.activation_assign(activation),
        }
    }
    pub fn activation(&self, activation: ActivationType) -> Tensor<f32> {
        match self {
            Self::Cpu(t) => {
                let mut t = t.to_f32();
                t.activation_assign(activation);
                Tensor::Cpu(t)
            }
            Self::Gpu(t) => Tensor::Gpu(t.activation(activation)),
        }
    }
}

impl Tensor<f32> {
    pub fn matrix_mul(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Cpu(t1), Self::Cpu(t2)) => return Self::Cpu(t1.matrix_mul(t2)),
            (Self::Gpu(t1), Self::Gpu(t2)) => return Self::Gpu(t1.matrix_mul(t2)),
            (Self::Cpu(a), Self::Gpu(b)) => return Self::Gpu(a.to_gpu_tensor().matrix_mul(b)),
            (Self::Gpu(a), Self::Cpu(b)) => return Self::Gpu(a.matrix_mul(&b.to_gpu_tensor())),
        }
    }
    pub fn matrix_mul_assign(&self, other: &Self, out: &mut Self) {
        match (self, other, out) {
            (Self::Cpu(t1), Self::Cpu(t2), Self::Cpu(t3)) => t1.matrix_mul_assign(t2, t3),
            (Self::Gpu(t1), Self::Gpu(t2), Self::Gpu(t3)) => t1.matrix_mul_assign(t2, t3),
            (a, b, out) => {
                let mut a = a.clone();
                let a = a.to_gpu();
                let mut b = b.clone();
                let b = b.to_gpu();
                let mut result = Self::Gpu(a.matrix_mul(&b));
                assert!(result.shape() == out.shape());
                result.to_cpu();
                *out = result;
            }
        }
    }

    pub fn rmsnorm(&mut self, weight: &Self, eps: f32, add_unit_offset: bool) {
        match (self, weight) {
            (Self::Cpu(t1), Self::Cpu(t2)) => t1.rmsnorm(t2, eps, add_unit_offset),
            (a, w) => {
                let mut w = w.clone();
                let w = w.to_cpu();
                a.to_cpu().rmsnorm(&w, eps, add_unit_offset);
            }
        }
    }
}
