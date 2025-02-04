use std::borrow::Cow;

use half::bf16;
use rayon::prelude::*;
use wide::f32x8;

use super::{ActivationType, BytesView, GpuNum, GpuTensor};

#[derive(Debug, Clone)]
pub struct CpuTensor<T: GpuNum> {
    data: BytesView<T>,
    shape: [usize; 2], // [rows, cols], memory is alinged in rows one after another, so pliting by rows results in continous slices
}

impl<T: GpuNum> CpuTensor<T> {
    pub fn to_gpu_tensor(&self) -> GpuTensor<T> {
        GpuTensor::from_shape(&*self.data, self.shape)
    }
    pub fn from_gpu_tensor(src: &GpuTensor<T>) -> Self {
        let array = src.get_array();
        Self::from_shape(Cow::Owned(array), src.shape())
    }
    pub fn new_view(data: BytesView<T>, shape: [usize; 2]) -> Self {
        assert!(data.len() == shape[0] * shape[1], "data length must be equal to shape[0] * shape[1]");
        assert!(shape[0] * shape[1] > 0, "empty data");
        Self { data, shape }
    }
    pub fn new(data: &[impl AsRef<[T]>]) -> Self {
        let x = data.len();
        let mut y = None;

        let continus = data.iter().flat_map(|val| {
            let array = val.as_ref();
            if let Some(size) = y {
                assert!(size == array.len(), "all arrays must be the same size");
            } else {
                y = Some(array.len());
            }
            array.iter().copied()
        });
        let data = continus.collect::<Vec<_>>();
        let shape = [x, y.unwrap()];

        assert!(shape[0] * shape[1] > 0, "empty data");

        Self {
            data: BytesView::from_slice(data.into_boxed_slice()),
            shape,
        }
    }

    pub fn from_shape(data: Cow<[T]>, shape: [usize; 2]) -> Self {
        let size = shape[0] * shape[1];
        if size != data.len() {
            panic!("data length must be equal to shape[0] * shape[1]");
        }
        let data = BytesView::from_slice(data.into_owned().into_boxed_slice());
        Self { data, shape }
    }

    pub fn empty(shape: [usize; 2]) -> Self {
        assert!(shape[0] > 0 && shape[1] > 0, "shape must be greater than 0");
        let size = shape[0] * shape[1];
        Self {
            data: BytesView::from_slice(vec![T::zero(); size].into_boxed_slice()),
            shape,
        }
    }
    pub const fn shape(&self) -> [usize; 2] {
        self.shape
    }
    pub const fn rows(&self) -> usize {
        self.shape[0]
    }
    pub const fn len(&self) -> usize {
        self.shape[0] * self.shape[1]
    }
    pub fn get_array(&self) -> &[T] {
        &self.data
    }
}

impl<T: GpuNum> CpuTensor<T> {
    pub fn max(&self) -> T {
        let Some((first, rest)) = self.data.split_first() else {
            return T::zero();
        };
        let mut max = *first;
        for x in rest.iter().copied() {
            if x > max {
                max = x;
            }
        }
        max
    }

    pub fn elementwise_mul_assign(&mut self, other: &Self) {
        self.apply_elementwise_assign(other, |a, b| a * b);
    }
    pub fn elementwise_add_assign(&mut self, other: &Self) {
        self.apply_elementwise_assign(other, |a, b| a + b);
    }

    pub fn to_f32(&self) -> CpuTensor<f32> {
        let dim = self.shape[0].max(self.shape[1]);
        let mut out = CpuTensor::empty([self.shape[0], self.shape[1]]);
        self.data.par_chunks(dim).zip(out.data.par_chunks_mut(dim)).for_each(|(a, b)| {
            for (a, b) in a.iter().zip(b) {
                *b = a.as_f32();
            }
        });
        out
    }

    pub fn activation_assign(&mut self, activation: ActivationType) {
        match activation {
            ActivationType::ReLU => self.apply_elementwise(|v| if v < T::zero() { T::zero() } else { v }),
            ActivationType::SiLU => self.apply_elementwise(|val| {
                let mut val = val.as_f32();
                val *= 1.0 / (1.0 + (-val).exp());
                T::from_f32(val)
            }),
            ActivationType::GeLU => self.apply_elementwise(|val| {
                let mut val = val.as_f32();
                val *= 0.5 * (1.0 + ((0.7978845608028654 * (val + 0.044715 * val * val * val) as f64).tanh()) as f32);
                T::from_f32(val)
            }),
            ActivationType::Sigmoid => self.apply_elementwise(|val| {
                let mut val = val.as_f32();
                val = 1.0 / (1.0 + (-val).exp());
                T::from_f32(val)
            }),
        }
    }

    fn apply_elementwise(&mut self, func: impl Fn(T) -> T + Send + Sync) {
        let dim = self.shape[0].max(self.shape[1]);
        self.data.par_chunks_mut(dim).for_each(|a| {
            for i in a.iter_mut() {
                *i = func(*i);
            }
        });
    }

    fn apply_elementwise_assign(&mut self, other: &Self, func: impl Fn(T, T) -> T + Send + Sync) {
        assert!(self.shape == other.shape, "shapes must be equal");
        let dim = self.shape[0].max(self.shape[1]);
        self.data.par_chunks_mut(dim).zip(other.data.par_chunks(dim)).for_each(|(a, b)| {
            for (a, b) in a.iter_mut().zip(b) {
                *a = func(*a, *b);
            }
        });
    }

    pub fn copy_to(&self, dst: &mut Self) {
        assert!(self.shape() == dst.shape(), "tensors must be the same shape");
        dst.data.copy_from_slice(&self.data);
    }
    pub fn copy_from(&mut self, src: &Self) {
        src.copy_to(self);
    }

    pub fn softmax(&mut self) {
        assert!(self.shape[0] == 1 || self.shape[1] == 1, "tensor must be 1D");
        let max_val = self.max();

        let x = &mut *self.data;
        let mut sum = T::zero();
        for i in x.iter_mut() {
            *i = T::from_f32((*i - max_val).as_f32().exp());
            sum += *i;
        }

        for i in x.iter_mut() {
            *i /= sum;
        }
    }
}

impl CpuTensor<f32> {
    pub fn matrix_mul_assign(&self, other: &Self, output: &mut Self) {
        assert!(
            self.shape[1] == other.shape[0],
            "width (shape[1]) of first tensor must be equal to height (shape[0]) of second tensor"
        );
        assert!(
            output.shape[0] == self.shape[0] && output.shape[1] == other.shape[1],
            "output shape does not match"
        );
        matmul(&mut output.data, &self.data, &other.data, self.shape[0], other.shape[1]);
    }
    pub fn matrix_mul(&self, other: &CpuTensor<f32>) -> CpuTensor<f32> {
        let mut out = Self::empty([self.shape[0], other.shape[1]]);
        self.matrix_mul_assign(other, &mut out);
        out
    }

    pub fn rmsnorm(&mut self, weight: &Self, eps: f32, add_unit_offset: bool) {
        assert!(self.shape[1] == self.shape[1] && weight.shape[0] == 1, "weight must have shape [1, cols]");
        let rows = self.rows();
        let weight = &*weight.data;

        self.data.par_chunks_mut(rows).enumerate().for_each(|(i, xb)| {
            rmsnorm(xb, weight, eps, add_unit_offset);
        });
    }
    pub fn rmsnorm_bf16(&mut self, weight: &CpuTensor<bf16>, eps: f32, add_unit_offset: bool) {
        assert!(self.shape[1] == self.shape[1] && weight.shape[0] == 1, "weight must have shape [1, cols]");
        let rows = self.rows();
        let weight = &*weight.data;

        self.data.par_chunks_mut(rows).enumerate().for_each(|(i, xb)| {
            rmsnorm_bf16(xb, weight, eps, add_unit_offset);
        });
    }
}

fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, o: usize) {
    let n_simd = n / 8;

    xout.par_chunks_exact_mut(o).enumerate().for_each(|(j, elem)| {
        let xi = j * n;

        elem.par_chunks_exact_mut(4).enumerate().for_each(|(i, xout_elem)| {
            let new_i = i * 4;
            let ni0: usize = new_i * n;
            let ni1: usize = (new_i + 1) * n;
            let ni2: usize = (new_i + 2) * n;
            let ni3: usize = (new_i + 3) * n;

            xout_elem.iter_mut().for_each(|m| *m = 0.0);

            for j in 0..n_simd {
                let x_vec = f32x8::from(&x[xi + j * 8..xi + j * 8 + 8]);
                let w_vec0 = f32x8::from(&w[ni0 + j * 8..ni0 + j * 8 + 8]);
                let w_vec1 = f32x8::from(&w[ni1 + j * 8..ni1 + j * 8 + 8]);
                let w_vec2 = f32x8::from(&w[ni2 + j * 8..ni2 + j * 8 + 8]);
                let w_vec3 = f32x8::from(&w[ni3 + j * 8..ni3 + j * 8 + 8]);

                xout_elem[0] += (x_vec * w_vec0).reduce_add();
                xout_elem[1] += (x_vec * w_vec1).reduce_add();
                xout_elem[2] += (x_vec * w_vec2).reduce_add();
                xout_elem[3] += (x_vec * w_vec3).reduce_add();
            }
        });
    });
}

fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32, add_unit_offset: bool) {
    let size = x.len();
    let n_simd = size / 8;

    let mut ss_sim = f32x8::ZERO;

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        ss_sim += x_vec * x_vec;
    }

    let mut ss = ss_sim.reduce_add();

    ss /= size as f32;
    ss += eps;
    ss = 1.0 / ss.sqrt();

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        let w_vec = f32x8::from(&weight[j * 8..j * 8 + 8]);

        let r = if add_unit_offset {
            ((1.0 + w_vec) * (ss * x_vec)).to_array()
        } else {
            (w_vec * (ss * x_vec)).to_array()
        };

        for k in 0..8 {
            x[(j * 8) + k] = r[k];
        }
    }
}

fn rmsnorm_bf16(x: &mut [f32], weight: &[bf16], eps: f32, add_unit_offset: bool) {
    let size = x.len();

    let mut ss = x.iter().copied().map(|x| x * x).sum::<f32>();
    ss /= size as f32;
    ss += eps;
    ss = 1.0 / ss.sqrt();

    if add_unit_offset {
        for (xi, wi) in x.iter_mut().zip(weight) {
            *xi = (1.0 + wi.to_f32()) * (ss * *xi);
        }
    } else {
        for (xi, wi) in x.iter_mut().zip(weight) {
            *xi = wi.to_f32() * (ss * *xi);
        }
    }
}
