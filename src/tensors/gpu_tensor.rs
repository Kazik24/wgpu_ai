use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::Debug,
    mem::transmute,
    sync::{Arc, LazyLock},
};

use wgpu::*;

use crate::tensors::wgpu_context::WgpuContext;

use super::{
    op, op_inplace,
    pipelines::{self, WorkgroupSize},
    ActivationType, AnyGpuNum, FlowFunc, GpuNum, GpuVec, NumType, PipelineType,
};

pub enum AnyGpuTensor {
    F32(GpuTensor<f32>),
    I32(GpuTensor<i32>),
    U32(GpuTensor<u32>),
}

#[derive(Copy, Clone, Debug)]
pub enum AnyGpuTensorRef<'a> {
    F32(&'a GpuTensor<f32>),
    I32(&'a GpuTensor<i32>),
    U32(&'a GpuTensor<u32>),
}

impl AnyGpuTensor {
    pub(super) const fn as_ref(&self) -> AnyGpuTensorRef {
        match self {
            Self::F32(t) => AnyGpuTensorRef::F32(t),
            Self::I32(t) => AnyGpuTensorRef::I32(t),
            Self::U32(t) => AnyGpuTensorRef::U32(t),
        }
    }
    pub fn shape(&self) -> [usize; 2] {
        match self {
            Self::F32(t) => t.shape,
            Self::I32(t) => t.shape,
            Self::U32(t) => t.shape,
        }
    }
    pub fn num_type(&self) -> NumType {
        match self {
            Self::F32(_) => NumType::F32,
            Self::I32(_) => NumType::I32,
            Self::U32(_) => NumType::U32,
        }
    }
    #[inline]
    pub fn try_cast<T: GpuNum>(self) -> Result<GpuTensor<T>, Self> {
        AnyGpuNum::try_downcast_tensor(self)
    }
}

impl<'a> AnyGpuTensorRef<'a> {
    pub fn shape(&self) -> [usize; 2] {
        match self {
            Self::F32(t) => t.shape,
            Self::I32(t) => t.shape,
            Self::U32(t) => t.shape,
        }
    }
    pub fn num_type(&self) -> NumType {
        match self {
            Self::F32(_) => NumType::F32,
            Self::I32(_) => NumType::I32,
            Self::U32(_) => NumType::U32,
        }
    }
    pub(super) fn raw(&self) -> &'a wgpu::Buffer {
        match self {
            Self::F32(t) => t.data.rawr(),
            Self::I32(t) => t.data.rawr(),
            Self::U32(t) => t.data.rawr(),
        }
    }
}

#[derive(Clone)]
pub struct GpuTensor<T: GpuNum> {
    data: GpuVec<T>,
    shape: [usize; 2],
}

// Memory alignment: https://webgpufundamentals.org/webgpu/lessons/webgpu-memory-layout.html
//note: bind group layout is inferred from variables used in @compute function, not all variables defined!!!

#[repr(C)]
struct MatrixMulIndexes {
    pub out_rows: u32, //m, height of a and output, x
    pub out_cols: u32, //p, width of b and output, y
    pub length: u32,   // n, width of a and height of b, loop-iterations
}

impl MatrixMulIndexes {
    pub fn as_array(&self) -> [u32; 3] {
        [self.out_rows, self.out_cols, self.length]
    }
}

impl<T: GpuNum> GpuTensor<T> {
    pub fn new(data: &[impl AsRef<[T]>]) -> Self {
        Self::new_with(data, BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST)
    }
    pub fn new_with(data: &[impl AsRef<[T]>], usage: BufferUsages) -> Self {
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
        let continus = continus.collect::<Vec<_>>();
        let data = GpuVec::with_usage(&continus, usage);

        let shape = [x, y.expect("empty data")];

        let data = Self { data, shape };
        assert!(data.len() > 0, "empty data");
        data
    }

    pub fn from_shape(data: &[T], shape: [usize; 2]) -> Self {
        Self::from_shape_with(data, shape, BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST)
    }
    pub fn from_shape_with(data: &[T], shape: [usize; 2], usage: BufferUsages) -> Self {
        let size = shape[0] * shape[1];
        if size != data.len() {
            panic!("data length must be equal to shape[0] * shape[1]");
        }
        let data = GpuVec::with_usage(data, usage);
        Self { data, shape }
    }
    pub fn empty(shape: [usize; 2]) -> Self {
        Self::empty_with(shape, BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST)
    }
    pub fn empty_with(shape: [usize; 2], usage: BufferUsages) -> Self {
        assert!(shape[0] > 0 && shape[1] > 0, "shape must be greater than 0");
        let size = shape[0] * shape[1];
        let data = GpuVec::empty_with_usage(size, usage);
        Self { data, shape }
    }

    pub fn get_array(&self) -> Vec<Box<[T]>> {
        let data = self.data.to_cpu(..);
        data.chunks_exact(self.shape[1]).map(|c| c.to_vec().into_boxed_slice()).collect()
    }

    /// Shape of the tensor, [rows, cols]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn len(&self) -> usize {
        self.shape[0] * self.shape[1]
    }

    pub fn copy_to(&self, dst: &mut GpuTensor<T>) {
        assert!(self.shape() == dst.shape(), "tensors must be the same shape");
        self.data.copy_to(&mut dst.data);
    }
    pub fn copy_from(&mut self, src: &GpuTensor<T>) {
        src.copy_to(self);
    }

    #[inline]
    pub fn as_any(self) -> AnyGpuTensor {
        AnyGpuNum::upcast_tensor(self)
    }
    #[inline]
    pub fn as_any_ref(&self) -> AnyGpuTensorRef {
        AnyGpuNum::upcast_tensor_ref(self)
    }

    pub fn apply<R: GpuNum>(&self, func: FlowFunc) -> GpuTensor<R> {
        op(self, func)
    }
    pub fn apply2<A: GpuNum, R: GpuNum>(&self, other: &GpuTensor<A>, func: FlowFunc) -> GpuTensor<R> {
        op((self, other), func)
    }
    pub fn apply3<A: GpuNum, B: GpuNum, R: GpuNum>(&self, other1: &GpuTensor<A>, other2: &GpuTensor<B>, func: FlowFunc) -> GpuTensor<R> {
        op((self, other1, other2), func)
    }
    pub fn apply4<A: GpuNum, B: GpuNum, C: GpuNum, R: GpuNum>(
        &self, other1: &GpuTensor<A>, other2: &GpuTensor<B>, other3: &GpuTensor<C>, func: FlowFunc,
    ) -> GpuTensor<R> {
        op((self, other1, other2, other3), func)
    }
    pub fn apply_assign(&mut self, func: FlowFunc) {
        op_inplace(self, func);
    }
    pub fn apply_assign2<A: GpuNum>(&mut self, other: &GpuTensor<A>, func: FlowFunc) {
        op_inplace((self, other), func);
    }
    pub fn apply_assign3<A: GpuNum, B: GpuNum>(&mut self, other1: &GpuTensor<A>, other2: &GpuTensor<B>, func: FlowFunc) {
        op_inplace((self, other1, other2), func);
    }
    pub fn apply_assign4<A: GpuNum, B: GpuNum, C: GpuNum>(&mut self, other1: &GpuTensor<A>, other2: &GpuTensor<B>, other3: &GpuTensor<C>, func: FlowFunc) {
        op_inplace((self, other1, other2, other3), func);
    }
    pub fn activation_assign(&mut self, activation: ActivationType) {
        self.apply_assign(match T::num_type() {
            NumType::F32 => FlowFunc::Argument(0, NumType::F32).activation(activation),
            ty => FlowFunc::Argument(0, ty).cast_to(NumType::F32).activation(activation).cast_to(ty), //double cast
        });
    }
    pub fn activation(&self, activation: ActivationType) -> GpuTensor<f32> {
        self.apply(match T::num_type() {
            NumType::F32 => FlowFunc::Argument(0, NumType::F32).activation(activation),
            ty => FlowFunc::Argument(0, ty).cast_to(NumType::F32).activation(activation), //double cast
        })
    }

    pub fn gated_activation_assign<A: GpuNum>(&mut self, activation: ActivationType, mul_activation_by_elementwise: &GpuTensor<A>) {
        assert!(self.shape() == mul_activation_by_elementwise.shape(), "shapes must be equal");
        let func = FlowFunc::Argument(0, T::num_type()).optimize_cast_to(NumType::F32).activation(activation)
            * FlowFunc::Argument(1, A::num_type()).optimize_cast_to(NumType::F32);
        self.apply_assign2(mul_activation_by_elementwise, func.optimize_cast_to(T::num_type()));
    }
    pub fn gated_activation_assign_gate(&self, activation: ActivationType, mul_activation_by_elementwise: &mut GpuTensor<f32>) {
        assert!(self.shape() == mul_activation_by_elementwise.shape(), "shapes must be equal");
        let func = FlowFunc::Argument(0, T::num_type()).optimize_cast_to(NumType::F32).activation(activation) * FlowFunc::Argument(1, NumType::F32);
        op_inplace((self, mul_activation_by_elementwise), func.optimize_cast_to(NumType::F32));
    }

    pub fn gated_activation<A: GpuNum>(&self, activation: ActivationType, mul_activation_by_elementwise: &GpuTensor<A>) -> GpuTensor<f32> {
        assert!(self.shape() == mul_activation_by_elementwise.shape(), "shapes must be equal");
        let func = FlowFunc::Argument(0, T::num_type()).optimize_cast_to(NumType::F32).activation(activation)
            * FlowFunc::Argument(1, T::num_type()).optimize_cast_to(NumType::F32);
        self.apply2(mul_activation_by_elementwise, func.optimize_cast_to(NumType::F32))
    }

    pub fn matrix_mul(&self, other: &Self) -> Self {
        assert!(
            self.shape[1] == other.shape[0],
            "width (shape[1]) of first tensor must be equal to height (shape[0]) of second tensor"
        );
        assert!(self.len() > 0 && other.len() > 0, "tensors must not be empty");

        let mut tout = Self::empty([self.shape[0], other.shape[1]]);
        Self::matrix_mul_native(self, other, &mut tout, false);
        tout
    }

    pub fn matrix_mul_assign(&self, other: &Self, output: &mut Self) {
        assert!(
            self.shape[1] == other.shape[0],
            "width (shape[1]) of first tensor must be equal to height (shape[0]) of second tensor"
        );
        assert!(self.len() > 0 && other.len() > 0, "tensors must not be empty");
        assert!(output.shape() == [self.shape[0], other.shape[1]], "output shape does not match");
        Self::matrix_mul_native(self, other, output, false);
    }

    pub fn matrix_mul_add(&self, other: &Self, output: &mut Self) {
        assert!(
            self.shape[1] == other.shape[0],
            "width (shape[1]) of first tensor must be equal to height (shape[0]) of second tensor"
        );
        assert!(self.len() > 0 && other.len() > 0, "tensors must not be empty");
        assert!(output.shape() == [self.shape[0], other.shape[1]], "output shape does not match");
        Self::matrix_mul_native(self, other, output, true);
    }

    fn matrix_mul_native(value_a: &Self, value_b: &Self, tout: &mut Self, add: bool) {
        let ctx = WgpuContext::get();
        let pipeline = ctx.pipelines.get(PipelineType::MatrixMul(T::num_type(), add), ctx.wg_2d());

        let indexes = MatrixMulIndexes {
            out_rows: value_a.shape()[0] as u32,
            out_cols: value_b.shape()[1] as u32,
            length: value_a.shape()[1] as u32,
        };

        let workgroup_x = (indexes.out_cols as f64 / ctx.wg_2d().x_size as f64).ceil() as u32;
        let workgroup_y = (indexes.out_rows as f64 / ctx.wg_2d().y_size as f64).ceil() as u32;
        let out_length = indexes.out_rows as usize * indexes.out_cols as usize;
        let indexes = GpuVec::with_usage(&indexes.as_array(), BufferUsages::UNIFORM | BufferUsages::COPY_SRC);

        let entries = &bind_entries([(0, value_a.data.rawr()), (1, value_b.data.rawr()), (2, indexes.rawr()), (3, tout.data.rawr())]);
        let commands = ctx.encode_workgroup(&pipeline, entries, workgroup_x, workgroup_y);
        ctx.execute_commands(commands);
    }
}

impl<T: GpuNum> Debug for GpuTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GpuTensor<{}>[", T::num_type().gpu_type())?;
        let array = self.get_array();
        for row in array {
            writeln!(f, "    {row:?},")?;
        }
        write!(f, "]")
    }
}

fn bind_entries<const N: usize>(array: [(u32, &wgpu::Buffer); N]) -> [BindGroupEntry; N] {
    array.map(|(binding, value)| BindGroupEntry {
        binding,
        resource: value.as_entire_binding(),
    })
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rayon::vec;

    use crate::tensors::{ContextPreference, CpuTensor};

    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_matrix_mul() {
        WgpuContext::init(ContextPreference::HighPerformance);
        let tensor1 = GpuTensor::new(&[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]);
        let tensor2 = GpuTensor::new(&[
            [12.0, 11.0, 10.0],
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ]);

        println!("tensor1 {:?}: {tensor1:?}", tensor1.shape());
        println!("tensor2 {:?}: {tensor2:?}", tensor2.shape());

        let mut values = tensor1.matrix_mul(&tensor2);
        println!("values: {values:?}");
        values.activation_assign(ActivationType::SiLU);
        println!("values: {values:?}");

        let dim = 1024 * 2;

        let t1 = (1..=dim).map(|v| (0..dim).map(|i| (i + v) as f32).collect::<Vec<_>>()).collect::<Vec<_>>();
        let t2 = (1..=dim).map(|v| (0..dim).map(|i| (i + v) as f32).collect::<Vec<_>>()).collect::<Vec<_>>();
        println!("data prepared");

        let t1 = GpuTensor::new(&t1);
        let t2 = GpuTensor::new(&t2);
        println!("tensors prepared");

        let start = Instant::now();
        let values = t1.matrix_mul(&t2);
        println!("time: {:.03?}", start.elapsed());
        // let values = &values.get_array()[0];
        // println!("v: {values:?}");
    }

    #[test]
    fn test_activations_normal() {
        WgpuContext::init(ContextPreference::Default);
        let data = (-128..=128).map(|v| v as f32 * 0.1).collect::<Vec<_>>();
        let silu = data.iter().copied().map(|v| ActivationType::SiLU.apply(v)).collect::<Vec<_>>();
        let relu = data.iter().copied().map(|v| ActivationType::ReLU.apply(v)).collect::<Vec<_>>();
        let gelu = data.iter().copied().map(|v| ActivationType::GeLU.apply(v)).collect::<Vec<_>>();
        let sigm = data.iter().copied().map(|v| ActivationType::Sigmoid.apply(v)).collect::<Vec<_>>();

        let start = Instant::now();
        for (vec, act) in [
            (silu, ActivationType::SiLU),
            (relu, ActivationType::ReLU),
            (gelu, ActivationType::GeLU),
            (sigm, ActivationType::Sigmoid),
        ] {
            let mut tensor = GpuTensor::new(&[&data]);
            tensor.activation_assign(act);
            let values = tensor.get_array()[0].clone();
            assert!(vec.iter().zip(values).all(|(a, b)| (a - b).abs() < 0.000001));

            tensor = GpuTensor::new(&[&data]);
            tensor.activation_assign(act);
            let values = tensor.get_array()[0].clone();
            assert!(vec.iter().zip(values).all(|(a, b)| (a - b).abs() < 0.000001));

            tensor = GpuTensor::new(&[&data]).activation(act);
            let values = tensor.get_array()[0].clone();
            assert!(vec.iter().zip(values).all(|(a, b)| (a - b).abs() < 0.000001));
        }
        println!("time: {:.03?}", start.elapsed());
    }

    #[test]
    fn test_mul_enbedings_size() {
        WgpuContext::init(ContextPreference::HighPerformance);
        const DIM1: usize = 92416;
        const DIM2: usize = 4096;
        let data = (0..DIM1).map(|v| vec![v as f32; DIM2]).collect::<Vec<_>>();

        let tensor1 = GpuTensor::new(&data);
        let data2 = (0..DIM2).map(|v| vec![v as f32; 1]).collect::<Vec<_>>();
        let tensor2 = GpuTensor::new(&data2);
        drop(data);
        drop(data2);
        println!("tensors prepared, shape1: {:?}, shape2: {:?}", tensor1.shape(), tensor2.shape());

        let start = Instant::now();

        let res = tensor1.matrix_mul(&tensor2);
    }

    #[test]
    fn test_activations_gated() {
        WgpuContext::init(ContextPreference::Default);
        let data = (-128..=128).map(|v| v as f32 * 0.1).collect::<Vec<_>>();
        let gate = (-128..=128).map(|v| v as f32 * 0.333).collect::<Vec<_>>();
        let silu = data.iter().zip(&gate).map(|(v, g)| ActivationType::SiLU.apply(*v) * g).collect::<Vec<_>>();
        let relu = data.iter().zip(&gate).map(|(v, g)| ActivationType::ReLU.apply(*v) * g).collect::<Vec<_>>();
        let gelu = data.iter().zip(&gate).map(|(v, g)| ActivationType::GeLU.apply(*v) * g).collect::<Vec<_>>();
        let sigm = data.iter().zip(&gate).map(|(v, g)| ActivationType::Sigmoid.apply(*v) * g).collect::<Vec<_>>();

        let start = Instant::now();
        for (vec, act) in [
            (silu, ActivationType::SiLU),
            (relu, ActivationType::ReLU),
            (gelu, ActivationType::GeLU),
            (sigm, ActivationType::Sigmoid),
        ] {
            let mut tensor = GpuTensor::new(&[&data]);
            tensor.gated_activation_assign(act, &GpuTensor::new(&[&gate]));
            let values = tensor.get_array()[0].clone();
            assert!(vec.iter().zip(values).all(|(a, b)| (a - b).abs() < 0.0001));

            tensor = GpuTensor::new(&[&data]);
            let mut result = GpuTensor::new(&[&gate]);
            tensor.gated_activation_assign_gate(act, &mut result);
            let values = result.get_array()[0].clone();
            assert!(vec.iter().zip(values).all(|(a, b)| (a - b).abs() < 0.0001));

            let tensor = GpuTensor::new(&[&data]).gated_activation(act, &GpuTensor::new(&[&gate]));
            let values = tensor.get_array()[0].clone();
            assert!(vec.iter().zip(values).all(|(a, b)| (a - b).abs() < 0.0001));
        }
        println!("time: {:.03?}", start.elapsed());
    }
}
