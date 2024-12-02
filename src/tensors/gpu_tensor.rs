use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, LazyLock},
};

use wgpu::*;

use crate::tensors::wgpu_context::WgpuContext;

use super::{
    pipelines::{self, WorkgroupSize},
    ActivationType, GpuNum, GpuVec, PipelineType,
};

pub struct GpuTensor<T: GpuNum> {
    data: GpuVec<T>,
    shape: [usize; 2],
}

// Memory alignment: https://webgpufundamentals.org/webgpu/lessons/webgpu-memory-layout.html
//note: bind group layout is inferred from variables used in @compute function, not all variables defined!!!

#[repr(C)]
pub struct MatrixMulIndexes {
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

        let shape = [x, y.unwrap()];

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

    pub fn get_array(&self) -> Vec<Vec<T>> {
        let data = self.data.to_cpu(..);
        data.chunks_exact(self.shape[1]).map(|c| c.to_vec()).collect()
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn len(&self) -> usize {
        self.shape[0] * self.shape[1]
    }
}

impl GpuTensor<f32> {
    pub fn matrix_mul(&self, other: &Self) -> Self {
        assert!(
            self.shape[1] == other.shape[0],
            "width (shape[1]) of first tensor must be equal to height (shape[0]) of second tensor"
        );
        Self::matrix_mul_native(self, other)
    }

    fn matrix_mul_native(value_a: &Self, value_b: &Self) -> Self {
        let ctx = WgpuContext::get();
        let pipeline = ctx.pipelines.get(PipelineType::MatrixMul, ctx.wg_2d());

        let indexes = MatrixMulIndexes {
            out_rows: value_a.shape()[0] as u32,
            out_cols: value_b.shape()[1] as u32,
            length: value_a.shape()[1] as u32,
        };

        let workgroup_x = (indexes.out_cols as f64 / ctx.wg_2d().x_size as f64).ceil() as u32;
        let workgroup_y = (indexes.out_rows as f64 / ctx.wg_2d().y_size as f64).ceil() as u32;
        let out_length = indexes.out_rows as usize * indexes.out_cols as usize;
        let indexes = GpuVec::with_usage(&indexes.as_array(), BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let tout = Self::empty_with(
            [value_a.shape[0], value_b.shape[1]],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );

        let entries = &bind_entries([(0, value_a.data.rawr()), (1, value_b.data.rawr()), (2, indexes.rawr()), (3, tout.data.rawr())]);
        ctx.dispatch_workgroup(&pipeline, entries, workgroup_x, workgroup_y);

        tout
    }

    pub fn activation_assign(&mut self, activation: ActivationType) {
        let ctx = WgpuContext::get();
        let pipeline = ctx.pipelines.get(
            PipelineType::Activation {
                gated: false,
                activation,
                in_place_first_arg: Some(true),
            },
            ctx.wg_1d(),
        );

        let workgroup_x = (self.len() as f64 / ctx.wg_1d().x_size as f64).ceil() as u32;
        ctx.dispatch_workgroup(&pipeline, &bind_entries([(0, self.data.rawr())]), workgroup_x, 1);
    }

    pub fn gated_activation_assign(&mut self, activation: ActivationType, mul_activation_by_elementwise: &Self) {
        assert!(self.shape() == mul_activation_by_elementwise.shape(), "shapes must be equal");
        let ctx = WgpuContext::get();
        let pipeline = ctx.pipelines.get(
            PipelineType::Activation {
                gated: true,
                activation,
                in_place_first_arg: Some(true),
            },
            ctx.wg_1d(),
        );

        let workgroup_x = (self.len() as f64 / ctx.wg_1d().x_size as f64).ceil() as u32;
        let entries = &bind_entries([(0, self.data.rawr()), (1, mul_activation_by_elementwise.data.rawr())]);
        ctx.dispatch_workgroup(&pipeline, entries, workgroup_x, 1);
    }

    pub fn gated_activation_assign_other(&self, activation: ActivationType, mul_activation_by_elementwise: &mut Self) {
        assert!(self.shape() == mul_activation_by_elementwise.shape(), "shapes must be equal");
        let ctx = WgpuContext::get();
        let pipeline = ctx.pipelines.get(
            PipelineType::Activation {
                gated: true,
                activation,
                in_place_first_arg: Some(false),
            },
            ctx.wg_1d(),
        );

        let workgroup_x = (self.len() as f64 / ctx.wg_1d().x_size as f64).ceil() as u32;
        let entries = &bind_entries([(0, self.data.rawr()), (1, mul_activation_by_elementwise.data.rawr())]);
        ctx.dispatch_workgroup(&pipeline, entries, workgroup_x, 1);
    }

    pub fn activation(&self, activation: ActivationType) -> Self {
        let ctx = WgpuContext::get();
        let pipeline = ctx.pipelines.get(
            PipelineType::Activation {
                gated: false,
                activation,
                in_place_first_arg: None,
            },
            ctx.wg_1d(),
        );

        let workgroup_x = (self.len() as f64 / ctx.wg_1d().x_size as f64).ceil() as u32;
        let out = Self::empty_with(self.shape(), BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST);
        let entries = &bind_entries([(0, self.data.rawr()), (2, out.data.rawr())]);
        ctx.dispatch_workgroup(&pipeline, entries, workgroup_x, 1);
        out
    }

    pub fn gated_activation(&self, activation: ActivationType, mul_activation_by_elementwise: &Self) -> Self {
        assert!(self.shape() == mul_activation_by_elementwise.shape(), "shapes must be equal");
        let ctx = WgpuContext::get();
        let pipeline = ctx.pipelines.get(
            PipelineType::Activation {
                gated: true,
                activation,
                in_place_first_arg: None,
            },
            ctx.wg_1d(),
        );

        let workgroup_x = (self.len() as f64 / ctx.wg_1d().x_size as f64).ceil() as u32;
        let out = Self::empty_with(self.shape(), BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST);
        let entries = &bind_entries([(0, self.data.rawr()), (1, mul_activation_by_elementwise.data.rawr()), (2, out.data.rawr())]);
        ctx.dispatch_workgroup(&pipeline, entries, workgroup_x, 1);
        out
    }
}

impl<T: GpuNum + Copy> Debug for GpuTensor<T> {
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

        let t1 = (1..=dim).map(|v| vec![v as f32; dim]).collect::<Vec<_>>();
        let t2 = (1..=dim).map(|v| vec![v as f32; dim]).collect::<Vec<_>>();
        println!("data prepared");

        let t1 = GpuTensor::new(&t1);
        let t2 = GpuTensor::new(&t2);
        println!("tensors prepared");

        let start = Instant::now();
        let values = t1.matrix_mul(&t2);
        println!("time: {:.03?}", start.elapsed());
        //let values = values.get_array()[0][0];
        //println!("v: {values}");
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
            tensor.gated_activation_assign_other(act, &mut result);
            let values = result.get_array()[0].clone();
            assert!(vec.iter().zip(values).all(|(a, b)| (a - b).abs() < 0.0001));

            let tensor = GpuTensor::new(&[&data]).gated_activation(act, &GpuTensor::new(&[&gate]));
            let values = tensor.get_array()[0].clone();
            assert!(vec.iter().zip(values).all(|(a, b)| (a - b).abs() < 0.0001));
        }
        println!("time: {:.03?}", start.elapsed());
    }
}
