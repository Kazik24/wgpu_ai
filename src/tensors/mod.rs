mod cpu_tensor;
mod flow_functions;
mod gpu_tensor;
mod gpu_vec;
mod pipelines;
mod wgpu_context;

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use bytemuck::Pod;
pub use cpu_tensor::*;
pub use flow_functions::*;
pub use gpu_tensor::*;
pub use gpu_vec::*;
pub use pipelines::*;
use rayon::prelude::*;
pub use wgpu_context::*;
use wide::*;

// https://github.com/kurtschelfthout/tensorken
pub struct Tensor<'a> {
    data: &'a [f32],
    shape: Vec<usize>,
    name: String,
}
