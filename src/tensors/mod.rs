mod any_tensor;
mod bytes_view;
mod cpu_tensor;
mod elementwise;
mod flow_functions;
mod gpu_num;
mod gpu_tensor;
mod gpu_vec;
mod pipelines;
mod quantized;
mod stream;
mod wgpu_context;

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

pub use any_tensor::*;
pub use bytes_view::*;
pub use cpu_tensor::*;
pub use elementwise::*;
pub use flow_functions::*;
pub use gpu_num::*;
pub use gpu_tensor::*;
pub use gpu_vec::*;
pub use pipelines::*;
pub use quantized::*;
use rayon::prelude::*;
pub use stream::*;
pub use wgpu_context::*;
use wide::*;

// https://github.com/kurtschelfthout/tensorken
// pub struct Tensor<'a> {
//     data: &'a [f32],
//     shape: Vec<usize>,
//     name: String,
// }
