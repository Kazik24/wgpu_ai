use std::{borrow::Cow, collections::HashMap, sync::Arc};

use fxhash::FxHashMap;
use parking_lot::{RwLock, RwLockReadGuard};
use wgpu::*;

use super::{FlowFunc, WgpuContext};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum PipelineType {
    MatrixMul,
    CustomMatrixMul {
        dot_accumulate_func: FlowFunc, //for elementwise multiply and add
        modify_func: FlowFunc,         //for writing values to output, e.g overwrite, or some_func(current, result)
    },
    Activation {
        gated: bool,
        activation: ActivationType,
        in_place_first_arg: Option<bool>,
    },
    FunctionElementwise {
        func: FlowFunc,
        in_place_arg: Option<u8>,
    },
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum ActivationType {
    SiLU,
    ReLU,
    GeLU,
    Sigmoid,
}
// webgpu builtin functions https://webgpufundamentals.org/webgpu/lessons/webgpu-wgsl-function-reference.html
impl ActivationType {
    fn shader_expression(&self) -> &'static str {
        match self {
            Self::SiLU => "val = silu_activation(val);",
            Self::ReLU => "val = max(0.0, val);",
            Self::GeLU => "val = gelu_activation(val);",
            Self::Sigmoid => "val = sigmoid_activation(val);",
        }
    }

    pub fn apply(&self, val: f32) -> f32 {
        match self {
            Self::SiLU => val * (1.0 / (1.0 + (-val).exp())),
            Self::ReLU => val.max(0.0),
            Self::GeLU => val * (0.5 * (1.0 + (0.7978845608028654 * (val + 0.044715 * val * val * val)).tanh())),
            Self::Sigmoid => 1.0 / (1.0 + (-val).exp()),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct WorkgroupSize {
    pub x_size: u32,
    pub y_size: u32,
}
impl WorkgroupSize {
    pub const fn new(x_size: u32, y_size: u32) -> Self {
        Self { x_size, y_size }
    }
}

type PipelineKey = (PipelineType, WorkgroupSize);
pub struct PipelineRegistry {
    pipelines: RwLock<FxHashMap<PipelineKey, Arc<wgpu::ComputePipeline>>>,
}

impl PipelineRegistry {
    pub fn new() -> Self {
        Self {
            pipelines: RwLock::new(FxHashMap::default()),
        }
    }

    pub fn get(&self, ty: PipelineType, workgroup_size: WorkgroupSize) -> Arc<wgpu::ComputePipeline> {
        let key = (ty, workgroup_size);
        {
            let pipelines = self.pipelines.read();
            if let Some(pipeline) = pipelines.get(&key).cloned() {
                return pipeline;
            };
        }
        let mut pipelines = self.pipelines.write();
        pipelines
            .entry(key)
            .or_insert_with_key(|ty| Arc::new(Self::new_pipeline(&ty.0, workgroup_size)))
            .clone()
    }

    fn new_pipeline(ty: &PipelineType, workgroup_size: WorkgroupSize) -> wgpu::ComputePipeline {
        let wg_attribute = format!("@workgroup_size({}, {})", workgroup_size.x_size, workgroup_size.y_size);
        match ty {
            PipelineType::MatrixMul => {
                let code = MATRIX_MUL_SHADER.replace("@pipeline_workgroup_size", &wg_attribute).into();
                Self::compile_compute_pipeline(code, "matrix_mul_f32")
            }
            PipelineType::Activation {
                gated,
                activation,
                in_place_first_arg,
            } => {
                let activation_expr = activation.shader_expression();
                let entry_name = "activation_func";
                let template = gated_activation_shader_template(*gated, *in_place_first_arg, activation_expr, "f32", entry_name);
                Self::compile_compute_pipeline(template.replace("@pipeline_workgroup_size", &wg_attribute).into(), entry_name)
            }
            _ => unimplemented!(),
        }
    }

    fn compile_compute_pipeline(wgsl_shader_code: Cow<str>, entry_point: &str) -> wgpu::ComputePipeline {
        let ctx = WgpuContext::get();
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl_shader_code),
        });
        let info = futures::executor::block_on(shader.get_compilation_info());
        if !info.messages.is_empty() {
            println!("info: {info:?}");
        }

        ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: PipelineCompilationOptions {
                constants: &HashMap::new(),
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        })
    }
}

const MATRIX_MUL_SHADER: &str = r###"
@group(0) @binding(0)
var<storage, read_write> input_a: array<f32>;
@group(0) @binding(1)
var<storage, read_write> input_b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> indexes: MatrixMulIndexes;
@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

struct MatrixMulIndexes{
    out_rows: u32,
    out_cols: u32,
    length: u32
}

@compute
@pipeline_workgroup_size
fn matrix_mul_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if x >= indexes.out_rows || y >= indexes.out_cols {
        return;
    }

    var sum = 0.0;
    let a_offset = y * indexes.length;
    for(var i: u32 = 0; i < indexes.length; i++) {
        sum += input_a[a_offset + i] * input_b[x + i * indexes.out_cols];
    }
    output[x + y * indexes.out_cols] = sum;
}
"###;

fn gated_activation_shader_template(
    gated: bool,                      // if this is gated activation, gated means activation(in_a) * in_b, otherwise activation(in_a)
    in_place_first_arg: Option<bool>, // if first argument should be modified in-place, None - create a new array, Some(true) - modify the first argument, Some(false) - modify second argument
    activation_expr: &str, // expression of activation function, 'val' is the input value and result should be stored in it e.g `val = max(0.0, val);``
    data_type: &str,       // data type of the inputs/output arrays, e.g. f32
    entry_name: &str,
) -> String {
    const SILU_IN_PLACE_SHADER: &str = r###"
@group(0) @binding(0) // activation argument
var<storage, read_write> inout_a: array<@DATA_TYPE>;
@group(0) @binding(1) // optional gate argument, multiplied with the result of activation
var<storage, read_write> inout_b: array<@DATA_TYPE>;
@group(0) @binding(2) // optional output if in-place is disabled
var<storage, read_write> new_out: array<@DATA_TYPE>;

fn silu_activation(val: f32) -> f32 {
    return val * 1.0 / (1.0 + exp(-val));
}
fn gelu_activation(val: f32) -> f32 {
    return val * 0.5 * (1.0 + tanh(0.7978845608028654 * (val + 0.044715 * val * val * val)));
}
fn sigmoid_activation(val: f32) -> f32 {
    return 1.0 / (1.0 + exp(-val));
}

@compute
@pipeline_workgroup_size
fn #ENTRY_FUNC_NAME(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x > arrayLength(&inout_a) {
        return;
    }
    var val = inout_a[global_id.x];
    @ACTIVATION_EXPR
    @GATED_EXPR
}
"###;

    let gated_expr = match (gated, in_place_first_arg) {
        (true, None) => "new_out[global_id.x] = val * inout_b[global_id.x];",
        (false, None) => "new_out[global_id.x] = val;",
        (true, Some(true)) => "inout_a[global_id.x] = val * inout_b[global_id.x];",
        (false, Some(true)) => "inout_a[global_id.x] = val;",
        (true, Some(false)) => "inout_b[global_id.x] = val * inout_b[global_id.x];",
        (false, Some(false)) => "inout_b[global_id.x] = val;",
    };

    let value = SILU_IN_PLACE_SHADER
        .replace("@GATED_EXPR", gated_expr)
        .replace("@DATA_TYPE", data_type)
        .replace("#ENTRY_FUNC_NAME", entry_name)
        .replace("@ACTIVATION_EXPR", activation_expr);

    // println!("value: {}", value.trim());

    value
}
