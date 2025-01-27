use std::{borrow::Cow, collections::HashMap, sync::Arc};

use fxhash::FxHashMap;
use parking_lot::{RwLock, RwLockReadGuard};
use wgpu::*;

use super::{Dim, FlowFunc, NumType, WgpuContext};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum PipelineType {
    MatrixMul(NumType, bool), //second param is if to add value to output matrix
    QuantizeU8,
    Softmax(Dim),
    CustomMatrixMul {
        dot_accumulate_func: FlowFunc, //for elementwise multiply and add
        modify_func: FlowFunc,         //for writing values to output, e.g overwrite, or some_func(current, result)
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
            PipelineType::MatrixMul(ty, add) => {
                let mut code = MATRIX_MUL_SHADER.replacen("@pipeline_workgroup_size", &wg_attribute, 1);
                code = code.replace("@DATA_TYPE", ty.gpu_type());
                let operator = if *add { "+=" } else { "=" };
                code = code.replacen("@OPERATOR", operator, 1);
                Self::compile_compute_pipeline(code.into(), "matrix_mul")
            }
            PipelineType::FunctionElementwise { func, in_place_arg } => {
                let entry_name = "elementwise_func";
                let mut args = [(NumType::F32, false, false); 8];

                let out_ty = func.eval_type();
                let args = func.get_arguments().unwrap().into_iter().enumerate();
                let args = args.map(|(i, fa)| (fa, Some(i as u8) == *in_place_arg, true)).collect::<Vec<_>>();
                if args.len() > VAR_NAMES_MAX.len() {
                    panic!("too many arguments");
                }

                let func_expr = func.compile(VAR_NAMES_MAX);

                let template = elementwise_shader_template(&args, func_expr, entry_name, *in_place_arg, out_ty);
                Self::compile_compute_pipeline(template.replace("@pipeline_workgroup_size", &wg_attribute).into(), entry_name)
            }
            PipelineType::QuantizeU8 => {
                let mut code = QUANTIZE_TENSOR_SHADER.replacen("@pipeline_workgroup_size", &wg_attribute, 1);
                Self::compile_compute_pipeline(code.into(), "quantize_rows_u8")
            }
            PipelineType::Softmax(dim) => {
                let template = softmax_template(*dim, "softmax_f32");
                Self::compile_compute_pipeline(template.replace("@pipeline_workgroup_size", &wg_attribute).into(), "softmax_f32")
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

#[rustfmt::skip]
const VAR_NAMES_MAX: &[&str] = &[
    "arg0", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6", "arg7",
    "arg8", "arg9", "arg10", "arg11", "arg12", "arg13", "arg14", "arg15",
    "arg16", "arg17", "arg18", "arg19", "arg20", "arg21", "arg22", "arg23",
    "arg24", "arg25", "arg26", "arg27", "arg28", "arg29", "arg30", "arg31",
];

const MATRIX_MUL_SHADER: &str = r###"
@group(0) @binding(0)
var<storage, read_write> input_a: array<@DATA_TYPE>; //apparently read_write here is faster than read (by like 10 times on bigger matrices!!!)
@group(0) @binding(1)
var<storage, read> input_b: array<@DATA_TYPE>; //apparently read here is fastest
@group(0) @binding(2)
var<uniform> indexes: MatrixMulIndexes;
@group(0) @binding(3)
var<storage, read_write> output: array<@DATA_TYPE>;

struct MatrixMulIndexes{
    out_rows: u32,
    out_cols: u32,
    length: u32
}

@compute
@pipeline_workgroup_size
fn matrix_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if x >= indexes.out_rows || y >= indexes.out_cols {
        return;
    }

    var sum = @DATA_TYPE(0);
    let a_offset = y * indexes.length;
    for(var i: u32 = 0; i < indexes.length; i++) {
        sum += input_a[a_offset + i] * input_b[x + i * indexes.out_cols];
    }
    output[x + y * indexes.out_cols] @OPERATOR sum;
}
"###;

fn elementwise_shader_template(args: &[(NumType, bool, bool)], func_expr: String, entry_name: &str, out_arg: Option<u8>, out_type: NumType) -> String {
    const SHADER_CODE: &str = r###"
@VAR_DEFS

fn silu_activation(val: f32) -> f32 {
    return val / (1.0 + exp(-val));
}
fn gelu_activation(val: f32) -> f32 {
    return val * 0.5 * (1.0 + tanh(0.7978845608028654 * (val + 0.044715 * val * val * val)));
}
fn sigmoid_activation(val: f32) -> f32 {
    return 1.0 / (1.0 + exp(-val));
}

fn unpack_bf16_to_f32(val: u32) -> vec2f {
    let va = bitcast<f32>(bitcast<u32>(val & 0xFFFF) << 16);
    let vb = bitcast<f32>(bitcast<u32>(val >> 16));
    return vec2f(va, vb);
}


@compute
@pipeline_workgroup_size
fn #ENTRY_FUNC_NAME(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x > arrayLength(&inout0) { //inout0 should be always defined
        return;
    }
@VAR_EXPRS
    let result = @FLOW_FUNC_EXPR
    @ASSIGN_EXPR
}
"###;
    assert!(args.len() > 0);

    let mut var_defs = String::new();
    let mut var_exprs = String::new();

    for (i, (num_type, mutable, exists)) in args.iter().copied().enumerate() {
        if !exists {
            continue;
        }
        let rw = if mutable { "read_write" } else { "read" };
        let ty = num_type.gpu_type();
        var_defs = format!("{var_defs}@group(0) @binding({i})\nvar<storage, {rw}> inout{i}: array<{ty}>;\n");

        var_exprs = format!("{var_exprs}    let arg{i} = inout{i}[global_id.x];\n");
    }

    let mut value = SHADER_CODE.to_string();
    match out_arg {
        Some(i) => value = value.replacen("@ASSIGN_EXPR", &format!("inout{i}[global_id.x] = result;"), 1),
        None => {
            let i = args.len();
            let ty = out_type.gpu_type();
            var_defs = format!("{var_defs}@group(0) @binding({i})\nvar<storage, read_write> out_new: array<{ty}>;\n");
            value = value.replacen("@ASSIGN_EXPR", "out_new[global_id.x] = result;", 1);
        }
    }

    value = value.replacen("@VAR_DEFS", &var_defs, 1);
    value = value.replacen("@VAR_EXPRS", &var_exprs, 1);
    value = value.replacen("@FLOW_FUNC_EXPR", &format!("{func_expr};"), 1);
    value = value.replacen("#ENTRY_FUNC_NAME", entry_name, 1);
    value
}

fn softmax_template(dim: Dim, entry_name: &str) -> String {
    const SHADER_CODE: &str = r###"
@group(0) @binding(0)
var<storage, read_write> data: array<f32>;
@group(0) @binding(2)
var<uniform> size: u32;

@compute
@pipeline_workgroup_size
fn #ENTRY_FUNC_NAME(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= size {
        return;
    }

@INNER_CODE
}
"###;

    const INNER_DIM_ROWS: &str = r###"
    let offset = idx * size;
    var max_value = f32(data[offset]);
    for(var i: u32 = 1; i < size; i++) {
        max_value = max(max_value, f32(data[offset + i]));
    }

    var sum: f32 = 0.0;
    for(var i: u32 = 0; i < size; i++) {
        let exp_val = exp(f32(data[offset + i]) - max_value);
        data[offset + i] = exp_val;
        sum += exp_val;
    }

    for(var i: u32 = 0; i < size; i++) {
        data[offset + i] /= sum;
    }
"###;
    const INNER_DIM_COLS: &str = r###"
    let offset = idx * size;
    var max_value = f32(data[offset]);
    for(var i: u32 = 1; i < size; i++) {
        max_value = max(max_value, f32(data[offset + i * size]));
    }
"###;

    let value = SHADER_CODE.replacen("#ENTRY_FUNC_NAME", entry_name, 1);
    let value = match dim {
        Dim::Rows => value.replace("@INNER_CODE", INNER_DIM_ROWS.trim()),
        Dim::Cols => value.replace("@INNER_CODE", INNER_DIM_COLS.trim()), //todo
    };
    value
}

const QUANTIZE_TENSOR_SHADER: &str = r###"
@group(0) @binding(0)
var<storage, read> input: array<f32>;
@group(0) @binding(1)
var<storage, read_write> output: array<u32>;
@group(0) @binding(2)
var<storage, read_write> scales: array<f32>;
@group(0) @binding(4)
var<uniform> indexes: MatrixSizes;

struct MatrixSizes{
    rows: u32,
    cols: u32,
}


@compute
@pipeline_workgroup_size
fn quantize_rows_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let y = global_id.x;
    if y >= indexes.rows {
        return;
    }

    // calculate scales
    let offset = y * indexes.cols;
    var min_val = input[offset];
    var max_val = input[offset];
    for(var i: u32 = 1; i < indexes.cols; i++) {
        min_val = min(min_val, input[offset + i]);
        max_val = max(max_val, input[offset + i]);
    }
    let scale = max_val - min_val;
    scales[y] = scale;

    // quantize and pack row values into u8
    for(var i: u32 = 0; i < (indexes.cols / 4); i++) {
        var subarray: vec4f = vec4f(0.0, 0.0, 0.0, 0.0);
        let limit = min(i + 4, indexes.cols);
        for(var j: u32 = i + offset;j < limit; j++) {
            subarray[j] = (input[j]  - min_val) / scale;
        }
        output[i] = pack4x8snorm(subarray);
    }
}
"###;

#[test]
fn test_compile() {
    let ctx = WgpuContext::get();
    ctx.pipelines.get(PipelineType::Softmax(Dim::Rows), ctx.wg_1d());
}
