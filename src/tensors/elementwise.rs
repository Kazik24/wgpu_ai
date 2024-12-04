use wgpu::BindGroupEntry;

use crate::tensors::{PipelineType, WgpuContext};

use super::{AnyGpuTensor, AnyGpuTensorRef, FlowFunc, GpuNum, GpuTensor, NumType};

#[allow(non_snake_case)]
pub trait ElementwiseOp<T> {
    const RETURN_ARG: Option<u8>;

    fn arguments(&self) -> Vec<AnyGpuTensorRef>;
}

macro_rules! impl_elementwise_op {
    (@ mut $gen:ident) => {&mut GpuTensor<$gen>};
    (@ const $gen:ident) => {&GpuTensor<$gen>};
    ($($mutable:ident $gen:ident),* : $arg:expr $(, $out:ty)?) => {
        impl<$($gen: GpuNum),*> ElementwiseOp<($($out)?)> for ($(impl_elementwise_op!(@ $mutable $gen)),*) {
            const RETURN_ARG: Option<u8> = $arg;

            #[allow(non_snake_case)]
            fn arguments(&self) -> Vec<AnyGpuTensorRef> {
                let ($($gen),*) = self;
                let array = [$($gen.as_any_ref()),*];
                array.to_vec()
            }
        }
    };
}

macro_rules! impl_elementwise_op_gen {
    (@ mut $gen:ident) => {&mut GpuTensor<$gen>};
    (@ const $gen:ident) => {&GpuTensor<$gen>};
    ($($mutable:ident $gen:ident),* : $arg:expr) => {
        impl<$($gen: GpuNum),*, RETURN: GpuNum> ElementwiseOp<RETURN> for ($(impl_elementwise_op!(@ $mutable $gen)),*) {
            const RETURN_ARG: Option<u8> = $arg;

            #[allow(non_snake_case)]
            fn arguments(&self) -> Vec<AnyGpuTensorRef> {
                let ($($gen),*) = self;
                let array = [$($gen.as_any_ref()),*];
                array.to_vec()
            }
        }
    };
}

// producing new tensor
impl_elementwise_op_gen!(const A : None);
impl_elementwise_op_gen!(const A, const B : None);
impl_elementwise_op_gen!(const A, const B, const C : None);
impl_elementwise_op_gen!(const A, const B, const C, const D : None);
impl_elementwise_op_gen!(const A, const B, const C, const D, const E : None);
impl_elementwise_op_gen!(const A, const B, const C, const D, const E, const F : None);
impl_elementwise_op_gen!(const A, const B, const C, const D, const E, const F, const G : None);
impl_elementwise_op_gen!(const A, const B, const C, const D, const E, const F, const G, const H : None);

// reusing existing tensor
impl_elementwise_op!(mut A : Some(0));

impl_elementwise_op!(mut A, const B : Some(0));
impl_elementwise_op!(const A, mut B : Some(1));

impl_elementwise_op!(mut A, const B, const C : Some(0));
impl_elementwise_op!(const A, mut B, const C : Some(1));
impl_elementwise_op!(const A, const B, mut C : Some(2));

impl_elementwise_op!(mut A, const B, const C, const D : Some(0));
impl_elementwise_op!(const A, mut B, const C, const D : Some(1));
impl_elementwise_op!(const A, const B, mut C, const D : Some(2));
impl_elementwise_op!(const A, const B, const C, mut D : Some(3));

impl_elementwise_op!(mut A, const B, const C, const D, const E : Some(0));
impl_elementwise_op!(const A, mut B, const C, const D, const E : Some(1));
impl_elementwise_op!(const A, const B, mut C, const D, const E : Some(2));
impl_elementwise_op!(const A, const B, const C, mut D, const E : Some(3));
impl_elementwise_op!(const A, const B, const C, const D, mut E : Some(4));

impl_elementwise_op!(mut A, const B, const C, const D, const E, const F : Some(0));
impl_elementwise_op!(const A, mut B, const C, const D, const E, const F : Some(1));
impl_elementwise_op!(const A, const B, mut C, const D, const E, const F : Some(2));
impl_elementwise_op!(const A, const B, const C, mut D, const E, const F : Some(3));
impl_elementwise_op!(const A, const B, const C, const D, mut E, const F : Some(4));
impl_elementwise_op!(const A, const B, const C, const D, const E, mut F : Some(5));

impl_elementwise_op!(mut A, const B, const C, const D, const E, const F, const G : Some(0));
impl_elementwise_op!(const A, mut B, const C, const D, const E, const F, const G : Some(1));
impl_elementwise_op!(const A, const B, mut C, const D, const E, const F, const G : Some(2));
impl_elementwise_op!(const A, const B, const C, mut D, const E, const F, const G : Some(3));
impl_elementwise_op!(const A, const B, const C, const D, mut E, const F, const G : Some(4));
impl_elementwise_op!(const A, const B, const C, const D, const E, mut F, const G : Some(5));
impl_elementwise_op!(const A, const B, const C, const D, const E, const F, mut G : Some(6));

impl_elementwise_op!(mut A, const B, const C, const D, const E, const F, const G, const H : Some(0));
impl_elementwise_op!(const A, mut B, const C, const D, const E, const F, const G, const H : Some(1));
impl_elementwise_op!(const A, const B, mut C, const D, const E, const F, const G, const H : Some(2));
impl_elementwise_op!(const A, const B, const C, mut D, const E, const F, const G, const H : Some(3));
impl_elementwise_op!(const A, const B, const C, const D, mut E, const F, const G, const H : Some(4));
impl_elementwise_op!(const A, const B, const C, const D, const E, mut F, const G, const H : Some(5));
impl_elementwise_op!(const A, const B, const C, const D, const E, const F, mut G, const H : Some(6));
impl_elementwise_op!(const A, const B, const C, const D, const E, const F, const G, mut H : Some(7));

pub fn op<T: GpuNum>(args: impl ElementwiseOp<T>, func: FlowFunc) -> GpuTensor<T> {
    let arg_tensors = args.arguments();
    let result = runtime_op_internal(&arg_tensors, None, func);
    let result = result.expect("expected new tensor function");
    match result.try_cast() {
        Err(ty) => panic!("Wrong output type inferred, expected {:?}, got {ty:?}", T::num_type()),
        Ok(result) => result,
    }
}

pub fn op_inplace<O: ElementwiseOp<()>>(args: O, func: FlowFunc) {
    let arg_tensors = args.arguments();
    assert!(O::RETURN_ARG.is_some(), "expected in place operation");
    let result = runtime_op_internal(&arg_tensors, O::RETURN_ARG, func);
    assert!(result.is_none(), "expected in place operation");
}

fn runtime_op_internal(args: &[AnyGpuTensorRef], in_place: Option<u8>, func: FlowFunc) -> Option<AnyGpuTensor> {
    assert!(!args.is_empty() && args.len() <= 8, "bad number of arguments {}", args.len());
    let shape = args[0].shape();
    let func_args = func.get_arguments().unwrap();
    assert!(func_args.len() == args.len(), "number of arguments does not match FlowFunc");
    if let Some(i) = in_place {
        assert!((i as usize) < args.len(), "in place argument index out of bounds");
    }
    for (i, (arg, exp_type)) in args.iter().zip(&func_args).enumerate() {
        assert!(arg.shape() == shape, "argument {i} has different shape than first tensor");
        let ty = arg.num_type();
        assert!(ty == *exp_type, "argument {i} has different type, expected {exp_type:?} got {ty:?}");
    }
    let count = shape[0] as usize * shape[1] as usize;
    assert!(count > 0, "empty tensor");
    let eval_type = func.eval_type();
    if let Some(in_place) = in_place {
        let o = args[in_place as usize].num_type();
        assert!(o == eval_type, "output has different type than FlowFunc, expected {eval_type:?} got {o:?}");
    }

    let output = match in_place {
        Some(i) => None,
        None => match eval_type {
            NumType::F32 => Some(AnyGpuTensor::F32(GpuTensor::empty(shape))),
            NumType::I32 => Some(AnyGpuTensor::I32(GpuTensor::empty(shape))),
            NumType::U32 => Some(AnyGpuTensor::U32(GpuTensor::empty(shape))),
        },
    };

    // compile/cache shader
    let ctx = WgpuContext::get();
    let pipeline = ctx.pipelines.get(
        PipelineType::FunctionElementwise {
            func: func,
            in_place_arg: in_place,
        },
        ctx.wg_1d(),
    );

    // run shader
    let mut entries = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        entries.push(BindGroupEntry {
            binding: i as u32,
            resource: arg.raw().as_entire_binding(),
        });
    }
    if let Some(output) = &output {
        entries.push(BindGroupEntry {
            binding: args.len() as u32,
            resource: output.as_ref().raw().as_entire_binding(),
        });
    }
    let workgroup_x = (count as f64 / ctx.wg_1d().x_size as f64).ceil() as u32;
    let commands = ctx.encode_workgroup(&pipeline, &entries, workgroup_x, 1);
    ctx.execute_commands(commands);

    output
}

#[cfg(test)]
mod tests {
    use crate::tensors::ActivationType;

    use super::*;
    use FlowFunc::*;
    use NumType::*;

    #[test]
    #[rustfmt::skip]
    fn test_function() {
        let tensor1 = GpuTensor::new(&[
            [1u32, 2, 3],
            [4, 5, 6],
        ]);
        let tensor2 = GpuTensor::new(&[
            [1u32, 2, 3],
            [1, 2, 3],
        ]);
        let mut tensor3 = GpuTensor::new(&[
            [10, 20, 30],
            [40, 50, 60],
        ]);

        let func = Argument(0, U32).cast_to(F32) * Argument(1, U32).cast_to(F32) + Argument(2, I32).cast_to(F32).activation(ActivationType::Sigmoid);
        op_inplace((&tensor1, &tensor2, &mut tensor3), func.cast_to(I32));

        println!("tensor3: {tensor3:?}");
    }
}
