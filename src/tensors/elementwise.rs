use super::{GpuNum, GpuTensor, NumType};

pub trait ElementwiseOp<Output = ()> {
    const RETURN_ARG: Option<u8>;
    type ReturnType;
}

macro_rules! impl_elementwise_op {
    (@ mut $gen:ident) => {&mut GpuTensor<$gen>};
    (@ const $gen:ident) => {&GpuTensor<$gen>};
    ($($mutable:ident $gen:ident),* : $arg:expr $(, $out:ty)?) => {
        impl<$($gen: GpuNum),*> ElementwiseOp<($(GpuTensor<$out>)?)> for ($(impl_elementwise_op!(@ $mutable $gen)),*) {
            const RETURN_ARG: Option<u8> = $arg;
            type ReturnType = ($(GpuTensor<$out>)?);
        }
    };
}

// producing new tensor
impl_elementwise_op!(const A, const B : None);
impl_elementwise_op!(const A, const B, const C : None);
impl_elementwise_op!(const A, const B, const C, const D : None);
impl_elementwise_op!(const A, const B, const C, const D, const E : None);
impl_elementwise_op!(const A, const B, const C, const D, const E, const F : None);
impl_elementwise_op!(const A, const B, const C, const D, const E, const F, const G : None);
impl_elementwise_op!(const A, const B, const C, const D, const E, const F, const G, const H : None);

// reusing existing tensor
impl_elementwise_op!(mut A, const B : Some(0), f32);
impl_elementwise_op!(mut A, const B : Some(0), u32);
impl_elementwise_op!(mut A, const B : Some(0), i32);
impl_elementwise_op!(const A, mut B : Some(1), f32);
impl_elementwise_op!(const A, mut B : Some(1), u32);
impl_elementwise_op!(const A, mut B : Some(1), i32);

impl_elementwise_op!(mut A, const B, const C : Some(0), f32);
impl_elementwise_op!(mut A, const B, const C : Some(0), u32);
impl_elementwise_op!(mut A, const B, const C : Some(0), i32);
impl_elementwise_op!(const A, mut B, const C : Some(1), f32);
impl_elementwise_op!(const A, mut B, const C : Some(1), u32);
impl_elementwise_op!(const A, mut B, const C : Some(1), i32);
impl_elementwise_op!(const A, const B, mut C : Some(2), f32);
impl_elementwise_op!(const A, const B, mut C : Some(2), u32);
impl_elementwise_op!(const A, const B, mut C : Some(2), i32);

impl_elementwise_op!(mut A, const B, const C, const D : Some(0), f32);
impl_elementwise_op!(mut A, const B, const C, const D : Some(0), u32);
impl_elementwise_op!(mut A, const B, const C, const D : Some(0), i32);
impl_elementwise_op!(const A, mut B, const C, const D : Some(1), f32);
impl_elementwise_op!(const A, mut B, const C, const D : Some(1), u32);
impl_elementwise_op!(const A, mut B, const C, const D : Some(1), i32);
impl_elementwise_op!(const A, const B, mut C, const D : Some(2), f32);
impl_elementwise_op!(const A, const B, mut C, const D : Some(2), u32);
impl_elementwise_op!(const A, const B, mut C, const D : Some(2), i32);
impl_elementwise_op!(const A, const B, const C, mut D : Some(3), f32);
impl_elementwise_op!(const A, const B, const C, mut D : Some(3), u32);
impl_elementwise_op!(const A, const B, const C, mut D : Some(3), i32);

//todo hoe to handle Output
pub fn op<O: ElementwiseOp>(args: O) -> O::ReturnType {
    todo!()
}
