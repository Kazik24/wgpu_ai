use std::{alloc::Layout, fmt::Display, hash::Hash, i32, mem::transmute, ops::*};

use super::{AnyGpuTensor, AnyGpuTensorRef, FlowFunc, GpuTensor, HashF32};
use half::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NumType {
    F32,
    F16,  //only supported in spirv-passthrough - emulate it
    BF16, // todo - implement
    I32,
    U32,
}

impl NumType {
    pub const ALL: [NumType; 3] = [NumType::F32, NumType::I32, NumType::U32];
    pub const fn gpu_type(self) -> &'static str {
        match self {
            NumType::F32 => "f32",
            NumType::I32 => "i32",
            NumType::U32 => "u32",
            NumType::F16 => unimplemented!(),
            NumType::BF16 => unimplemented!(),
        }
    }
    pub const fn gpu_layout(self) -> Layout {
        const SIZE_4_4: Layout = match Layout::from_size_align(4, 4) {
            Ok(l) => l,
            Err(_) => unreachable!(),
        };
        const SIZE_2_2: Layout = match Layout::from_size_align(2, 2) {
            Ok(l) => l,
            Err(_) => unreachable!(),
        };

        match self {
            NumType::F32 => SIZE_4_4,
            NumType::I32 => SIZE_4_4,
            NumType::U32 => SIZE_4_4,
            NumType::F16 => SIZE_2_2,
            NumType::BF16 => SIZE_2_2,
        }
    }
}

trait Sealed {}

#[allow(private_bounds)]
pub trait GpuNum: Sealed + std::fmt::Debug + Display + Clone + Copy + Send + Sync + 'static
where
    Self: From<AnyGpuNum>,
    AnyGpuNum: From<Self>,
    Self: PartialEq + PartialOrd,
    Self: Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign,
    Self: Mul<Output = Self> + MulAssign + Div<Output = Self> + DivAssign + Rem<Output = Self> + RemAssign,
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f32(f: f32) -> Self {
        Self::from_any(AnyGpuNum::F32(f))
    }
    fn from_i32(i: i32) -> Self {
        Self::from_any(AnyGpuNum::I32(i))
    }
    fn from_u32(u: u32) -> Self {
        Self::from_any(AnyGpuNum::U32(u))
    }
    fn as_f32(&self) -> f32 {
        self.as_any().into()
    }
    fn as_i32(&self) -> i32 {
        self.as_any().into()
    }
    fn as_u32(&self) -> u32 {
        self.as_any().into()
    }
    fn num_type() -> NumType;
    fn as_any(&self) -> AnyGpuNum;
    fn from_any(any: AnyGpuNum) -> Self;
}

#[derive(Debug, Clone, Copy)]
pub enum AnyGpuNum {
    F32(f32),
    I32(i32),
    U32(u32),
    F16(f16),
    BF16(bf16),
}

impl PartialEq for AnyGpuNum {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AnyGpuNum::F32(f1), AnyGpuNum::F32(f2)) => f1.to_bits() == f2.to_bits(),
            (AnyGpuNum::I32(i1), AnyGpuNum::I32(i2)) => i1 == i2,
            (AnyGpuNum::U32(u1), AnyGpuNum::U32(u2)) => u1 == u2,
            (AnyGpuNum::F16(f1), AnyGpuNum::F16(f2)) => f1.to_bits() == f2.to_bits(),
            (AnyGpuNum::BF16(b1), AnyGpuNum::BF16(b2)) => b1.to_bits() == b2.to_bits(),
            _ => false,
        }
    }
}
impl Hash for AnyGpuNum {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            AnyGpuNum::F32(f) => f.to_bits().hash(state),
            AnyGpuNum::I32(i) => i.hash(state),
            AnyGpuNum::U32(u) => u.hash(state),
            AnyGpuNum::F16(f) => f.to_bits().hash(state),
            AnyGpuNum::BF16(b) => b.to_bits().hash(state),
        }
    }
}

impl AnyGpuNum {
    pub fn flow_const(&self) -> FlowFunc {
        match self {
            AnyGpuNum::F32(f) => FlowFunc::ConstF32(HashF32(*f)),
            AnyGpuNum::I32(i) => FlowFunc::ConstI32(*i),
            AnyGpuNum::U32(u) => FlowFunc::ConstU32(*u),
            AnyGpuNum::F16(f) => FlowFunc::ConstF32(HashF32(f.to_f32())), //todo - implement f16
            AnyGpuNum::BF16(f) => FlowFunc::ConstF32(HashF32(f.to_f32())),
        }
    }
    pub fn num_type(&self) -> NumType {
        match self {
            AnyGpuNum::F32(_) => NumType::F32,
            AnyGpuNum::I32(_) => NumType::I32,
            AnyGpuNum::U32(_) => NumType::U32,
            AnyGpuNum::F16(_) => NumType::F16,
            AnyGpuNum::BF16(_) => NumType::BF16,
        }
    }
    pub fn as_f32(self) -> f32 {
        self.into()
    }
    pub fn as_i32(self) -> i32 {
        self.into()
    }
    pub fn as_u32(self) -> u32 {
        self.into()
    }
    pub fn as_f16(self) -> f16 {
        self.into()
    }
    pub fn as_bf16(self) -> bf16 {
        self.into()
    }

    #[inline]
    pub(super) fn try_downcast_tensor<T: GpuNum>(tensor: AnyGpuTensor) -> Result<GpuTensor<T>, AnyGpuTensor> {
        match tensor {
            AnyGpuTensor::F32(t) => try_downcast_any(t).map_err(AnyGpuTensor::F32),
            AnyGpuTensor::I32(t) => try_downcast_any(t).map_err(AnyGpuTensor::I32),
            AnyGpuTensor::U32(t) => try_downcast_any(t).map_err(AnyGpuTensor::U32),
        }
    }
    #[inline]
    pub(super) fn upcast_tensor<T: GpuNum>(tensor: GpuTensor<T>) -> AnyGpuTensor {
        match T::num_type() {
            NumType::F32 => AnyGpuTensor::F32(try_downcast_any(tensor).unwrap()),
            NumType::I32 => AnyGpuTensor::I32(try_downcast_any(tensor).unwrap()),
            NumType::U32 => AnyGpuTensor::U32(try_downcast_any(tensor).unwrap()),
            NumType::F16 => unimplemented!(),
            NumType::BF16 => unimplemented!(),
        }
    }
    #[inline]
    pub(super) fn upcast_tensor_ref<T: GpuNum>(tensor: &GpuTensor<T>) -> AnyGpuTensorRef {
        match T::num_type() {
            NumType::F32 => AnyGpuTensorRef::F32(try_downcast_any_ref(tensor).unwrap()),
            NumType::I32 => AnyGpuTensorRef::I32(try_downcast_any_ref(tensor).unwrap()),
            NumType::U32 => AnyGpuTensorRef::U32(try_downcast_any_ref(tensor).unwrap()),
            NumType::F16 => unimplemented!(),
            NumType::BF16 => unimplemented!(),
        }
    }
}

#[inline]
pub(super) fn try_downcast_any<T: 'static, K: 'static>(k: K) -> Result<T, K> {
    let mut k = Some(k);
    if let Some(k) = <dyn std::any::Any>::downcast_mut::<Option<T>>(&mut k) {
        Ok(k.take().unwrap())
    } else {
        Err(k.unwrap())
    }
}
#[inline]
pub(super) fn try_downcast_any_ref<T: 'static, K: 'static>(k: &K) -> Option<&T> {
    <dyn std::any::Any>::downcast_ref::<T>(k)
}
#[inline]
pub(super) fn try_downcast_any_mut<T: 'static, K: 'static>(k: &mut K) -> Option<&mut T> {
    <dyn std::any::Any>::downcast_mut::<T>(k)
}

impl<T: GpuNum> From<T> for AnyGpuNum {
    fn from(t: T) -> Self {
        t.as_any()
    }
}

impl From<AnyGpuNum> for f32 {
    fn from(any: AnyGpuNum) -> Self {
        match any {
            AnyGpuNum::F32(f) => f,
            AnyGpuNum::I32(i) => i as f32,
            AnyGpuNum::U32(u) => u as f32,
            AnyGpuNum::F16(f) => f.to_f32(),
            AnyGpuNum::BF16(b) => b.to_f32(),
        }
    }
}
impl From<AnyGpuNum> for i32 {
    fn from(any: AnyGpuNum) -> Self {
        match any {
            AnyGpuNum::F32(f) => f as i32,
            AnyGpuNum::I32(i) => i,
            AnyGpuNum::U32(u) => u as i32,
            AnyGpuNum::F16(f) => f.to_f32() as i32,
            AnyGpuNum::BF16(b) => b.to_f32() as i32,
        }
    }
}
impl From<AnyGpuNum> for u32 {
    fn from(any: AnyGpuNum) -> Self {
        match any {
            AnyGpuNum::F32(f) => f as u32,
            AnyGpuNum::I32(i) => i as u32,
            AnyGpuNum::U32(u) => u,
            AnyGpuNum::F16(f) => f.to_f32() as u32,
            AnyGpuNum::BF16(b) => b.to_f32() as u32,
        }
    }
}
impl From<AnyGpuNum> for f16 {
    fn from(any: AnyGpuNum) -> Self {
        match any {
            AnyGpuNum::F32(f) => f16::from_f32(f),
            AnyGpuNum::I32(i) => f16::from_f32(i as f32),
            AnyGpuNum::U32(u) => f16::from_f32(u as f32),
            AnyGpuNum::F16(f) => f,
            AnyGpuNum::BF16(b) => f16::from_f32(b.to_f32()),
        }
    }
}
impl From<AnyGpuNum> for bf16 {
    fn from(any: AnyGpuNum) -> Self {
        match any {
            AnyGpuNum::F32(f) => bf16::from_f32(f),
            AnyGpuNum::I32(i) => bf16::from_f32(i as f32),
            AnyGpuNum::U32(u) => bf16::from_f32(u as f32),
            AnyGpuNum::F16(f) => bf16::from_f32(f.to_f32()),
            AnyGpuNum::BF16(b) => b,
        }
    }
}

macro_rules! impl_gpu_num {
    ($typ:ty, $type_name:ident, $zero:expr, $one:expr) => {
        impl Sealed for $typ {}
        impl GpuNum for $typ {
            fn zero() -> Self {
                $zero
            }
            fn one() -> Self {
                $one
            }
            fn as_any(&self) -> AnyGpuNum {
                AnyGpuNum::$type_name(*self)
            }
            fn num_type() -> NumType {
                NumType::$type_name
            }
            fn from_any(any: AnyGpuNum) -> Self {
                any.into()
            }
        }
    };
}

impl_gpu_num!(f32, F32, 0.0, 1.0);
impl_gpu_num!(i32, I32, 0, 1);
impl_gpu_num!(u32, U32, 0, 1);
impl_gpu_num!(f16, F16, f16::from_f32(0.0), f16::from_f32(1.0));
impl_gpu_num!(bf16, BF16, bf16::from_f32(0.0), bf16::from_f32(1.0));

pub trait AppendTuple {
    type Push<T>: AppendTuple;

    fn append<T>(self, value: T) -> Self::Push<T>;
}

macro_rules! impl_append_tuple {
    ($($gen:ident),*) => {
        impl<$($gen),*> AppendTuple for ($($gen,)*) {
            type Push<T> = ($($gen),* , T);

            #[allow(non_snake_case)]
            fn append<T>(self, value: T) -> Self::Push<T> {
                let ($($gen),*) = self;
                ($($gen),* ,value)
            }
        }
    };
}
//first and second sequence
impl AppendTuple for () {
    type Push<T> = (T,);
    fn append<T>(self, value: T) -> Self::Push<T> {
        (value,)
    }
}
impl<A1> AppendTuple for (A1,) {
    type Push<T> = (A1, T);
    fn append<T>(self, value: T) -> Self::Push<T> {
        let (a1,) = self;
        (a1, value)
    }
}
// ending sequence, max size 32
#[rustfmt::skip]
impl<A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27 ,A28 ,A29 ,A30, A31, A32> AppendTuple for (A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27 ,A28 ,A29 ,A30, A31, A32) {
    type Push<T> = (A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27 ,A28 ,A29 ,A30, A31, A32);
    fn append<T>(self, value: T) -> Self::Push<A3> {
        self
    }
}

impl_append_tuple!(A1, A2);
impl_append_tuple!(A1, A2, A3);
impl_append_tuple!(A1, A2, A3, A4);
impl_append_tuple!(A1, A2, A3, A4, A5);
impl_append_tuple!(A1, A2, A3, A4, A5, A6);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20);
impl_append_tuple!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21);
impl_append_tuple!(
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22
);
impl_append_tuple!(
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23
);
impl_append_tuple!(
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24
);
impl_append_tuple!(
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25
);
impl_append_tuple!(
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26
);
impl_append_tuple!(
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27
);
impl_append_tuple!(
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28
);
impl_append_tuple!(
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29
);
impl_append_tuple!(
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30
);
impl_append_tuple!(
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31
);
