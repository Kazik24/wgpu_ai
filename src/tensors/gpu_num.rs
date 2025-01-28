use std::{alloc::Layout, fmt::Display, hash::Hash, i32, mem::transmute, ops::*};

use super::{AnyGpuTensor, AnyGpuTensorRef, FlowFunc, GpuTensor, HashF32};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NumType {
    F32,
    // F16, //only supported in spirv-passthrough
    // BF16, // todo - implement
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
        }
    }
    pub const fn gpu_layout(self) -> Layout {
        const SIZE_4_4: Layout = match Layout::from_size_align(4, 4) {
            Ok(l) => l,
            Err(_) => unreachable!(),
        };

        match self {
            NumType::F32 => SIZE_4_4,
            NumType::I32 => SIZE_4_4,
            NumType::U32 => SIZE_4_4,
        }
    }
}

trait Sealed {}

#[allow(private_bounds)]
pub trait GpuNum: Sealed + std::fmt::Debug + Display + Clone + Copy + Send + Sync + 'static
where
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
}

impl PartialEq for AnyGpuNum {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AnyGpuNum::F32(f1), AnyGpuNum::F32(f2)) => f1.to_bits() == f2.to_bits(),
            (AnyGpuNum::I32(i1), AnyGpuNum::I32(i2)) => i1 == i2,
            (AnyGpuNum::U32(u1), AnyGpuNum::U32(u2)) => u1 == u2,
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
        }
    }
}

impl AnyGpuNum {
    pub fn flow_const(&self) -> FlowFunc {
        match self {
            AnyGpuNum::F32(f) => FlowFunc::ConstF32(HashF32(*f)),
            AnyGpuNum::I32(i) => FlowFunc::ConstI32(*i),
            AnyGpuNum::U32(u) => FlowFunc::ConstU32(*u),
        }
    }
    pub fn num_type(&self) -> NumType {
        match self {
            AnyGpuNum::F32(_) => NumType::F32,
            AnyGpuNum::I32(_) => NumType::I32,
            AnyGpuNum::U32(_) => NumType::U32,
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
        }
    }
    #[inline]
    pub(super) fn upcast_tensor_ref<T: GpuNum>(tensor: &GpuTensor<T>) -> AnyGpuTensorRef {
        match T::num_type() {
            NumType::F32 => AnyGpuTensorRef::F32(try_downcast_any_ref(tensor).unwrap()),
            NumType::I32 => AnyGpuTensorRef::I32(try_downcast_any_ref(tensor).unwrap()),
            NumType::U32 => AnyGpuTensorRef::U32(try_downcast_any_ref(tensor).unwrap()),
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
        Self::from_any(any)
    }
}
impl From<AnyGpuNum> for i32 {
    fn from(any: AnyGpuNum) -> Self {
        Self::from_any(any)
    }
}
impl From<AnyGpuNum> for u32 {
    fn from(any: AnyGpuNum) -> Self {
        Self::from_any(any)
    }
}

impl Sealed for f32 {}
impl Sealed for i32 {}
impl Sealed for u32 {}

impl GpuNum for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn as_any(&self) -> AnyGpuNum {
        AnyGpuNum::F32(*self)
    }
    fn num_type() -> NumType {
        NumType::F32
    }
    fn from_any(any: AnyGpuNum) -> Self {
        match any {
            AnyGpuNum::F32(f) => f,
            AnyGpuNum::I32(i) => i as f32,
            AnyGpuNum::U32(u) => u as f32,
        }
    }
}
impl GpuNum for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn as_any(&self) -> AnyGpuNum {
        AnyGpuNum::I32(*self)
    }
    fn num_type() -> NumType {
        NumType::I32
    }
    fn from_any(any: AnyGpuNum) -> Self {
        match any {
            AnyGpuNum::F32(f) => f as i32,
            AnyGpuNum::I32(i) => i,
            AnyGpuNum::U32(u) => u as i32,
        }
    }
}
impl GpuNum for u32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn as_any(&self) -> AnyGpuNum {
        AnyGpuNum::U32(*self)
    }
    fn num_type() -> NumType {
        NumType::U32
    }
    fn from_any(any: AnyGpuNum) -> Self {
        match any {
            AnyGpuNum::F32(f) => f as u32,
            AnyGpuNum::I32(i) => i as u32,
            AnyGpuNum::U32(u) => u,
        }
    }
}
