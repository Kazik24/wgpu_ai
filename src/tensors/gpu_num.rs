use std::fmt::Display;

use super::{FlowFunc, HashF32};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NumType {
    F32,
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
}

trait Sealed {}

#[allow(private_bounds)]
pub trait GpuNum: std::fmt::Debug + Display + Sealed + Clone + Copy + Send + Sync + 'static {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyGpuNum {
    F32(f32),
    I32(i32),
    U32(u32),
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
