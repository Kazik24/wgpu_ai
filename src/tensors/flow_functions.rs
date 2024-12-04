use core::panic;
use std::{any::Any, collections::HashSet, fmt::Display, ops::*};

use super::{ActivationType, AnyGpuNum, GpuNum, GpuTensor, NumType};

// webgpu builtin functions https://webgpufundamentals.org/webgpu/lessons/webgpu-wgsl-function-reference.html
#[allow(private_interfaces)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FlowFunc {
    // values
    Argument(u8, NumType),
    ConstF32(HashF32), //this is why it doesnt implement eq and hash
    ConstI32(i32),
    ConstU32(u32),
    // casts
    CastTo(Box<Self>, NumType),
    // birary ops
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
    Pow(Box<Self>, Box<Self>),
    Min(Box<Self>, Box<Self>),
    Max(Box<Self>, Box<Self>),
    // unary ops
    Neg(Box<Self>),
    Abs(Box<Self>),
    Exp(Box<Self>),
    Log(Box<Self>),
    Sqrt(Box<Self>),
    Tanh(Box<Self>),

    Sigmoid(Box<Self>),
    ReLU(Box<Self>),
    SiLU(Box<Self>),
    GeLU(Box<Self>),
}

impl FlowFunc {
    const MAX_DEPTH: u8 = 255;

    // *** UTILITY ***
    pub fn new_const(val: impl GpuNum) -> Self {
        val.as_any().flow_const()
    }

    pub fn cast_to(mut self, num_type: NumType) -> Self {
        if let FlowFunc::CastTo(_, ty) = &mut self {
            *ty = num_type;
            return self;
        }
        FlowFunc::CastTo(Box::new(self), num_type)
    }

    pub fn optimize_cast_to(mut self, num_type: NumType) -> Self {
        if let FlowFunc::CastTo(_, ty) = &mut self {
            *ty = num_type;
            return self;
        }
        let ty = self.eval_type();
        if ty == num_type {
            return self;
        }
        FlowFunc::CastTo(Box::new(self), num_type)
    }

    pub fn activation(self, act: ActivationType) -> Self {
        match act {
            ActivationType::Sigmoid => FlowFunc::Sigmoid(Box::new(self)),
            ActivationType::ReLU => FlowFunc::ReLU(Box::new(self)),
            ActivationType::SiLU => FlowFunc::SiLU(Box::new(self)),
            ActivationType::GeLU => FlowFunc::GeLU(Box::new(self)),
        }
    }

    pub fn eval(&self, arg: impl GpuNum) -> AnyGpuNum {
        self.eval_any(&[arg.as_any()])
    }
    pub fn eval2(&self, arg1: impl GpuNum, arg2: impl GpuNum) -> AnyGpuNum {
        self.eval_any(&[arg1.as_any(), arg2.as_any()])
    }
    pub fn eval3(&self, arg1: impl GpuNum, arg2: impl GpuNum, arg3: impl GpuNum) -> AnyGpuNum {
        self.eval_any(&[arg1.as_any(), arg2.as_any(), arg3.as_any()])
    }
    pub fn eval4(&self, arg1: impl GpuNum, arg2: impl GpuNum, arg3: impl GpuNum, arg4: impl GpuNum) -> AnyGpuNum {
        self.eval_any(&[arg1.as_any(), arg2.as_any(), arg3.as_any(), arg4.as_any()])
    }

    // *** EVALUATION ***
    pub fn eval_any(&self, args: &[AnyGpuNum]) -> AnyGpuNum {
        let args = args.iter().map(|arg| arg.flow_const()).collect::<Vec<_>>();
        let mut func = self.clone();
        func.visit_args_replace(&args);
        match func.eval_const(Self::MAX_DEPTH) {
            Self::ConstF32(f) => AnyGpuNum::F32(f.0),
            Self::ConstI32(i) => AnyGpuNum::I32(i),
            Self::ConstU32(u) => AnyGpuNum::U32(u),
            _ => panic!("eval error, not constant folded"),
        }
    }

    fn visit_args_replace(&mut self, args: &[Self]) {
        use FlowFunc::*;
        match self {
            Argument(index, ty) => {
                let a = args.get(*index as usize).expect("bad argument index");
                assert_eq!(a.eval_type(), *ty, "bad argument type");
                *self = a.clone();
            }
            CastTo(a, ty) => a.visit_args_replace(args),
            ConstF32(_) | ConstI32(_) | ConstU32(_) => {}
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) => _ = (a.visit_args_replace(args), b.visit_args_replace(args)),
            Rem(a, b) => _ = (a.visit_args_replace(args), b.visit_args_replace(args)),
            Pow(a, b) | Min(a, b) | Max(a, b) => _ = (a.visit_args_replace(args), b.visit_args_replace(args)),
            Neg(a) | Abs(a) | Exp(a) | Log(a) | Sqrt(a) | Tanh(a) => a.visit_args_replace(args),
            Sigmoid(a) | ReLU(a) | GeLU(a) | SiLU(a) => a.visit_args_replace(args),
        }
    }

    pub const fn is_const(&self) -> bool {
        matches!(self, Self::ConstF32(_) | Self::ConstI32(_) | Self::ConstU32(_))
    }
    pub const fn const_as_f32(&self) -> f32 {
        match self {
            Self::ConstF32(f) => f.0,
            Self::ConstI32(i) => *i as f32,
            Self::ConstU32(u) => *u as f32,
            _ => panic!("not a constant"),
        }
    }
    pub const fn const_as_i32(&self) -> i32 {
        match self {
            Self::ConstF32(f) => f.0 as i32,
            Self::ConstI32(i) => *i,
            Self::ConstU32(u) => *u as i32,
            _ => panic!("not a constant"),
        }
    }
    pub const fn const_as_u32(&self) -> u32 {
        match self {
            Self::ConstF32(f) => f.0 as u32,
            Self::ConstI32(i) => *i as u32,
            Self::ConstU32(u) => *u,
            _ => panic!("not a constant"),
        }
    }
    pub const fn cast_const(&self, num_type: NumType) -> Self {
        match num_type {
            NumType::F32 => Self::ConstF32(HashF32(self.const_as_f32())),
            NumType::I32 => Self::ConstI32(self.const_as_i32()),
            NumType::U32 => Self::ConstU32(self.const_as_u32()),
        }
    }

    pub fn eval_type(&self) -> NumType {
        use FlowFunc::*;
        match self {
            ConstF32(_) => NumType::F32,
            ConstI32(_) => NumType::I32,
            ConstU32(_) => NumType::U32,
            CastTo(_, t) | Argument(_, t) => *t,
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) => Self::type_match(a.eval_type(), b.eval_type()),
            Rem(a, b) => Self::type_match(a.eval_type(), b.eval_type()),
            Pow(a, b) => Self::type_match(Self::type_match(a.eval_type(), b.eval_type()), NumType::F32),
            Min(a, b) | Max(a, b) => Self::type_match(a.eval_type(), b.eval_type()),
            ReLU(a) => a.eval_type(),
            Abs(a) => a.eval_type(),
            Neg(a) => {
                let t = a.eval_type();
                assert!(matches!(t, NumType::F32 | NumType::I32), "neg only supports f32 or i32");
                t
            }
            Exp(a) | Log(a) | Sqrt(a) | Tanh(a) => Self::type_match(a.eval_type(), NumType::F32),
            Sigmoid(a) | GeLU(a) | SiLU(a) => {
                let t = a.eval_type();
                assert!(matches!(t, NumType::F32), "function only supports f32");
                t
            }
        }
    }

    fn type_match(t1: NumType, t2: NumType) -> NumType {
        assert!(t1 == t2, "types do not match");
        t1
    }

    // evaluates value only if there is no arguments, return Const* value
    fn eval_const(&self, depth: u8) -> Self {
        if depth == 0 {
            panic!("Recursion depth reached");
        }
        use FlowFunc::*;
        fn fc(v: f32) -> FlowFunc {
            FlowFunc::ConstF32(HashF32(v))
        }
        fn ic(v: i32) -> FlowFunc {
            FlowFunc::ConstI32(v)
        }
        fn uc(v: u32) -> FlowFunc {
            FlowFunc::ConstU32(v)
        }
        match self {
            v @ ConstF32(_) | v @ ConstI32(_) | v @ ConstU32(_) => v.clone(),
            CastTo(v, t) => v.eval_const(depth.saturating_sub(1)).cast_const(*t),
            Argument(i, _) => panic!("not a constant, contains argument with index {i}"),
            Add(a, b) => {
                let a = a.eval_const(depth.saturating_sub(1));
                let b = b.eval_const(depth.saturating_sub(1));
                Self::bin_op_same_types(&a, &b, "add", |a, b| fc(a + b), |a, b| ic(a + b), |a, b| uc(a + b))
            }
            Sub(a, b) => {
                let a = a.eval_const(depth.saturating_sub(1));
                let b = b.eval_const(depth.saturating_sub(1));
                Self::bin_op_same_types(&a, &b, "sub", |a, b| fc(a - b), |a, b| ic(a - b), |a, b| uc(a - b))
            }
            Mul(a, b) => {
                let a = a.eval_const(depth.saturating_sub(1));
                let b = b.eval_const(depth.saturating_sub(1));
                Self::bin_op_same_types(&a, &b, "mul", |a, b| fc(a * b), |a, b| ic(a * b), |a, b| uc(a * b))
            }
            Div(a, b) => {
                let a = a.eval_const(depth.saturating_sub(1));
                let b = b.eval_const(depth.saturating_sub(1));
                Self::bin_op_same_types(&a, &b, "div", |a, b| fc(a / b), |a, b| ic(a / b), |a, b| uc(a / b))
            }
            Rem(a, b) => {
                let a = a.eval_const(depth.saturating_sub(1));
                let b = b.eval_const(depth.saturating_sub(1));
                Self::bin_op_same_types(&a, &b, "mod", |a, b| fc(a % b), |a, b| ic(a % b), |a, b| uc(a % b))
            }
            Pow(a, b) => match (a.eval_const(depth.saturating_sub(1)), b.eval_const(depth.saturating_sub(1))) {
                (ConstF32(a), ConstF32(b)) => ConstF32(HashF32(a.0.powf(b.0))),
                _ => panic!("types of pow do not match, must be F32"),
            },
            Min(a, b) => {
                let a = a.eval_const(depth.saturating_sub(1));
                let b = b.eval_const(depth.saturating_sub(1));
                Self::bin_op_same_types(&a, &b, "min", |a, b| fc(a.min(b)), |a, b| ic(a.min(b)), |a, b| uc(a.min(b)))
            }
            Max(a, b) => {
                let a = a.eval_const(depth.saturating_sub(1));
                let b = b.eval_const(depth.saturating_sub(1));
                Self::bin_op_same_types(&a, &b, "max", |a, b| fc(a.max(b)), |a, b| ic(a.max(b)), |a, b| uc(a.max(b)))
            }
            Neg(a) => match a.eval_const(depth.saturating_sub(1)) {
                ConstF32(a) => fc(-a.0),
                ConstI32(a) => ic(-a),
                _ => panic!("types of neg do not match, must be F32 or I32"),
            },
            Abs(a) => match a.eval_const(depth.saturating_sub(1)) {
                ConstF32(a) => fc(a.0.abs()),
                ConstI32(a) => ic(a.abs()),
                ConstU32(a) => uc(a),
                _ => panic!("types of relu do not match, must be F32 or I32 or U32"),
            },
            Exp(a) => match a.eval_const(depth.saturating_sub(1)) {
                ConstF32(a) => fc(a.0.exp()),
                _ => panic!("types of exp do not match, must be F32"),
            },
            Log(a) => match a.eval_const(depth.saturating_sub(1)) {
                ConstF32(a) => fc(a.0.ln()),
                _ => panic!("types of log do not match, must be F32"),
            },
            Sqrt(a) => match a.eval_const(depth.saturating_sub(1)) {
                ConstF32(a) => fc(a.0.sqrt()),
                _ => panic!("types of sqrt do not match, must be F32"),
            },
            Tanh(a) => match a.eval_const(depth.saturating_sub(1)) {
                ConstF32(a) => fc(a.0.tanh()),
                _ => panic!("types of tanh do not match, must be F32"),
            },
            Sigmoid(a) => match a.eval_const(depth.saturating_sub(1)) {
                ConstF32(a) => fc(1.0 / (1.0 + (-a.0).exp())),
                _ => panic!("types of sigmoid do not match, must be F32"),
            },
            ReLU(a) => match a.eval_const(depth.saturating_sub(1)) {
                ConstF32(a) => fc(a.0.max(0.0)),
                ConstI32(a) => ic(a.max(0)),
                ConstU32(a) => uc(a.max(0)),
                _ => panic!("types of relu do not match, must be F32 or I32 or U32"),
            },
            SiLU(a) => match a.eval_const(depth.saturating_sub(1)) {
                ConstF32(a) => fc(a.0 * (1.0 / (1.0 + (-a.0).exp()))),
                _ => panic!("types of silu do not match, must be F32"),
            },
            GeLU(a) => match a.eval_const(depth.saturating_sub(1)) {
                ConstF32(HashF32(a)) => fc(a * (0.5 * (1.0 + (0.7978845608028654 * (a + 0.044715 * a * a * a)).tanh()))),
                _ => panic!("types of gelu do not match, must be F32"),
            },
        }
    }

    pub fn get_arguments(&self) -> Result<Vec<NumType>, String> {
        let mut args = HashSet::new();
        self.fetch_arguments(&mut args);
        let mut result = Vec::new();
        for i in 0..=u8::MAX {
            let mut found = None;
            for ty in NumType::ALL {
                if args.remove(&(i, ty)) {
                    match found {
                        None => found = Some(ty),
                        Some(t) => return Err(format!("argument {i} has multiple types {t:?} and {ty:?}")),
                    }
                }
            }
            match found {
                Some(t) => result.push(t),
                None if args.is_empty() => return Ok(result),
                None => return Err(format!("non continuous arguments {:?}", args.iter().map(|(i, _)| *i).collect::<Vec<_>>())),
            }
        }
        Ok(result)
    }

    fn fetch_arguments(&self, args: &mut HashSet<(u8, NumType)>) {
        use FlowFunc::*;
        match self {
            ConstF32(_) | ConstI32(_) | ConstU32(_) => {}
            Argument(index, t) => _ = args.insert((*index, *t)),
            CastTo(a, _) => a.fetch_arguments(args),
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) => _ = (a.fetch_arguments(args), b.fetch_arguments(args)),
            Rem(a, b) => _ = (a.fetch_arguments(args), b.fetch_arguments(args)),
            Pow(a, b) | Min(a, b) | Max(a, b) => _ = (a.fetch_arguments(args), b.fetch_arguments(args)),
            Neg(a) | Abs(a) | Exp(a) | Log(a) | Sqrt(a) | Tanh(a) => a.fetch_arguments(args),
            Sigmoid(a) | ReLU(a) | GeLU(a) | SiLU(a) => a.fetch_arguments(args),
        }
    }

    fn bin_op_same_types(
        a: &Self, b: &Self, op: &str, f: impl FnOnce(f32, f32) -> Self, i: impl FnOnce(i32, i32) -> Self, u: impl FnOnce(u32, u32) -> Self,
    ) -> Self {
        match (a, b) {
            (Self::ConstF32(a), Self::ConstF32(b)) => f(a.0, b.0),
            (Self::ConstI32(a), Self::ConstI32(b)) => i(*a, *b),
            (Self::ConstU32(a), Self::ConstU32(b)) => u(*a, *b),
            _ => panic!("types of {op} do not match, found lhs: {a:?}, rhs: {b:?}"),
        }
    }

    pub(crate) fn compile(&self, var_names: &[&str]) -> String {
        use FlowFunc::*;
        self.eval_type(); //panic if types don't match
        match self {
            Argument(index, _) => var_names[*index as usize].to_string(),
            ConstF32(f) => f.0.to_string(),
            ConstI32(i) => i.to_string(),
            ConstU32(u) => u.to_string(),
            CastTo(f, ty) => format!("{}({})", ty.gpu_type(), f.compile(var_names)),
            Add(a, b) => format!("({} + {})", a.compile(var_names), b.compile(var_names)),
            Sub(a, b) => format!("({} - {})", a.compile(var_names), b.compile(var_names)),
            Mul(a, b) => format!("({} * {})", a.compile(var_names), b.compile(var_names)),
            Div(a, b) => format!("({} / {})", a.compile(var_names), b.compile(var_names)),
            Rem(a, b) => format!("({} % {})", a.compile(var_names), b.compile(var_names)),
            Pow(a, b) => format!("pow({}, {})", a.compile(var_names), b.compile(var_names)),
            Min(a, b) => format!("min({}, {})", a.compile(var_names), b.compile(var_names)),
            Max(a, b) => format!("max({}, {})", a.compile(var_names), b.compile(var_names)),
            Neg(a) => format!("(-{})", a.compile(var_names)),
            Abs(a) => format!("abs({})", a.compile(var_names)),
            Exp(a) => format!("exp({})", a.compile(var_names)),
            Log(a) => format!("log({})", a.compile(var_names)),
            Sqrt(a) => format!("sqrt({})", a.compile(var_names)),
            Tanh(a) => format!("tanh({})", a.compile(var_names)),
            Sigmoid(a) => format!("sigmoid_activation({})", a.compile(var_names)),
            ReLU(a) => format!("max({}, {}(0))", a.compile(var_names), a.eval_type().gpu_type()),
            SiLU(a) => format!("silu_activation({})", a.compile(var_names)),
            GeLU(a) => format!("gelu_activation({})", a.compile(var_names)),
        }
    }
}

#[derive(Copy, Clone)]
pub(super) struct HashF32(pub f32);
impl std::hash::Hash for HashF32 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}
impl Eq for HashF32 {}
impl PartialEq for HashF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl std::fmt::Debug for HashF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <f32 as std::fmt::Debug>::fmt(&self.0, f)
    }
}

macro_rules! impl_bin_op {
    ($op:ident, $func:ident, $enum_val:ident) => {
        impl $op for FlowFunc {
            type Output = FlowFunc;
            fn $func(self, rhs: Self) -> Self::Output {
                FlowFunc::$enum_val(Box::new(self), Box::new(rhs))
            }
        }
        impl $op for &FlowFunc {
            type Output = FlowFunc;
            fn $func(self, rhs: Self) -> Self::Output {
                FlowFunc::$enum_val(Box::new(self.clone()), Box::new(rhs.clone()))
            }
        }
        impl $op<FlowFunc> for &FlowFunc {
            type Output = FlowFunc;
            fn $func(self, rhs: FlowFunc) -> Self::Output {
                FlowFunc::$enum_val(Box::new(self.clone()), Box::new(rhs))
            }
        }
        impl $op<&FlowFunc> for FlowFunc {
            type Output = FlowFunc;
            fn $func(self, rhs: &FlowFunc) -> Self::Output {
                FlowFunc::$enum_val(Box::new(self), Box::new(rhs.clone()))
            }
        }
    };
}

impl_bin_op!(Add, add, Add);
impl_bin_op!(Sub, sub, Sub);
impl_bin_op!(Mul, mul, Mul);
impl_bin_op!(Div, div, Div);
impl_bin_op!(Rem, rem, Rem);

#[cfg(test)]
mod tests {
    use super::*;

    use FlowFunc::*;

    #[test]
    #[should_panic]
    fn test_basic_eval_error() {
        let func = Argument(0, NumType::F32) + Argument(1, NumType::U32) * Argument(2, NumType::F32);
        func.eval3(1.0, 2, 3.0).as_i32();
    }

    #[test]
    fn test_basic_eval() {
        let func = Argument(0, NumType::F32) + Argument(1, NumType::F32) * Argument(2, NumType::F32);
        assert_eq!(func.eval3(1.0, 2.0, 3.0).as_i32(), 7);
    }
}
