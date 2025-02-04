use super::{AppendTuple, GpuNum, Tensor};

pub struct ExecutionStream<Res: AppendTuple> {
    owned: Res,
    bindings: Vec<()>,
}

impl<Res: AppendTuple> ExecutionStream<Res> {
    pub fn bind_ref<'a, T: GpuNum>(self, tensor: &'a Tensor<T>) -> (ExecutionStream<Res::Push<&'a Tensor<T>>>, usize) {
        let index = self.bindings.len(); //todo

        let stream = ExecutionStream {
            owned: self.owned.append(tensor),
            bindings: self.bindings,
        };
        (stream, index)
    }

    pub fn bind_mut<'a, T: GpuNum>(self, tensor: &'a mut Tensor<T>) -> (ExecutionStream<Res::Push<&'a mut Tensor<T>>>, usize) {
        let index = self.bindings.len(); //todo

        let stream = ExecutionStream {
            owned: self.owned.append(tensor),
            bindings: self.bindings,
        };
        (stream, index)
    }
}
