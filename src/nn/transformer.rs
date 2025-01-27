use crate::tensors::{GpuTensor, Tensor};

pub struct Transformer {
    output_logits: Vec<f32>,
}

impl Transformer {
    /// runs model forward with filling any key/val cache, token_index is the index of the token in embedings table,
    /// token_pos is the position of the token in the context for positional encoding
    /// returns vector of logits to sample next token from
    pub fn forward(&mut self, token_index: u32, token_pos: u32) -> &mut [f32] {
        todo!();

        &mut self.output_logits
    }
}

struct TransformerLayer {
    // Attention
    weights_query: GpuTensor<f32>,  //wq
    weights_key: GpuTensor<f32>,    //wk
    weights_value: GpuTensor<f32>,  //wv
    weights_output: GpuTensor<f32>, //wo

    weights_rms_attention: GpuTensor<f32>, //w_rms_att

    // FFN
    mlp_gate_proj: GpuTensor<f32>, //w1
    mlp_down_proj: GpuTensor<f32>, // w2
    mlp_up_proj: GpuTensor<f32>,   // w3

                                   //  w_rms_post_att: Tensor<'a>,

                                   //  w_rms_pre_ffn: Option<Tensor<'a>>,
                                   //  w_rms_post_ffn: Option<Tensor<'a>>,

                                   //  w_rms_final: Tensor<'a>,
}
