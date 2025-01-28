use crate::tensors::{CpuTensor, GpuTensor, Tensor};

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
    weights_query: CpuTensor<f32>,  //wq
    weights_key: CpuTensor<f32>,    //wk
    weights_value: CpuTensor<f32>,  //wv
    weights_output: CpuTensor<f32>, //wo

    rms_norm_epsilon: f32,
    rms_norm_add_unit_offset: bool,        // always false for Llama, true for Gemma
    weights_rms_attention: CpuTensor<f32>, //w_rms_att [1, token_dim]

    // FFN
    mlp_gate_proj: CpuTensor<f32>, //w1
    mlp_down_proj: CpuTensor<f32>, // w2
    mlp_up_proj: CpuTensor<f32>,   // w3

                                   //  w_rms_post_att: Tensor<'a>,

                                   //  w_rms_pre_ffn: Option<Tensor<'a>>,
                                   //  w_rms_post_ffn: Option<Tensor<'a>>,

                                   //  w_rms_final: Tensor<'a>,
}

impl TransformerLayer {
    // input tensor shape: [1, token_dim], where 1 is a token count in current residual stream
    pub fn forward(&self, x: &mut CpuTensor<f32>, token_pos: u32) {
        x.rmsnorm(&self.weights_rms_attention, self.rms_norm_epsilon, self.rms_norm_add_unit_offset);
    }
}
