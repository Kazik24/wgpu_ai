use std::collections::VecDeque;

use crate::tensors::{ActivationType, CpuTensor, GpuTensor, Tensor};

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

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct TransformerArgs {
    pub attn_rms_norm_epsilon: f32, // epsilon for rms norm operation at the start of each layer (before attention layer)
    pub mlp_rms_norm_epsilon: f32,  // epsilon for rms norm operation after attention before mlp (should be thesame as attn_rms_norm_epsilon)
    pub attn_head_size: usize,      // size of the vector comming out of the key/query projection
    pub attn_heads: usize,          // number of attention heads
    pub n_key_value_heads: usize,   // number of key/value heads

    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    head_size: u32,
    n_kv_heads: u32,
    pub vocab_size: u32,
    seq_len: u32,

    rope_theta: f32,
    group_size: u32,
    pub multimodal: bool,
}

pub struct LayerState {
    key_cache: VecDeque<Tensor<f32>>,
    value_cache: VecDeque<Tensor<f32>>,
}

struct TransformerLayer {
    args: TransformerArgs,

    // Attention
    weights_query: Tensor<f32>,  //wq
    weights_key: Tensor<f32>,    //wk
    weights_value: Tensor<f32>,  //wv
    weights_output: Tensor<f32>, //wo

    rms_norm_add_unit_offset: bool,     // always false for Llama, true for Gemma
    weights_rms_attention: Tensor<f32>, //w_rms_att [1, token_dim]

    // FFN
    weight_rms_mlp: Tensor<f32>, //w_rms_post_att [1, token_dim]
    mlp_gate_proj: Tensor<f32>,  //w1
    mlp_down_proj: Tensor<f32>,  // w2
    mlp_up_proj: Tensor<f32>,    // w3

                                 //  w_rms_pre_ffn: Option<Tensor<'a>>,
                                 //  w_rms_post_ffn: Option<Tensor<'a>>,

                                 //  w_rms_final: Tensor<'a>,
}

impl TransformerLayer {
    // todo description of llama: https://pub.towardsai.net/build-your-own-llama-3-architecture-from-scratch-using-pytorch-2ce1ecaa901c

    // input tensor shape: [1, token_dim], where 1 is a token count in current residual stream
    pub fn forward(&self, residual_stream: &mut Tensor<f32>, token_pos: u32) {
        let mut x = residual_stream.clone();

        // apply attention and produce delta change
        self.attention_block(&mut x, token_pos);
        // add delta to residual stream
        residual_stream.elementwise_add_assign(&x);
        x.copy_from(residual_stream); //update x to new residual stream

        // apply MLP and produce delta change
        self.mlp_block(&mut x);
        // add delta to residual stream
        residual_stream.elementwise_add_assign(&x);
    }

    fn attention_block(&self, x: &mut Tensor<f32>, token_pos: u32) {
        //apply normalization before doing anything else
        x.rmsnorm(&self.weights_rms_attention, self.args.attn_rms_norm_epsilon, self.rms_norm_add_unit_offset);

        let key_value_dim = self.args.attn_head_size * self.args.n_key_value_heads;
        let attention_dim = self.args.attn_head_size * self.args.attn_heads;
        let mut key = Tensor::<f32>::empty([1, key_value_dim]); // will be cached
        let mut value = Tensor::<f32>::empty([1, key_value_dim]); // will be cached
        let mut query = Tensor::<f32>::empty([1, attention_dim]);

        x.matrix_mul_assign(&self.weights_query, &mut query);
        x.matrix_mul_assign(&self.weights_key, &mut key);
        x.matrix_mul_assign(&self.weights_value, &mut value);
    }

    fn mlp_block(&self, x: &mut Tensor<f32>) {
        x.rmsnorm(&self.weight_rms_mlp, self.args.mlp_rms_norm_epsilon, self.rms_norm_add_unit_offset);

        //project up
        let mut hidden = x.matrix_mul(&self.mlp_up_proj);
        let gate = x.matrix_mul(&self.mlp_gate_proj);

        // gate/activation
        hidden.gated_activation_assign(ActivationType::SiLU, &gate);

        //project down
        hidden.matrix_mul_assign(&self.mlp_down_proj, x);
    }
}
