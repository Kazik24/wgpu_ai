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
