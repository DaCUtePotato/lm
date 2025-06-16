
use crate::*;

pub struct TransformerModel {
    pub embedding: Embedding,
    pub transformer: TransformerBlock,
    pub output: OutputProjection,
}

impl TransformerModel {
    pub fn new(vocab_size: usize, emb_dim: usize, max_len: usize, num_heads: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, emb_dim, max_len),
            transformer: TransformerBlock::new(emb_dim, num_heads, emb_dim * 4),
            output: OutputProjection::new(emb_dim, vocab_size),
        }
    }

    pub fn forward(&mut self, input: &[usize]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let embedded = self.embedding.forward(input);
        let transformed = self.transformer.forward(&embedded);
        let logits = self.output.forward(transformed.clone());
        (embedded, transformed, logits)
    }

    pub fn backward(
        &mut self,
        input: &[usize],
        grad_logits: &[Vec<f32>],
        transformed: &[Vec<f32>],
        embedded: &[Vec<f32>],
    ) {
        let grad_transformer_out = self
            .output
            .backward(&grad_logits.to_vec(), &transformed.to_vec());
        let grad_embedded = self.transformer.backward(&grad_transformer_out);
        self.embedding.backward(&grad_embedded, input);
    }

    pub fn step(&mut self, lr: f32) {
        self.embedding.step(lr);
        self.transformer.step(lr);
        self.output.step(lr);
    }
}
