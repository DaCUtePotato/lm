
use crate::*; // Bring everything from the crate scope, so this file can access Embedding, TransformerBlock, etc.

pub struct TransformerModel {
    pub embedding: Embedding, // Converts token indices into dense vectors
    pub transformer: TransformerBlock, // The core transformer block: attention + feed-forward + norm layers
    pub output: OutputProjection,      // Final linear projection to vocab size (logits)
}

impl TransformerModel {
    // Constructor: builds a new TransformerModel with given vocab size, embedding dimension, max sequence length, and number of attention heads.
    pub fn new(vocab_size: usize, emb_dim: usize, max_len: usize, num_heads: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, emb_dim, max_len),
            transformer: TransformerBlock::new(emb_dim, num_heads, emb_dim * 4), // FFN hidden dim is 4 times embedding dim (typical)
            output: OutputProjection::new(emb_dim, vocab_size),
        }
    }

    // Forward pass through the whole model: input token indices -> embeddings -> transformer -> logits
    pub fn forward(&mut self, input: &[usize]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let embedded = self.embedding.forward(input); // Token embedding: discrete → dense vectors
        let transformed = self.transformer.forward(&embedded); // Transformer layers apply self-attention + feed-forward
        let logits = self.output.forward(transformed.clone()); // Project transformer outputs to vocab logits for prediction
        (embedded, transformed, logits) // Return all intermediate states for backprop
    }

    // Backward pass to propagate gradients and update internal gradients for parameters
    pub fn backward(
        &mut self,
        input: &[usize],
        grad_logits: &[Vec<f32>],
        transformed: &[Vec<f32>],
        embedded: &[Vec<f32>],
    ) {
        // Backprop through output projection layer first, getting gradients wrt transformer output
        let grad_transformer_out = self
            .output
            .backward(&grad_logits.to_vec(), &transformed.to_vec());

        // Backprop through transformer block, getting gradients wrt embeddings
        let grad_embedded = self.transformer.backward(&grad_transformer_out);

        // Backprop through embeddings (embedding table lookup), adjusting embeddings
        self.embedding.backward(&grad_embedded, input);
    }

    // Perform an optimization step on all components using the given learning rate
    pub fn step(&mut self, lr: f32) {
        self.embedding.step(lr);
        self.transformer.step(lr);
        self.output.step(lr);
    }
}
