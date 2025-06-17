
use super::attention::{scaled_dot_product_attention, MultiHeadAttention};
use super::embedding;
use super::feed_forward::FeedForward;
use super::layer_norm::{self, LayerNorm};

/// A single Transformer block:
/// Consists of multi-head self-attention + feed-forward, each with layer norm and skip connections.
pub struct TransformerBlock {
    mha: MultiHeadAttention, // Multi-head self-attention layer
    norm1: LayerNorm,        // LayerNorm after attention + residual
    ffn: FeedForward,        // Position-wise feed-forward network
    norm2: LayerNorm,        // LayerNorm after FFN + residual

    // Cached inputs/outputs for backward pass
    input: Vec<Vec<f32>>,
    attn_output: Vec<Vec<f32>>,
    ffn_output: Vec<Vec<f32>>,
}

impl TransformerBlock {
    /// Creates a new transformer block with given dimensions
    pub fn new(embed_dim: usize, num_heads: usize, ff_hidden_dim: usize) -> Self {
        Self {
            mha: MultiHeadAttention::new(embed_dim, num_heads),
            norm1: LayerNorm::new(embed_dim),
            ffn: FeedForward::new(embed_dim, ff_hidden_dim),
            norm2: LayerNorm::new(embed_dim),

            // Initialise caches for forward/backward
            input: vec![],
            attn_output: vec![],
            ffn_output: vec![],
        }
    }

    /// Forward pass:
    /// x -> MHA + skip -> norm1 -> FFN + skip -> norm2
    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.input = x.to_vec(); // Store input for backward

        // === Multi-head attention with skip connection ===
        let attn = self.mha.forward(x, true); // Compute self-attention

        // Add residual connection: x + attention(x)
        let x1: Vec<Vec<f32>> = x
            .iter()
            .zip(attn.iter())
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect();

        // LayerNorm after attention
        let norm1_out = self.norm1.forward(&x1);
        self.attn_output = norm1_out.clone(); // Cache for backward

        // === Feed Forward with skip connection ===
        let ffn_out = self.ffn.forward(&norm1_out);

        // Residual connection again: norm1_out + FFN(norm1_out)
        let x2: Vec<Vec<f32>> = norm1_out
            .iter()
            .zip(ffn_out.iter())
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect();

        // LayerNorm after FFN
        let norm2_out = self.norm2.forward(&x2);
        self.ffn_output = norm2_out.clone(); // Cache for backward

        norm2_out
    }

    /// Backward pass:
    /// Handles all gradients from output → input, chaining through layers.
    pub fn backward(&mut self, grad_output: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        // === Norm2 backward ===
        // grad_output → norm2
        let grad_norm2 = self.norm2.backward(grad_output, &self.ffn_output);

        // === FFN backward ===
        // FFN has residual, so grad_norm2 passes through both paths
        let grad_ffn_out = grad_norm2.clone(); // direct gradient path
        let grad_ffn = self.ffn.backward(&grad_ffn_out, &self.attn_output); // FFN path

        // Merge gradients from skip connection: grad1 + grad2
        let grad_skip_ffn: Vec<Vec<f32>> = grad_ffn_out
            .iter()
            .zip(grad_ffn.iter())
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect();

        // === Norm1 backward ===
        let grad_norm1 = self.norm1.backward(&grad_skip_ffn, &self.attn_output);

        // === MHA backward ===
        let grad_mha = self.mha.backward(&grad_norm1);

        // Final residual add: grad_norm1 + grad_mha
        let grad_input: Vec<Vec<f32>> = grad_norm1
            .iter()
            .zip(grad_mha.iter())
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect();

        grad_input
    }

    /// Performs parameter update for each sublayer (MHA, FFN, LayerNorms)
    pub fn step(&mut self, lr: f32) {
        self.mha.step(lr);
        self.norm1.step(lr);
        self.ffn.step(lr);
        self.norm2.step(lr);
    }
}
