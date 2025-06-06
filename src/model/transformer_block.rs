use super::attention::{scaled_dot_product_attention, MultiHeadAttention};
use super::embedding;
use super::feed_forward::FeedForward;
use super::layer_norm::{self, LayerNorm};

pub struct TransformerBlock {
    mha: MultiHeadAttention,
    norm1: LayerNorm,
    ffn: FeedForward,
    norm2: LayerNorm,

    input: Vec<Vec<f32>>,
    attn_output: Vec<Vec<f32>>,
    ffn_output: Vec<Vec<f32>>,
}

impl TransformerBlock {
    pub fn new(embed_dim: usize, num_heads: usize, ff_hidden_dim: usize) -> Self {
        Self {
            mha: MultiHeadAttention::new(embed_dim, num_heads),
            norm1: LayerNorm::new(embed_dim),
            ffn: FeedForward::new(embed_dim, ff_hidden_dim),
            norm2: LayerNorm::new(embed_dim),

            input: vec![],
            attn_output: vec![],
            ffn_output: vec![],
        }
    }

    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.input = x.to_vec();

        let attn = self.mha.forward(x, true);
        let x1: Vec<Vec<f32>> = x
            .iter()
            .zip(attn.iter())
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect();

        let norm1_out = self.norm1.forward(&x1);
        self.attn_output = norm1_out.clone();

        let ffn_out = self.ffn.forward(&norm1_out);
        let x2: Vec<Vec<f32>> = norm1_out
            .iter()
            .zip(ffn_out.iter())
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect();

        let norm2_out = self.norm2.forward(&x2);
        self.ffn_output = norm2_out.clone();

        norm2_out
    }

    pub fn backward(&mut self, grad_output: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        // Norm2 backward
        let grad_norm2 = self.norm2.backward(grad_output, &self.ffn_output);

        // Residual connection from FFN
        let grad_ffn_out = grad_norm2.clone();
        let grad_ffn = self.ffn.backward(&grad_ffn_out, &self.attn_output);

        // Add grad from skip connection
        let grad_skip_ffn: Vec<Vec<f32>> = grad_ffn_out
            .iter()
            .zip(grad_ffn.iter())
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect();

        // Norm1 backward
        let grad_norm1 = self.norm1.backward(&grad_skip_ffn, &self.attn_output);

        // MHA backward
        let grad_mha = self.mha.backward(&grad_norm1);

        // Add grad from skip connection
        let grad_input: Vec<Vec<f32>> = grad_norm1
            .iter()
            .zip(grad_mha.iter())
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect();

        grad_input
    }

    pub fn step(&mut self, lr: f32) {
        self.mha.step(lr);
        self.norm1.step(lr);
        self.ffn.step(lr);
        self.norm2.step(lr);
    }
}
