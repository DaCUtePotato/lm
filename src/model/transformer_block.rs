use super::attention::{scaled_dot_product_attention, MultiHeadAttention};
use super::embedding;
use super::feed_forward::FeedForward;
use super::layer_norm::{self, LayerNorm};

pub struct TransformerBlock {
    mha: MultiHeadAttention,
    norm1: LayerNorm,
    ffn: FeedForward,
    norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(embed_dim: usize) -> Self {
        Self {
            mha: MultiHeadAttention::new(embed_dim),
            norm1: LayerNorm::new(embed_dim, 1e-5),
            ffn: FeedForward::new(embed_dim, embed_dim * 4),
            norm2: LayerNorm::new(embed_dim, 1e-5),
        }
    }

    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Self-attention+residual+norm
        let attn_out = self.mha.forward(x);
        let res1: Vec<Vec<f32>> = x
            .iter()
            .zip(attn_out.iter())
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect();
        let normed1 = self.norm1.forward(&res1);

        let ff_out = self.ffn.forward(&normed1);
        let res2: Vec<Vec<f32>> = normed1
            .iter()
            .zip(ff_out.iter())
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect();
        let normed2 = self.norm2.forward(&res2);

        normed2
    }
}
