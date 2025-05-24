use std::{f32::MAX_10_EXP, sync::Exclusive};

use rand::Rng;

pub struct Embedding {
    pub weight: Vec<Vec<f32>>, // shape is [vocab_size][embedding_dim]
    pub pos_encoding: Vec<Vec<f32>>,
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_dim: usize, max_len: usize) -> Self {
        let mut rng = rand::rng();
        let weight = (0..vocab_size)
            .map(|_| {
                (0..embedding_dim)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect()
            })
            .collect();
        let pos_encoding = positional_encoding(max_len, embedding_dim);
        Self {
            weight,
            pos_encoding,
        }
    }

    pub fn positional_encoding(max_len: usize, embedding_dim: usize) -> Vec<Vec<f32>> {
        let mut pe = vec![vec![0.0; embedding_dim]; max_len];
        for pos in 0..max_len {
            for i in 0..embedding_dim {
                let angle =
                    pos as f32 / (1000_f32.powf((2 * (i / 2)) as f32 / embedding_dim as f32));
                pe[pos][i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }
        pe
    }

    pub fn forward(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        token_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                self.weight[id]
                    .iter()
                    .zip(self.pos_encoding[i].iter())
                    .map(|(w, p)| w + p)
                    .collect()
            })
            .collect()
    }
}
