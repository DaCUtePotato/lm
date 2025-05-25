use std::{f32::NEG_INFINITY, vec};

use rand::Rng;

/// This computes scaled dot-product attention
///
/// Q, K, V are matrices with shape: (sequence length, embedding dim)
/// Returns the output of shape (sequence length, embedding dim)

pub fn scaled_dot_product_attention(
    q: &Vec<Vec<f32>>,
    k: &Vec<Vec<f32>>,
    v: &Vec<Vec<f32>>,
    mask: Option<&Vec<Vec<f32>>>,
) -> Vec<Vec<f32>> {
    let seq_len = q.len();
    let dim = q[0].len();
    let scale = (dim as f32).sqrt();

    // 1. Compute QK^T
    let mut scores = vec![vec![0.0; seq_len]; seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0;
            for d in 0..dim {
                dot += q[i][d] * k[j][d];
            }
            scores[i][j] = dot / scale;
        }
    }

    // 2. Apply mask if provided
    if let Some(mask) = mask {
        for i in 0..seq_len {
            for j in 0..seq_len {
                if mask[i][j] == 0.0 {
                    scores[i][j] = f32::NEG_INFINITY;
                }
            }
        }
    }

    // 3. Softmax
    let mut weights = vec![vec![0.0; seq_len]; seq_len];
    for i in 0..seq_len {
        let max_score = scores[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores[i].iter().map(|s| (*s - max_score).exp()).collect();
        let sum_exp: f32 = exp.iter().sum();
        for j in 0..seq_len {
            weights[i][j] = exp[j] / sum_exp;
        }
    }

    // 4. Compute output
    let mut output = vec![vec![0.0; dim]; seq_len];
    for i in 0..seq_len {
        for d in 0..dim {
            for j in 0..seq_len {
                output[i][d] += weights[i][j] * v[j][d];
            }
        }
    }

    output
}

pub struct MultiHeadAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,

    pub w_q: Vec<Vec<f32>>,
    pub b_q: Vec<f32>,
    pub w_k: Vec<Vec<f32>>,
    pub b_k: Vec<f32>,
    pub w_v: Vec<Vec<f32>>,
    pub b_v: Vec<f32>,
    pub w_o: Vec<Vec<f32>>,
    pub b_o: Vec<f32>,
}
impl MultiHeadAttention {
    pub fn new(embed_dim: usize) -> Self {
        let num_heads = 4;
        assert!(
            embed_dim % num_heads == 0,
            "Embed dim must be divisible by num_heads"
        );
        let head_dim = embed_dim / num_heads;

        fn random_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
            let mut rng = rand::thread_rng();
            (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen_range(-0.1..0.1)).collect())
                .collect()
        }

        fn random_bias(size: usize) -> Vec<f32> {
            let mut rng = rand::thread_rng();
            (0..size).map(|_| rng.gen_range(-0.1..0.1)).collect()
        }

        Self {
            embed_dim,
            num_heads,
            head_dim,
            w_q: random_matrix(embed_dim, embed_dim),
            w_k: random_matrix(embed_dim, embed_dim),
            w_v: random_matrix(embed_dim, embed_dim),
            w_o: random_matrix(embed_dim, embed_dim),
            b_q: random_bias(embed_dim),
            b_k: random_bias(embed_dim),
            b_v: random_bias(embed_dim),
            b_o: random_bias(embed_dim),
        }
    }

    fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let (n, m) = (a.len(), b.len());
        let p = b[0].len();
        let mut result = vec![vec![0.0; p]; n];
        for i in 0..n {
            for j in 0..p {
                for k in 0..m {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        result
    }

    fn add_bias(x: &mut Vec<Vec<f32>>, bias: &[f32]) {
        for row in x.iter_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                *val += bias[i];
            }
        }
    }

    pub fn forward(&self, x: &[Vec<f32>], train: bool) -> Vec<Vec<f32>> {
        let mut q = Self::matmul(x, &self.w_q);
        Self::add_bias(&mut q, &self.b_q);
        let mut k = Self::matmul(x, &self.w_k);
        Self::add_bias(&mut k, &self.b_k);
        let mut v = Self::matmul(x, &self.w_v);
        Self::add_bias(&mut v, &self.b_v);

        let mut heads = vec![];

        for i in 0..self.num_heads {
            let start = i * self.head_dim;
            let end = start + self.head_dim;

            let q_head: Vec<Vec<f32>> = q.iter().map(|row| row[start..end].to_vec()).collect();
            let k_head: Vec<Vec<f32>> = k.iter().map(|row| row[start..end].to_vec()).collect();
            let v_head: Vec<Vec<f32>> = v.iter().map(|row| row[start..end].to_vec()).collect();

            let mask = causal_mask(x.len());
            let effective_mask = mask.clone(); //.unwrap_or_else(|| causal_mask(x.len()));

            let head_out = scaled_dot_product_attention(&q_head, &k_head, &v_head, Some(&mask));

            heads.push(head_out);
        }

        // Concatenate heads: (seq_len, num_heads * head_dim)
        let seq_len = x.len();
        let mut concat = vec![vec![]; seq_len];
        for head in heads {
            for (i, row) in head.into_iter().enumerate() {
                concat[i].extend(row);
            }
        }

        // Final projection
        let mut output = Self::matmul(&concat, &self.w_o);
        if train {
            dropout(&mut output, 0.1);
        }
        Self::add_bias(&mut output, &self.b_o);
        output
    }
}

pub fn causal_mask(seq_len: usize) -> Vec<Vec<f32>> {
    let mut mask = vec![vec![0.0; seq_len]; seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask[i][j] = 1.0;
        }
    }
    mask
}

pub fn dropout(x: &mut Vec<Vec<f32>>, p: f32) {
    let mut rng = rand::thread_rng();
    for row in x.iter_mut() {
        for val in row.iter_mut() {
            if rng.gen::<f32>() < p {
                *val = 0.0;
            }
        }
    }
}
