use std::{f32::NEG_INFINITY, vec};

use rand::Rng;

/// This computes scaled dot-product attention
///
/// Q, K, V are matrices with shape: (sequence length, embedding dim)
/// Returns the output of shape (sequence length, embedding dim)

pub fn scaled_dot_product_attention(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    mask: Option<&Vec<Vec<f32>>>,
) -> Vec<Vec<f32>> {
    let seq_len = q.len();
    let dim = q[0].len();
    let scale = (dim as f32).sqrt();

    // 1. Compute QK^T
    let mut scores = vec![vec![0.0; seq_len]; seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let dot = (0..dim).map(|d| q[i][d] * k[j][d]).sum::<f32>();
            scores[i][j] = dot / scale;
        }
    }

    // 2. Apply mask if provided
    if let Some(mask) = mask {
        for i in 0..seq_len {
            for j in 0..seq_len {
                if mask[i][j] == 0.0 {
                    scores[i][j] = NEG_INFINITY;
                }
            }
        }
    }

    // 3. Softmax
    let mut weights = vec![vec![0.0; seq_len]; seq_len];
    for i in 0..seq_len {
        let max_score = scores[i].iter().cloned().fold(NEG_INFINITY, f32::max);
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
    pub w_k: Vec<Vec<f32>>,
    pub w_v: Vec<Vec<f32>>,
    pub w_o: Vec<Vec<f32>>,

    pub b_q: Vec<f32>,
    pub b_k: Vec<f32>,
    pub b_v: Vec<f32>,
    pub b_o: Vec<f32>,

    pub grad_w_q: Vec<Vec<f32>>,
    pub grad_w_k: Vec<Vec<f32>>,
    pub grad_w_v: Vec<Vec<f32>>,
    pub grad_w_o: Vec<Vec<f32>>,

    pub grad_b_q: Vec<f32>,
    pub grad_b_k: Vec<f32>,
    pub grad_b_v: Vec<f32>,
    pub grad_b_o: Vec<f32>,

    pub cached_x: Vec<Vec<Vec<f32>>>, // input x
    pub cached_q: Vec<Vec<Vec<f32>>>, // Q after linear + bias
    pub cached_k: Vec<Vec<Vec<f32>>>,
    pub cached_v: Vec<Vec<Vec<f32>>>,
    pub cached_concat_heads: Vec<Vec<f32>>, // concatenated output of heads before w_o
    pub cached_heads: Vec<Vec<Vec<f32>>>,   // each head out
    pub cached_softmax_weights: Vec<Vec<Vec<f32>>>, // [num_heads][seq_len][seq_len]
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(embed_dim % num_heads == 0);
        let head_dim = embed_dim / num_heads;

        fn random_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
            let mut rng = rand::thread_rng();
            (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen_range(-0.1..0.1)).collect())
                .collect()
        }

        fn zero_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
            vec![vec![0.0; cols]; rows]
        }

        fn random_bias(size: usize) -> Vec<f32> {
            let mut rng = rand::thread_rng();
            (0..size).map(|_| rng.gen_range(-0.1..0.1)).collect()
        }

        fn zero_bias(size: usize) -> Vec<f32> {
            vec![0.0; size]
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

            grad_w_q: zero_matrix(embed_dim, embed_dim),
            grad_w_k: zero_matrix(embed_dim, embed_dim),
            grad_w_v: zero_matrix(embed_dim, embed_dim),
            grad_w_o: zero_matrix(embed_dim, embed_dim),

            grad_b_q: zero_bias(embed_dim),
            grad_b_k: zero_bias(embed_dim),
            grad_b_v: zero_bias(embed_dim),
            grad_b_o: zero_bias(embed_dim),

            cached_x: vec![],
            cached_q: vec![],
            cached_k: vec![],
            cached_v: vec![],
            cached_concat_heads: vec![],
            cached_heads: vec![],
            cached_softmax_weights: vec![],
        }
    }
    pub fn forward(&mut self, x: &[Vec<f32>], train: bool) -> Vec<Vec<f32>> {
        self.cached_x = split_heads(&x, self.num_heads);
        self.cached_softmax_weights.clear();

        let mut q = Self::matmul(x, &self.w_q);
        Self::add_bias(&mut q, &self.b_q);
        self.cached_q = split_heads(&q, self.num_heads);

        let mut k = Self::matmul(x, &self.w_k);
        Self::add_bias(&mut k, &self.b_k);
        self.cached_k = split_heads(&k, self.num_heads);

        let mut v = Self::matmul(x, &self.w_v);
        Self::add_bias(&mut v, &self.b_v);
        self.cached_v = split_heads(&v, self.num_heads);

        let mut heads = vec![];
        for i in 0..self.num_heads {
            let start = i * self.head_dim;
            let end = start + self.head_dim;

            let q_head: Vec<Vec<f32>> = q.iter().map(|row| row[start..end].to_vec()).collect();
            let k_head: Vec<Vec<f32>> = k.iter().map(|row| row[start..end].to_vec()).collect();
            let v_head: Vec<Vec<f32>> = v.iter().map(|row| row[start..end].to_vec()).collect();

            let mask = causal_mask(x.len());
            let (head_out, softmax_weights) =
                scaled_dot_product_attention_with_weights(&q_head, &k_head, &v_head, Some(&mask));

            self.cached_softmax_weights.push(softmax_weights);
            heads.push(head_out);
        }
        self.cached_heads = heads.clone();

        let seq_len = x.len();
        let mut concat = vec![vec![]; seq_len];
        for head in &heads {
            for (i, row) in head.iter().enumerate() {
                concat[i].extend(row);
            }
        }
        self.cached_concat_heads = concat.clone();

        let mut output = Self::matmul(&concat, &self.w_o);
        if train {
            dropout(&mut output, 0.1);
        }
        Self::add_bias(&mut output, &self.b_o);

        output
    }

    pub fn step(&mut self, lr: f32) {
        let clip_value = 1.;
        // Clip 1D gradients
        clip_gradient(&mut self.grad_b_q, clip_value);
        clip_gradient(&mut self.grad_b_k, clip_value);
        clip_gradient(&mut self.grad_b_v, clip_value);
        clip_gradient(&mut self.grad_b_o, clip_value);

        // Clip 2D gradients row-wise
        for row in &mut self.grad_w_q {
            clip_gradient(row, clip_value);
        }
        for row in &mut self.grad_w_k {
            clip_gradient(row, clip_value);
        }
        for row in &mut self.grad_w_v {
            clip_gradient(row, clip_value);
        }
        for row in &mut self.grad_w_o {
            clip_gradient(row, clip_value);
        }
        for i in 0..self.embed_dim {
            self.b_q[i] -= lr * self.grad_b_q[i];
            self.b_k[i] -= lr * self.grad_b_k[i];
            self.b_v[i] -= lr * self.grad_b_v[i];
            self.b_o[i] -= lr * self.grad_b_o[i];

            self.grad_b_q[i] = 0.0;
            self.grad_b_k[i] = 0.0;
            self.grad_b_v[i] = 0.0;
            self.grad_b_o[i] = 0.0;

            for j in 0..self.embed_dim {
                self.w_q[i][j] -= lr * self.grad_w_q[i][j];
                self.w_k[i][j] -= lr * self.grad_w_k[i][j];
                self.w_v[i][j] -= lr * self.grad_w_v[i][j];
                self.w_o[i][j] -= lr * self.grad_w_o[i][j];

                self.grad_w_q[i][j] = 0.0;
                self.grad_w_k[i][j] = 0.0;
                self.grad_w_v[i][j] = 0.0;
                self.grad_w_o[i][j] = 0.0;
            }
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
    pub fn backward(&mut self, grad_output: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = grad_output.len();

        // 1. Backprop output linear layer: output = concat_heads * w_o + b_o
        // grad_w_o += concat_heads^T ⋅ grad_output
        // grad_b_o += sum over batch grad_output

        for i in 0..self.embed_dim {
            self.grad_b_o[i] += grad_output.iter().map(|row| row[i]).sum::<f32>();
        }
        for i in 0..self.embed_dim {
            for j in 0..self.embed_dim {
                let mut grad = 0.0;
                for b in 0..seq_len {
                    grad += self.cached_concat_heads[b][i] * grad_output[b][j];
                }
                self.grad_w_o[i][j] += grad;
            }
        }

        // grad_concat_heads = grad_output ⋅ w_o^T
        let mut grad_concat_heads = vec![vec![0.0; self.embed_dim]; seq_len];
        for b in 0..seq_len {
            for i in 0..self.embed_dim {
                for j in 0..self.embed_dim {
                    grad_concat_heads[b][i] += grad_output[b][j] * self.w_o[i][j];
                }
            }
        }

        // 2. Split grad_concat_heads into heads
        let mut grad_heads: Vec<Vec<Vec<f32>>> =
            vec![vec![vec![0.0; self.head_dim]; seq_len]; self.num_heads];
        for i in 0..self.num_heads {
            let start = i * self.head_dim;
            let end = start + self.head_dim;
            for b in 0..seq_len {
                grad_heads[i][b].copy_from_slice(&grad_concat_heads[b][start..end]);
            }
        }

        // 3. Backprop through each head's scaled dot-product attention
        // We need to have cached q_head, k_head, v_head, and softmax weights from forward for each head.

        // Since you do not currently cache q_head/k_head/v_head or softmax weights for each head,
        // you'd need to extend your struct and forward to cache these.
        // For the sake of example, I assume you add:
        // self.cached_q_heads: Vec<Vec<Vec<f32>>>, self.cached_k_heads, self.cached_v_heads, self.cached_softmax_weights

        // For now, let's just assume you have those caches and implement backward accordingly:
        let mut grad_q = vec![vec![0.0; self.embed_dim]; seq_len];
        let mut grad_k = vec![vec![0.0; self.embed_dim]; seq_len];
        let mut grad_v = vec![vec![0.0; self.embed_dim]; seq_len];

        for i in 0..self.num_heads {
            // Extract cached heads for this i
            let q_head = &self.cached_q[i];
            let k_head = &self.cached_k[i];
            let v_head = &self.cached_v[i];
            let softmax_weights = &self.cached_softmax_weights[i];

            let (grad_q_head, grad_k_head, grad_v_head) = scaled_dot_product_attention_backward(
                q_head,
                k_head,
                v_head,
                Some(&causal_mask(seq_len)),
                &grad_heads[i],
                softmax_weights,
            );

            // Accumulate gradients into full grad_q/k/v
            let start = i * self.head_dim;
            let end = start + self.head_dim;
            for b in 0..seq_len {
                for d in 0..self.head_dim {
                    grad_q[b][start + d] += grad_q_head[b][d];
                    grad_k[b][start + d] += grad_k_head[b][d];
                    grad_v[b][start + d] += grad_v_head[b][d];
                }
            }
        }

        // 4. Backprop through input projections:
        // grad_w_q += x^T ⋅ grad_q, grad_b_q += sum over batch grad_q
        // grad_x += grad_q ⋅ w_q^T
        // similarly for k and v

        // Initialise grad_x zero matrix
        let mut grad_x = vec![vec![0.0; self.embed_dim]; seq_len];

        // Helper for one projection backward
        fn back_proj(
            x: &[Vec<Vec<f32>>],     // [batch][seq_len][embed_dim]
            grad_out: &[Vec<f32>],   // [batch][proj_dim]
            w: &[Vec<f32>],          // [embed_dim][proj_dim]
            grad_w: &mut [Vec<f32>], // [embed_dim][proj_dim]
            grad_b: &mut [f32],      // [proj_dim]
            grad_x: &mut [Vec<f32>], // [seq_len][embed_dim]
        ) {
            let batch_size = x.len();
            let seq_len = x[0].len();
            let embed_dim = x[0][0].len();
            let proj_dim = grad_out[0].len();

            // grad_w
            for i in 0..embed_dim {
                for j in 0..proj_dim {
                    let mut grad = 0.0;
                    for b in 0..batch_size {
                        for t in 0..seq_len {
                            grad += x[b][t][i] * grad_out[b][j];
                        }
                    }
                    grad_w[i][j] += grad;
                }
            }

            // grad_b
            for j in 0..proj_dim {
                for b in 0..batch_size {
                    grad_b[j] += grad_out[b][j];
                }
            }

            // grad_x
            for t in 0..seq_len {
                for i in 0..embed_dim {
                    let mut grad = 0.0;
                    for b in 0..batch_size {
                        for j in 0..proj_dim {
                            grad += w[i][j] * grad_out[b][j];
                        }
                    }
                    grad_x[t][i] += grad;
                }
            }
        }

        back_proj(
            &self.cached_x,
            &grad_q,
            &self.w_q,
            &mut self.grad_w_q,
            &mut self.grad_b_q,
            &mut grad_x,
        );
        back_proj(
            &self.cached_x,
            &grad_k,
            &self.w_k,
            &mut self.grad_w_k,
            &mut self.grad_b_k,
            &mut grad_x,
        );
        back_proj(
            &self.cached_x,
            &grad_v,
            &self.w_v,
            &mut self.grad_w_v,
            &mut self.grad_b_v,
            &mut grad_x,
        );

        grad_x
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
pub fn scaled_dot_product_attention_backward(
    q: &Vec<Vec<f32>>,
    k: &Vec<Vec<f32>>,
    v: &Vec<Vec<f32>>,
    mask: Option<&Vec<Vec<f32>>>,
    grad_output: &[Vec<f32>],
    cached_weights: &[Vec<f32>], // softmax weights cached from forward pass
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    // Returns gradients wrt q, k, v (all shape seq_len x head_dim)
    let seq_len = q.len();
    let dim = q[0].len();
    let scale = (dim as f32).sqrt();

    // Step 1: grad_output wrt v is weighted by softmax weights
    // grad_v = softmax_weights^T ⋅ grad_output
    let mut grad_v = vec![vec![0.0; dim]; seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            for d in 0..dim {
                grad_v[j][d] += cached_weights[i][j] * grad_output[i][d];
            }
        }
    }

    // Step 2: grad_output wrt softmax weights = grad_output ⋅ v^T
    let mut grad_weights = vec![vec![0.0; seq_len]; seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            for d in 0..dim {
                grad_weights[i][j] += grad_output[i][d] * v[j][d];
            }
        }
    }

    // Step 3: Backprop through softmax
    // For each i, softmax gradient wrt scores
    let mut grad_scores = vec![vec![0.0; seq_len]; seq_len];
    for i in 0..seq_len {
        let mut dot = 0.0;
        for j in 0..seq_len {
            dot += grad_weights[i][j] * cached_weights[i][j];
        }
        for j in 0..seq_len {
            grad_scores[i][j] = cached_weights[i][j] * (grad_weights[i][j] - dot);
        }
    }

    // Step 4: Backprop through scaling and dot product QK^T:
    // scores = QK^T / scale
    // grad_scores wrt Q and K:
    let mut grad_q = vec![vec![0.0; dim]; seq_len];
    let mut grad_k = vec![vec![0.0; dim]; seq_len];

    for i in 0..seq_len {
        for j in 0..seq_len {
            for d in 0..dim {
                grad_q[i][d] += grad_scores[i][j] * k[j][d] / scale;
                grad_k[j][d] += grad_scores[i][j] * q[i][d] / scale;
            }
        }
    }

    // Apply mask: gradients of masked positions are zeroed (if mask is Some)
    if let Some(mask) = mask {
        for i in 0..seq_len {
            for j in 0..seq_len {
                if mask[i][j] == 0.0 {
                    for d in 0..dim {
                        grad_q[i][d] = 0.0;
                        grad_k[j][d] = 0.0;
                    }
                }
            }
        }
    }

    (grad_q, grad_k, grad_v)
}

pub fn scaled_dot_product_attention_with_weights(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    mask: Option<&Vec<Vec<f32>>>,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let seq_len = q.len();
    let dim = q[0].len();
    let scale = (dim as f32).sqrt();

    // 1. Compute QK^T / sqrt(d_k)
    let mut scores = vec![vec![0.0; seq_len]; seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let dot = (0..dim).map(|d| q[i][d] * k[j][d]).sum::<f32>();
            scores[i][j] = dot / scale;
        }
    }

    // 2. Apply mask
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
    let mut softmax_weights = vec![vec![0.0; seq_len]; seq_len];
    for i in 0..seq_len {
        let max_score = scores[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores[i].iter().map(|s| (*s - max_score).exp()).collect();
        let sum_exp: f32 = exp.iter().sum();
        for j in 0..seq_len {
            softmax_weights[i][j] = exp[j] / sum_exp;
        }
    }

    // 4. Attention output = softmax_weights @ V
    let mut output = vec![vec![0.0; dim]; seq_len];
    for i in 0..seq_len {
        for d in 0..dim {
            for j in 0..seq_len {
                output[i][d] += softmax_weights[i][j] * v[j][d];
            }
        }
    }

    (output, softmax_weights)
}

pub fn split_heads(x: &[Vec<f32>], num_heads: usize) -> Vec<Vec<Vec<f32>>> {
    let seq_len = x.len();
    let embed_dim = x[0].len();
    let head_dim = embed_dim / num_heads;
    let mut result = vec![vec![vec![0.0; head_dim]; seq_len]; num_heads];

    for t in 0..seq_len {
        for h in 0..num_heads {
            for d in 0..head_dim {
                result[h][t][d] = x[t][h * head_dim + d];
            }
        }
    }
    result
}

fn clip_gradient(grad: &mut [f32], clip_value: f32) {
    let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > clip_value {
        let scale = clip_value / norm;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
}
