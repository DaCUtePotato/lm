
use rand::thread_rng;
use rand::Rng;

/// A simple embedding layer with optional positional encoding and gradient tracking
pub struct Embedding {
    pub weight: Vec<Vec<f32>>,       // shape: [vocab_size][embedding_dim]
    pub pos_encoding: Vec<Vec<f32>>, // shape: [max_len][embedding_dim]
    pub grad_weight: Vec<Vec<f32>>,  // same shape, used to accumulate gradients
}

impl Embedding {
    /// Create a new embedding layer
    /// Randomly initialises the weights and generates positional encodings
    pub fn new(vocab_size: usize, embedding_dim: usize, max_len: usize) -> Self {
        let mut rng = thread_rng(); // Thread-local RNG for random weight init

        // Random init of embedding weights in range [-0.01, 0.01]
        let weight = (0..vocab_size)
            .map(|_| {
                (0..embedding_dim)
                    .map(|_| rng.gen_range(-0.01..0.01))
                    .collect()
            })
            .collect();

        // Initialise gradients to zero
        let grad_weight = vec![vec![0.0; embedding_dim]; vocab_size];

        // Precompute positional encodings (e.g. for transformers)
        let pos_encoding = Self::positional_encoding(max_len, embedding_dim);

        Self {
            weight,
            pos_encoding,
            grad_weight,
        }
    }

    /// Compute sinusoidal positional encoding (like in the original Transformer paper)
    pub fn positional_encoding(max_len: usize, embedding_dim: usize) -> Vec<Vec<f32>> {
        let mut pe = vec![vec![0.0; embedding_dim]; max_len];
        for pos in 0..max_len {
            for i in 0..embedding_dim {
                // Compute angle using inverse frequency method
                let angle =
                    pos as f32 / (10000_f32.powf((2 * (i / 2)) as f32 / embedding_dim as f32));
                pe[pos][i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }
        pe
    }

    /// Forward pass: fetch embeddings and add positional encodings
    pub fn forward(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        token_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                // Elementwise addition of token embedding and its position encoding
                self.weight[id]
                    .iter()
                    .zip(self.pos_encoding[i].iter())
                    .map(|(w, p)| w + p)
                    .collect()
            })
            .collect()
    }

    /// Backward pass: accumulate gradients into grad_weight for used token indices
    pub fn backward(&mut self, grad_output: &Vec<Vec<f32>>, input: &[usize]) {
        for (i, &token_id) in input.iter().enumerate() {
            let grad = &grad_output[i];
            let grad_row = &mut self.grad_weight[token_id];
            for j in 0..grad.len() {
                grad_row[j] += grad[j]; // accumulate gradient
            }
        }
    }

    /// Optimisation step: apply SGD update and reset gradients
    pub fn step(&mut self, lr: f32) {
        for (w_row, g_row) in self.weight.iter_mut().zip(self.grad_weight.iter_mut()) {
            clip_gradient(g_row, 1.); // prevent exploding gradients
            for (w, g) in w_row.iter_mut().zip(g_row.iter_mut()) {
                *w -= lr * *g; // SGD step
                *g = 0.0; // reset gradient
            }
        }
    }
}

/// Gradient clipping to avoid instability in training
fn clip_gradient(grad: &mut [f32], clip_value: f32) {
    let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt(); // L2 norm
    if norm > clip_value {
        let scale = clip_value / norm;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
}
