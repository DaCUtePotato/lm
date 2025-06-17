
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::{thread_rng, Rng};

/// OutputProjection maps final embeddings to logits over a vocabulary.
/// It's essentially a linear layer: logits = embeddings * W^T + b
pub struct OutputProjection {
    weights: Vec<Vec<f32>>, // [vocab_size x embedding_dim] — each row is a word vector
    biases: Vec<f32>,       // [vocab_size] — bias per vocabulary item

    grad_weights: Vec<Vec<f32>>, // Accumulates gradients for weights during backprop
    grad_biases: Vec<f32>,       // Accumulates gradients for biases during backprop
}

impl OutputProjection {
    /// Creates a new projection layer with random small weights and zero biases.
    pub fn new(embedding_dim: usize, vocab_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialise weights in range [-0.1, 0.1]
        let weights = (0..vocab_size)
            .map(|_| {
                (0..embedding_dim)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();

        let biases = vec![0.0; vocab_size];

        // Gradients are initially zero
        let grad_weights = vec![vec![0.0; embedding_dim]; vocab_size];
        let grad_biases = vec![0.0; vocab_size];

        OutputProjection {
            weights,
            biases,
            grad_weights,
            grad_biases,
        }
    }

    /// Forward pass: computes logits from the input embeddings.
    /// For each input embedding: computes dot(weight_row, embedding) + bias
    pub fn forward(&self, embeddings: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        embeddings
            .iter()
            .map(|embedding| {
                self.weights
                    .iter()
                    .zip(self.biases.iter())
                    .map(|(w_row, &b)| {
                        w_row
                            .iter()
                            .zip(embedding.iter())
                            .map(|(w, e)| w * e)
                            .sum::<f32>()
                            + b
                    })
                    .collect()
            })
            .collect()
    }

    /// Backward pass: computes gradient w.r.t. input and accumulates grads for weights & biases.
    ///
    /// `grad_output[b][v]` is the gradient of loss w.r.t. logits
    /// `input[b][e]` is the input embedding for token `b`
    pub fn backward(
        &mut self,
        grad_output: &Vec<Vec<f32>>,
        input: &Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        let batch_size = grad_output.len();
        let vocab_size = self.biases.len();
        let embedding_dim = self.weights[0].len();

        // Zero-out old gradients
        for g_w_row in self.grad_weights.iter_mut() {
            for g_w in g_w_row.iter_mut() {
                *g_w = 0.0;
            }
        }
        for g_b in self.grad_biases.iter_mut() {
            *g_b = 0.0;
        }

        // We'll return gradients w.r.t. input embeddings
        let mut grad_input = vec![vec![0.0; embedding_dim]; batch_size];

        for b in 0..batch_size {
            for v in 0..vocab_size {
                let grad_out_val = grad_output[b][v];

                // ∂L/∂b_v = sum over batch of grad_output[b][v]
                self.grad_biases[v] += grad_out_val;

                for e in 0..embedding_dim {
                    // ∂L/∂W[v][e] += grad_output[b][v] * input[b][e]
                    self.grad_weights[v][e] += grad_out_val * input[b][e];

                    // ∂L/∂input[b][e] += grad_output[b][v] * weights[v][e]
                    grad_input[b][e] += grad_out_val * self.weights[v][e];
                }
            }
        }

        grad_input
    }

    /// Gradient descent update with simple gradient clipping
    pub fn step(&mut self, lr: f32) {
        let clip_value: f32 = 1.0;

        // Clip gradients for each row of weights to avoid exploding updates
        for grad_row in &mut self.grad_weights {
            clip_gradient(grad_row, clip_value);
        }

        // Clip bias gradients
        clip_gradient(&mut self.grad_biases, clip_value);

        let vocab_size = self.biases.len();
        let embedding_dim = self.weights[0].len();

        for v in 0..vocab_size {
            // Update biases
            self.biases[v] -= lr * self.grad_biases[v];

            for e in 0..embedding_dim {
                // Update weights
                self.weights[v][e] -= lr * self.grad_weights[v][e];
            }
        }

        // Gradient reset not necessary here, done in `backward()`
    }
}

/// Clips the L2 norm of the gradient vector to `clip_value`
fn clip_gradient(grad: &mut [f32], clip_value: f32) {
    let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > clip_value {
        let scale = clip_value / norm;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
}

/// Samples an index from logits using softmax + top-k + temperature sampling
///
/// This is used for inference to pick the next token.
/// - `temperature` controls randomness (lower → more deterministic)
/// - `top_k` restricts choices to the top-k logits
pub fn sample(logits: &[f32], temperature: f32, top_k: usize) -> usize {
    let mut rng = thread_rng();

    // Scale logits by temperature
    let mut logits: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x / temperature))
        .collect();

    // Keep top-k highest values
    logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    logits.truncate(top_k);

    // Compute softmax on top-k logits
    let max_logit = logits
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = logits.iter().map(|(_, x)| (x - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

    // Sample from categorical distribution
    let dist = WeightedIndex::new(&probs).unwrap();
    logits[dist.sample(&mut rng)].0
}
