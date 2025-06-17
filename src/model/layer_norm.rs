
use std::f32;

/// A custom implementation of Layer Normalization.
/// LayerNorm normalises each input vector (usually per data sample) independently.
pub struct LayerNorm {
    pub gamma: Vec<f32>,       // Learnable scale parameter per feature
    pub beta: Vec<f32>,        // Learnable shift parameter per feature
    pub eps: f32,              // Small epsilon to avoid division by zero
    pub mean: Vec<f32>,        // Stored mean per input vector (for backward)
    pub var: Vec<f32>,         // Stored variance per input vector (for backward)
    pub normed: Vec<Vec<f32>>, // Stored normalised inputs (used in backward)
    pub grad_gamma: Vec<f32>,  // Accumulated gradient for gamma
    pub grad_beta: Vec<f32>,   // Accumulated gradient for beta
}

impl LayerNorm {
    /// Creates a new LayerNorm with input dimension `dim`.
    pub fn new(dim: usize) -> Self {
        let eps = 1e-6;
        Self {
            gamma: vec![1.0; dim], // Initialise gamma to 1
            beta: vec![0.0; dim],  // Initialise beta to 0
            eps,
            mean: vec![0.0; 0], // These will be filled during forward()
            var: vec![0.0; 0],
            normed: Vec::new(),
            grad_gamma: vec![0.0; dim],
            grad_beta: vec![0.0; dim],
        }
    }

    /// Forward pass of LayerNorm.
    /// Applies per-vector normalization across feature dimension.
    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut normed_batch = Vec::with_capacity(x.len());
        let mut mean_batch = Vec::with_capacity(x.len());
        let mut var_batch = Vec::with_capacity(x.len());

        for vec in x {
            let mean = vec.iter().copied().sum::<f32>() / vec.len() as f32;
            let var = vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vec.len() as f32;

            let normed: Vec<f32> = vec
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    // Normalise each input then scale and shift it
                    self.gamma[i] * ((v - mean) / (var + self.eps).sqrt()) + self.beta[i]
                })
                .collect();

            mean_batch.push(mean);
            var_batch.push(var);
            normed_batch.push(normed);
        }

        // Save intermediate values for backpropagation
        self.normed = normed_batch.clone();
        self.mean = mean_batch;
        self.var = var_batch;

        normed_batch
    }

    /// Backward pass to compute gradient of the loss w.r.t. inputs.
    ///
    /// `grad_output` is ∂L/∂y, the gradient from the next layer.
    /// `input` is the original input batch.
    pub fn backward(
        &mut self,
        grad_output: &Vec<Vec<f32>>,
        input: &Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        let batch_size = grad_output.len();
        let dim = self.gamma.len();

        // Reset gradients to zero before accumulation
        for i in 0..dim {
            self.grad_gamma[i] = 0.0;
            self.grad_beta[i] = 0.0;
        }

        // Accumulate gradients for gamma and beta (per feature)
        for b in 0..batch_size {
            for i in 0..dim {
                self.grad_gamma[i] += grad_output[b][i] * self.normed[b][i];
                self.grad_beta[i] += grad_output[b][i];
            }
        }

        // Now compute gradients w.r.t. input
        let mut grad_input = vec![vec![0.0; dim]; batch_size];

        for b in 0..batch_size {
            // Compute intermediate values for the formula
            let var = self.var[b];
            let inv_std = 1.0 / (var + self.eps).sqrt();

            let mut grad_output_gamma_mean = 0.0;
            let mut grad_output_gamma_normed_mean = 0.0;

            for i in 0..dim {
                let go_gamma = grad_output[b][i] * self.gamma[i];
                grad_output_gamma_mean += go_gamma;
                grad_output_gamma_normed_mean += go_gamma * self.normed[b][i];
            }

            grad_output_gamma_mean /= dim as f32;
            grad_output_gamma_normed_mean /= dim as f32;

            // Now calculate gradient for each feature
            for i in 0..dim {
                let go_gamma = grad_output[b][i] * self.gamma[i];
                grad_input[b][i] = inv_std
                    * (go_gamma
                        - grad_output_gamma_mean
                        - self.normed[b][i] * grad_output_gamma_normed_mean);
            }
        }

        grad_input
    }

    /// Gradient descent step: update `gamma` and `beta` using accumulated gradients.
    pub fn step(&mut self, lr: f32) {
        for i in 0..self.gamma.len() {
            self.gamma[i] -= lr * self.grad_gamma[i];
            self.grad_gamma[i] = 0.0; // Reset after update
        }
        for i in 0..self.beta.len() {
            self.beta[i] -= lr * self.grad_beta[i];
            self.grad_beta[i] = 0.0; // Reset after update
        }
    }
}
