use std::f32;

pub struct LayerNorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub eps: f32,
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
    pub normed: Vec<Vec<f32>>,
    pub grad_gamma: Vec<f32>,
    pub grad_beta: Vec<f32>,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        let eps = 1e-6;
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            eps,
            mean: vec![0.0; dim],
            var: vec![0.0; dim],
            normed: Vec::new(), // No initial data, empty vec of vecs
            grad_gamma: vec![0.0; dim],
            grad_beta: vec![0.0; dim],
        }
    }
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
                .map(|(i, v)| self.gamma[i] * ((v - mean) / (var + self.eps).sqrt()) + self.beta[i])
                .collect();

            mean_batch.push(mean);
            var_batch.push(var);
            normed_batch.push(normed);
        }

        // Store for backward pass
        self.normed = normed_batch.clone();
        self.mean = mean_batch;
        self.var = var_batch;

        normed_batch
    }

    // grad_output shape: [batch_size][dim]
    // input shape: [batch_size][dim]
    pub fn backward(
        &mut self,
        grad_output: &Vec<Vec<f32>>,
        input: &Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        let batch_size = grad_output.len();
        let dim = self.gamma.len();

        // Reset grad_gamma and grad_beta before accumulation
        for i in 0..dim {
            self.grad_gamma[i] = 0.0;
            self.grad_beta[i] = 0.0;
        }

        // Calculate gradients for gamma and beta
        for b in 0..batch_size {
            for i in 0..dim {
                // dL/dgamma = sum over batch of grad_output * normed
                self.grad_gamma[i] += grad_output[b][i] * self.normed[b][i];
                // dL/dbeta = sum over batch of grad_output
                self.grad_beta[i] += grad_output[b][i];
            }
        }

        // Now compute gradient wrt input
        // Using LayerNorm backward formula:
        //
        // Let:
        // x_hat = normed[b][i]
        // N = dim
        //
        // dL/dx = (1 / sqrt(var + eps)) * (grad_output * gamma - mean(grad_output * gamma) - x_hat * mean(grad_output * gamma * x_hat))
        //
        // We'll compute these terms per batch

        let mut grad_input = vec![vec![0.0; dim]; batch_size];

        for b in 0..batch_size {
            // Compute mean of grad_output * gamma
            let mut grad_output_gamma_mean = 0.0;
            for i in 0..dim {
                grad_output_gamma_mean += grad_output[b][i] * self.gamma[i];
            }
            grad_output_gamma_mean /= dim as f32;

            // Compute mean of grad_output * gamma * normed
            let mut grad_output_gamma_normed_mean = 0.0;
            for i in 0..dim {
                grad_output_gamma_normed_mean +=
                    grad_output[b][i] * self.gamma[i] * self.normed[b][i];
            }
            grad_output_gamma_normed_mean /= dim as f32;

            for i in 0..dim {
                grad_input[b][i] = (1.0 / (self.var[i] + self.eps).sqrt())
                    * (grad_output[b][i] * self.gamma[i]
                        - grad_output_gamma_mean
                        - self.normed[b][i] * grad_output_gamma_normed_mean);
            }
        }

        grad_input
    }

    pub fn step(&mut self, lr: f32) {
        for i in 0..self.gamma.len() {
            self.gamma[i] -= lr * self.grad_gamma[i];
            self.grad_gamma[i] = 0.0;
        }
        for i in 0..self.beta.len() {
            self.beta[i] -= lr * self.grad_beta[i];
            self.grad_beta[i] = 0.0;
        }
    }
}
