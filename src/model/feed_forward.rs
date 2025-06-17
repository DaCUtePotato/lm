
pub struct FeedForward {
    w1: Vec<Vec<f32>>, // Weights for first linear layer: [input_dim][hidden_dim]
    b1: Vec<f32>,      // Bias for first layer: [hidden_dim]
    w2: Vec<Vec<f32>>, // Weights for second linear layer: [hidden_dim][input_dim]
    b2: Vec<f32>,      // Bias for second layer: [input_dim]

    // Cache for backward pass
    input: Vec<Vec<f32>>,  // Stores input to first layer during forward pass
    hidden: Vec<Vec<f32>>, // Stores output of ReLU from first layer

    // Gradients (same shape as weights and biases)
    grad_w1: Vec<Vec<f32>>,
    grad_b1: Vec<f32>,
    grad_w2: Vec<Vec<f32>>,
    grad_b2: Vec<f32>,
}

/// ReLU activation function
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

impl FeedForward {
    /// Constructs a new FeedForward network with random weights
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialise weights randomly in [-0.1, 0.1]
        let w1 = (0..input_dim)
            .map(|_| (0..hidden_dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let b1 = vec![0.0; hidden_dim];

        let w2 = (0..hidden_dim)
            .map(|_| (0..input_dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let b2 = vec![0.0; input_dim];

        // Zero-initialised gradients
        let grad_w1 = vec![vec![0.0; hidden_dim]; input_dim];
        let grad_b1 = vec![0.0; hidden_dim];
        let grad_w2 = vec![vec![0.0; input_dim]; hidden_dim];
        let grad_b2 = vec![0.0; input_dim];

        Self {
            w1,
            b1,
            w2,
            b2,
            input: vec![],
            hidden: vec![],
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
        }
    }

    /// Forward pass through the FeedForward network
    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.input = x.to_vec(); // Store input for backward pass

        // First linear layer + ReLU activation
        let hidden: Vec<Vec<f32>> = x
            .iter()
            .map(|vec| {
                self.w1
                    .iter()
                    .zip(&self.b1)
                    .map(|(row, b)| relu(row.iter().zip(vec).map(|(w, v)| w * v).sum::<f32>() + b))
                    .collect()
            })
            .collect();

        self.hidden = hidden.clone(); // Cache activations for backprop

        // Second linear layer (no activation)
        hidden
            .iter()
            .map(|h| {
                self.w2
                    .iter()
                    .zip(&self.b2)
                    .map(|(row, b)| row.iter().zip(h).map(|(w, h)| w * h).sum::<f32>() + b)
                    .collect()
            })
            .collect()
    }

    /// Backward pass: computes gradients w.r.t weights and inputs
    pub fn backward(
        &mut self,
        grad_output: &Vec<Vec<f32>>,   // dL/dOutput
        _input_unused: &Vec<Vec<f32>>, // Placeholder, not used
    ) -> Vec<Vec<f32>> {
        let batch_size = grad_output.len();
        let hidden_dim = self.hidden[0].len();
        let input_dim = self.w1.len();

        // Gradient of loss w.r.t hidden layer output (after ReLU)
        let mut grad_hidden = vec![vec![0.0; hidden_dim]; batch_size];

        // Backprop through second linear layer: accumulate grad_w2 and grad_b2
        for i in 0..batch_size {
            for j in 0..self.w2[0].len() {
                for k in 0..hidden_dim {
                    self.grad_w2[k][j] += self.hidden[i][k] * grad_output[i][j];
                    grad_hidden[i][k] += self.w2[k][j] * grad_output[i][j];
                }
                self.grad_b2[j] += grad_output[i][j];
            }
        }

        // Backprop through ReLU (zero out where ReLU output was zero)
        for i in 0..batch_size {
            for j in 0..hidden_dim {
                if self.hidden[i][j] <= 0.0 {
                    grad_hidden[i][j] = 0.0;
                }
            }
        }

        // Backprop through first linear layer: accumulate grad_w1 and grad_b1
        let mut grad_input = vec![vec![0.0; input_dim]; batch_size];
        for i in 0..batch_size {
            for j in 0..hidden_dim {
                for k in 0..input_dim {
                    self.grad_w1[k][j] += self.input[i][k] * grad_hidden[i][j];
                    grad_input[i][k] += self.w1[k][j] * grad_hidden[i][j];
                }
                self.grad_b1[j] += grad_hidden[i][j];
            }
        }

        grad_input // Return gradient w.r.t input for upstream layer
    }

    /// SGD update step with gradient clipping
    pub fn step(&mut self, lr: f32) {
        let clip_value: f32 = 1.0;

        // Clip gradients to prevent exploding gradients
        for row in &mut self.grad_w1 {
            clip_gradient(row, clip_value);
        }
        for row in &mut self.grad_w2 {
            clip_gradient(row, clip_value);
        }
        clip_gradient(&mut self.grad_b1, clip_value);
        clip_gradient(&mut self.grad_b2, clip_value);

        // Gradient descent: w = w - lr * grad
        for i in 0..self.w1.len() {
            for j in 0..self.w1[0].len() {
                self.w1[i][j] -= lr * self.grad_w1[i][j];
                self.grad_w1[i][j] = 0.0;
            }
        }

        for i in 0..self.b1.len() {
            self.b1[i] -= lr * self.grad_b1[i];
            self.grad_b1[i] = 0.0;
        }

        for i in 0..self.w2.len() {
            for j in 0..self.w2[0].len() {
                self.w2[i][j] -= lr * self.grad_w2[i][j];
                self.grad_w2[i][j] = 0.0;
            }
        }

        for i in 0..self.b2.len() {
            self.b2[i] -= lr * self.grad_b2[i];
            self.grad_b2[i] = 0.0;
        }
    }
}

/// Clip gradient vector to a maximum L2 norm
fn clip_gradient(grad: &mut [f32], clip_value: f32) {
    let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > clip_value {
        let scale = clip_value / norm;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
}
