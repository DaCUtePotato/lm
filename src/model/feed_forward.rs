pub struct FeedForward {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,

    // Cache for backward
    input: Vec<Vec<f32>>,
    hidden: Vec<Vec<f32>>,

    // Gradients
    grad_w1: Vec<Vec<f32>>,
    grad_b1: Vec<f32>,
    grad_w2: Vec<Vec<f32>>,
    grad_b2: Vec<f32>,
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

impl FeedForward {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let w1 = (0..input_dim)
            .map(|_| (0..hidden_dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let b1 = vec![0.0; hidden_dim];

        let w2 = (0..hidden_dim)
            .map(|_| (0..input_dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let b2 = vec![0.0; input_dim];

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

    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.input = x.to_vec();

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

        self.hidden = hidden.clone();

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

    pub fn backward(
        &mut self,
        grad_output: &Vec<Vec<f32>>,
        _input_unused: &Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        let batch_size = grad_output.len();
        let hidden_dim = self.hidden[0].len();
        let input_dim = self.w1.len();

        let mut grad_hidden = vec![vec![0.0; hidden_dim]; batch_size];

        for i in 0..batch_size {
            for j in 0..self.w2[0].len() {
                for k in 0..hidden_dim {
                    self.grad_w2[k][j] += self.hidden[i][k] * grad_output[i][j];
                    grad_hidden[i][k] += self.w2[k][j] * grad_output[i][j];
                }
                self.grad_b2[j] += grad_output[i][j];
            }
        }

        for i in 0..batch_size {
            for j in 0..hidden_dim {
                if self.hidden[i][j] <= 0.0 {
                    grad_hidden[i][j] = 0.0;
                }
            }
        }

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

        grad_input
    }

    pub fn step(&mut self, lr: f32) {
        let clip_value: f32 = 1.;
        // Clip 2D gradients row-wise
        for row in &mut self.grad_w1 {
            clip_gradient(row, clip_value);
        }
        for row in &mut self.grad_w2 {
            clip_gradient(row, clip_value);
        }

        // Clip 1D gradients
        clip_gradient(&mut self.grad_b1, clip_value);
        clip_gradient(&mut self.grad_b2, clip_value);
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

fn clip_gradient(grad: &mut [f32], clip_value: f32) {
    let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > clip_value {
        let scale = clip_value / norm;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
}
