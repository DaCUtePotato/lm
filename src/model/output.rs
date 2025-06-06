use rand::Rng;

pub struct OutputProjection {
    weights: Vec<Vec<f32>>, // vocab_size x embedding_dim
    biases: Vec<f32>,       // vocab_size

    grad_weights: Vec<Vec<f32>>, // same shape as weights
    grad_biases: Vec<f32>,       // same shape as biases
}

impl OutputProjection {
    pub fn new(embedding_dim: usize, vocab_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..vocab_size)
            .map(|_| {
                (0..embedding_dim)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        let biases = vec![0.0; vocab_size];

        let grad_weights = vec![vec![0.0; embedding_dim]; vocab_size];
        let grad_biases = vec![0.0; vocab_size];

        OutputProjection {
            weights,
            biases,
            grad_weights,
            grad_biases,
        }
    }

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

    pub fn backward(
        &mut self,
        grad_output: &Vec<Vec<f32>>,
        input: &Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        let batch_size = grad_output.len();
        let vocab_size = self.biases.len();
        let embedding_dim = self.weights[0].len();

        // Clear previous gradients
        for g_w_row in self.grad_weights.iter_mut() {
            for g_w in g_w_row.iter_mut() {
                *g_w = 0.0;
            }
        }
        for g_b in self.grad_biases.iter_mut() {
            *g_b = 0.0;
        }

        // Gradients w.r.t input embeddings (to return)
        let mut grad_input = vec![vec![0.0; embedding_dim]; batch_size];

        // Compute gradients:
        // grad_weights[v][e] += sum over batch of grad_output[b][v] * input[b][e]
        // grad_biases[v] += sum over batch of grad_output[b][v]
        // grad_input[b][e] += sum over vocab v of grad_output[b][v] * weights[v][e]

        for b in 0..batch_size {
            for v in 0..vocab_size {
                let grad_out_val = grad_output[b][v];
                self.grad_biases[v] += grad_out_val;

                for e in 0..embedding_dim {
                    self.grad_weights[v][e] += grad_out_val * input[b][e];
                    grad_input[b][e] += grad_out_val * self.weights[v][e];
                }
            }
        }

        grad_input
    }

    pub fn step(&mut self, lr: f32) {
        let vocab_size = self.biases.len();
        let embedding_dim = self.weights[0].len();

        for v in 0..vocab_size {
            self.biases[v] -= lr * self.grad_biases[v];

            for e in 0..embedding_dim {
                self.weights[v][e] -= lr * self.grad_weights[v][e];
            }
        }

        // After the step, gradients could be cleared here if desired
        // but since backward clears them, this isn't mandatory.
    }
}
