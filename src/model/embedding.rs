
use rand::Rng;
pub struct Embedding {
    pub weight: Vec<Vec<f32>>,       // shape: [vocab_size][embedding_dim]
    pub pos_encoding: Vec<Vec<f32>>, // shape: [max_len][embedding_dim]
    pub grad_weight: Vec<Vec<f32>>,  // shape: [vocab_size][embedding_dim]
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_dim: usize, max_len: usize) -> Self {
        let mut rng = rand::thread_rng(); // Fix: rng() → thread_rng()
        let weight = (0..vocab_size)
            .map(|_| {
                (0..embedding_dim)
                    .map(|_| rng.gen_range(-0.01..0.01))
                    .collect()
            })
            .collect();

        let grad_weight = vec![vec![0.0; embedding_dim]; vocab_size];
        let pos_encoding = Self::positional_encoding(max_len, embedding_dim);

        Self {
            weight,
            pos_encoding,
            grad_weight,
        }
    }

    pub fn positional_encoding(max_len: usize, embedding_dim: usize) -> Vec<Vec<f32>> {
        let mut pe = vec![vec![0.0; embedding_dim]; max_len];
        for pos in 0..max_len {
            for i in 0..embedding_dim {
                let angle =
                    pos as f32 / (10000_f32.powf((2 * (i / 2)) as f32 / embedding_dim as f32));
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

    pub fn backward(&mut self, grad_output: &Vec<Vec<f32>>, input: &[usize]) {
        for (i, &token_id) in input.iter().enumerate() {
            let grad = &grad_output[i];
            let grad_row = &mut self.grad_weight[token_id];
            for j in 0..grad.len() {
                grad_row[j] += grad[j];
            }
        }
    }

    pub fn step(&mut self, lr: f32) {
        for (w_row, g_row) in self.weight.iter_mut().zip(self.grad_weight.iter_mut()) {
            for (w, g) in w_row.iter_mut().zip(g_row.iter_mut()) {
                *w -= lr * *g;
                *g = 0.0; // reset gradient after update
            }
        }
    }
}
