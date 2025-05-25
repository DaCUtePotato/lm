use rand::Rng;
pub struct OutputProjection {
    weights: Vec<Vec<f32>>, //shape: vocab_size x embedding dim
    biases: Vec<f32>,       // shape: vocab_size
}

impl OutputProjection {
    pub fn new(embedding_dim: usize, vocab_size: usize) -> Self {
        let mut rng = rand::rng();
        let weights = (0..vocab_size)
            .map(|_| {
                (0..embedding_dim)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect::<Vec<f32>>()
            })
            .collect();
        let biases = vec![0.0; vocab_size];
        OutputProjection { weights, biases }
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
}
