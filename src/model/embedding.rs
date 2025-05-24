use rand::Rng;

pub struct Embedding {
    pub weight: Vec<Vec<f32>>, // shape is [vocab_size][embedding_dim]?
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::rng();
        let weight = (0..vocab_size)
            .map(|_| {
                (0..embedding_dim)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect()
            })
            .collect();
        Self { weight }
    }

    pub fn forward(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        token_ids
            .iter()
            .map(|&ids| self.weight[ids].clone())
            .collect()
    }
}
