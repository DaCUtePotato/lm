pub struct FeedForward {
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

impl FeedForward {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();

        let w1 = (0..input_dim)
            .map(|_| (0..hidden_dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let b1 = vec![0.0; hidden_dim];

        let w2 = (0..input_dim)
            .map(|_| (0..hidden_dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();

        let b2 = vec![0.0; hidden_dim];

        Self { w1, b1, w2, b2 }
    }

    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        x.iter()
            .map(|vec| {
                let h: Vec<f32> = self
                    .w1
                    .iter()
                    .zip(&self.b1)
                    .map(|(row, b)| relu(row.iter().zip(vec).map(|(w, v)| w * v).sum::<f32>() + b))
                    .collect();

                self.w2
                    .iter()
                    .zip(&self.b2)
                    .map(|(row, b)| row.iter().zip(&h).map(|(w, h)| w * h).sum::<f32>() + b)
                    .collect()
            })
            .collect()
    }
}
