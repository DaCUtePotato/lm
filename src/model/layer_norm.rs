use std::f32;

pub struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            eps,
        }
    }

    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        x.iter()
            .map(|vec| {
                let mean = vec.iter().copied().sum::<f32>() / vec.len() as f32;
                let var = vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vec.len() as f32;

                let normed = vec
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        self.gamma[i] * ((v - mean) / (var + self.eps).sqrt()) + self.beta[i]
                    })
                    .collect();
                normed
            })
            .collect()
    }
}
