pub struct MultiHeadAttention {
    pub embed_dim: usize,
    // Todo: number of heads, key/query/value matrices etc.
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize) -> Self {
        Self { embed_dim }
    }

    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Dummy function, will just return the input for now :\
        x.to_vec()
    }
}
