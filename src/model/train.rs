pub fn dot(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    exps.iter().map(|x| x / sum_exp).collect()
}
// logits: [seq_len][vocab_size], targets: [seq_len]
pub fn cross_entropy_loss(logits: &[Vec<f32>], targets: &[usize]) -> (f32, Vec<Vec<f32>>) {
    let mut loss = 0.0;
    let mut grads = Vec::with_capacity(logits.len());

    for (logit_vec, &target_idx) in logits.iter().zip(targets.iter()) {
        let probs = softmax(logit_vec);
        loss -= probs[target_idx].ln();

        // gradient for this timestep: probs with 1 subtracted at target
        let mut grad = probs;
        grad[target_idx] -= 1.0;

        grads.push(grad);
    }
    (loss / logits.len() as f32, grads) // mean loss and gradient
}
