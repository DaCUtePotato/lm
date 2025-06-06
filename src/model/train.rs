pub fn cross_entropy_loss(logits: &Vec<Vec<f32>>, targets: &[usize]) -> f32 {
    let mut loss = 0.0;
    for (logit, &target) in logits.iter().zip(targets.iter()) {
        let max_logit = logit.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logit.iter().map(|x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp.iter().sum();
        let log_probs: Vec<f32> = exp.iter().map(|x| x.ln() - sum_exp.ln()).collect();
        loss -= log_probs[target];
    }
    loss / logits.len() as f32
}

// Softmax over last dimension
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|x| x / sum).collect()
}

// Gradient of CE loss w.r.t. logits
pub fn cross_entropy_grad(logits: &Vec<Vec<f32>>, targets: &[usize]) -> Vec<Vec<f32>> {
    logits
        .iter()
        .zip(targets.iter())
        .map(|(logit, &target)| {
            let mut probs = softmax(logit);
            probs[target] -= 1.0; // dL/dz = softmax - 1 at the target index
            probs
        })
        .collect()
}
