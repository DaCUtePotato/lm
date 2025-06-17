
#![allow(unused)]
#![allow(dead_code)]
#![allow(deprecated)]

// Module declarations: dataset loading, model components, tokenizer, and transformer model.
mod dataset;
mod model;
mod tokenizer;
mod transformermodel;

use model::output::sample;
use std::collections::HashMap;
use std::fs::read_to_string;
use std::io::{stdout, Write};
use std::path::Path;
use transformermodel::TransformerModel;

use dataset::{chunk_data, load_token_ids_bin, load_vocab, save_vocab, split_dataset};
use model::embedding::Embedding;
use model::output::OutputProjection;
use model::train::*;
use model::transformer_block::TransformerBlock;
use tokenizer::build_char_vocab;

// Constants for gradient clipping to avoid "gradient explosion"
// Something biology likes to do in cells (cytokine storms), but here we prevent that.
const CLIP_GRADS: bool = true;
const MAX_GRAD_NORM: f32 = 5.0;

// Simple gradient clipping function: scales gradients if their norm exceeds max_norm.
fn clip_gradients(grads: &mut [Vec<f32>], max_norm: f32) {
    for g in grads.iter_mut() {
        // Compute L2 norm of gradient vector
        let norm = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > max_norm {
            // Scale down to max_norm proportionally
            for val in g.iter_mut() {
                *val *= max_norm / norm;
            }
        }
    }
}

fn main() {
    // Training flag
    let train = true;

    // Hyperparameters: attention heads, learning rate, vocab min frequency, etc.
    let num_heads = 4;
    let learning_rate = 1e-5;
    let min_freq = 10000;
    let max_len = 32; // maximum sequence length
    let embedding_dim = 16; // embedding vector size (smallish — chemistry likes smaller atoms)

    // Load dataset file as a string
    let dataset = read_to_string("dataset.txt").expect("Failed to read dataset");
    let lines: Vec<&str> = dataset.lines().collect();
    let vocab_filename = "vocab.txt";

    // Split dataset into training and validation sets (90% train)
    let (train_set, _val_set) = split_dataset(&lines, 0.9);
    let joined_train = train_set.join("\n");

    // Sampling parameters for generation
    let temperature = 0.6; // softmax temperature, lower = more confident
    let top_k = 10; // top-k sampling filter

    // Load or build vocabulary mapping char → index, caching to file
    let vocab: HashMap<char, usize> = if !Path::new(vocab_filename).exists() {
        let v = build_char_vocab(&lines, min_freq);
        save_vocab(&v, vocab_filename).unwrap();
        v
    } else {
        load_vocab(vocab_filename)
    };

    // Reverse mapping index → char for decoding output tokens back to text
    let vocab_size = vocab.len();
    let mut index_to_char = vec!['?'; vocab_size];
    for (ch, idx) in &vocab {
        if *idx < index_to_char.len() {
            index_to_char[*idx] = *ch;
        }
    }

    // Check if pre-tokenised binary tokens exist (Python preprocessing recommended)
    if !Path::new("tokens.bin").exists() {
        println!("Please run the Python script to pre-tokenise the dataset.");
        return;
    }

    // Initialise transformer model: vocab size, embedding dim, sequence length, attention heads
    let mut model = TransformerModel::new(vocab_size, embedding_dim, max_len, num_heads);
    println!(
        "Initialized model: Embedding dim={}, Vocab size={}, Max length={}",
        embedding_dim, vocab_size, max_len
    );

    // === Training loop ===
    if train {
        // Load pre-tokenised data
        let token_ids = load_token_ids_bin("tokens.bin").expect("Failed to load tokens");

        // Split tokens into chunks (batches) of max_len for training sequences
        let batches = chunk_data(&token_ids, max_len);
        println!("Loaded {} batches of training data", batches.len());

        // Exponential moving average of loss to smooth loss reporting
        let mut ema_loss = 1.0;

        // Loop over epochs
        for epoch in 0..10000 {
            for (batch_num, batch) in batches.iter().enumerate() {
                // Prepare input and target sequences: predict next token
                let input = &batch[..batch.len() - 1];
                let target = &batch[1..];

                // Forward pass: embed tokens, apply transformer, get logits
                let (embedded, transformed, logits) = model.forward(input);

                // Compute cross-entropy loss between predictions and targets
                let loss = cross_entropy_loss(&logits, target);

                // Update EMA loss for smoother reporting
                ema_loss = 0.98 * ema_loss + 0.02 * loss;

                // Print training progress in-place
                print!(
                    "\rEpoch: {}, Batch: {}/{} | Loss: {:.4}, EMA: {:.4}",
                    epoch,
                    batch_num + 1,
                    batches.len(),
                    loss,
                    ema_loss
                );
                stdout().flush().unwrap();

                // Check for infinite loss which usually signals gradient explosion
                if loss.is_infinite() {
                    println!("\nInfinite loss — gradient explosion? Biology-level disaster.");
                    return;
                }

                // Occasionally print sample predictions for sanity check
                if loss < 1e-4 || (batch_num % 500 == 0 && epoch % 10 == 0) {
                    println!("\nSample Prediction:");
                    let predicted: String = logits
                        .iter()
                        .map(|logit| {
                            // Pick index of max logit per token
                            let idx = logit
                                .iter()
                                .enumerate()
                                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                .unwrap()
                                .0;
                            index_to_char[idx]
                        })
                        .collect();

                    // Decode input tokens to chars for display
                    let input_str: String = input.iter().map(|&idx| index_to_char[idx]).collect();
                    println!("Input:  {}", input_str);
                    println!("Output: {}", predicted);
                }

                // Backward pass: compute gradients on logits
                let mut grad_logits = cross_entropy_grad(&logits, target);

                // Propagate gradients through the model
                model.backward(input, &mut grad_logits, &transformed, &embedded);

                // Update model parameters with learning rate
                model.step(learning_rate);
            }
        }
    }

    // === Manual inference/testing section ===
    let example = "Entropie ";
    // Tokenise example input string
    let token_ids: Vec<usize> = example
        .chars()
        .filter_map(|ch| vocab.get(&ch).copied())
        .collect();

    println!("\n--- Manual Inference ---");
    // Run model forward pass
    let (embedded, transformed, logits) = model.forward(&token_ids);

    // Decode predictions by picking highest logit index for each token
    let predicted: String = logits
        .iter()
        .map(|logit| {
            let idx = logit
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            index_to_char[idx]
        })
        .collect();

    println!("Input:  {}", example);
    println!("Output: {}", predicted);
}
