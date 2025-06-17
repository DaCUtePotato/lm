#![allow(unused)]
#![allow(dead_code)]
#![allow(deprecated)]
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

const CLIP_GRADS: bool = true;
const MAX_GRAD_NORM: f32 = 5.0;

fn clip_gradients(grads: &mut [Vec<f32>], max_norm: f32) {
    for g in grads.iter_mut() {
        let norm = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > max_norm {
            for val in g.iter_mut() {
                *val *= max_norm / norm;
            }
        }
    }
}

fn main() {
    let train = true;
    let num_heads = 4;
    let learning_rate = 1e-5;
    let min_freq = 10000;
    let max_len = 32;
    let embedding_dim = 16;

    let dataset = read_to_string("dataset.txt").expect("Failed to read dataset");
    let lines: Vec<&str> = dataset.lines().collect();
    let vocab_filename = "vocab.txt";

    let (train_set, _val_set) = split_dataset(&lines, 0.9);
    let joined_train = train_set.join("\n");

    let temperature = 0.6; // tune this, e.g., 0.8, 1.0, 1.2
    let top_k = 10; // tune top-k for filtering

    let vocab: HashMap<char, usize> = if !Path::new(vocab_filename).exists() {
        let v = build_char_vocab(&lines, min_freq);
        save_vocab(&v, vocab_filename).unwrap();
        v
    } else {
        load_vocab(vocab_filename)
    };

    let vocab_size = vocab.len();
    let mut index_to_char = vec!['?'; vocab_size];
    for (ch, idx) in &vocab {
        if *idx < index_to_char.len() {
            index_to_char[*idx] = *ch;
        }
    }

    if !Path::new("tokens.bin").exists() {
        println!("Please run the Python script to pre-tokenise the dataset.");
        return;
    }

    let mut model = TransformerModel::new(vocab_size, embedding_dim, max_len, num_heads);
    println!(
        "Initialized model: Embedding dim={}, Vocab size={}, Max length={}",
        embedding_dim, vocab_size, max_len
    );

    if train {
        let token_ids = load_token_ids_bin("tokens.bin").expect("Failed to load tokens");
        let batches = chunk_data(&token_ids, max_len);
        println!("Loaded {} batches of training data", batches.len());

        let mut ema_loss = 1.0;

        for epoch in 0..10000 {
            for (batch_num, batch) in batches.iter().enumerate() {
                let input = &batch[..batch.len() - 1];
                let target = &batch[1..];

                // Forward pass
                let (embedded, transformed, logits) = model.forward(input);
                let loss = cross_entropy_loss(&logits, target);
                ema_loss = 0.98 * ema_loss + 0.02 * loss;

                print!(
                    "\rEpoch: {}, Batch: {}/{} | Loss: {:.4}, EMA: {:.4}",
                    epoch,
                    batch_num + 1,
                    batches.len(),
                    loss,
                    ema_loss
                );
                stdout().flush().unwrap();

                if loss.is_infinite() {
                    println!("\nInfinite loss — gradient explosion? Biology-level disaster.");
                    return;
                }

                // Print predictions if loss is low
                if loss < 1e-4 || (batch_num % 500 == 0 && epoch % 10 == 0) {
                    println!("\nSample Prediction:");
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
                    let input_str: String = input.iter().map(|&idx| index_to_char[idx]).collect();
                    println!("Input:  {}", input_str);
                    println!("Output: {}", predicted);
                }

                // Backward pass
                let mut grad_logits = cross_entropy_grad(&logits, target);
                model.backward(input, &mut grad_logits, &transformed, &embedded);

                // Step
                model.step(learning_rate);
            }
        }
    }

    //testing
    let example = "Entropie ";
    let token_ids: Vec<usize> = example
        .chars()
        .filter_map(|ch| vocab.get(&ch).copied())
        .collect();

    println!("\n--- Manual Inference ---");
    let (embedded, transformed, logits) = model.forward(&token_ids);

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
