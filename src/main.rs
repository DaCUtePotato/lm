mod tokenizer;
use std::fs::read_to_string;
use tokenizer::{build_char_vocab, tokenize_char_level};
mod dataset;
use dataset::{load_vocab, save_vocab, split_dataset};
use std::path::Path;
mod model;
use model::embedding::Embedding;
use std::collections::HashMap;

fn main() {
    // declaring vars...
    let min_freq = 10000;
    let dataset = read_to_string("dataset.txt").expect("Failed to read Dataset");
    let lines: Vec<&str> = dataset.lines().collect();
    let vocab_filename = "vocab.txt";

    let max_len = 128;

    let (train_set, val_set) = split_dataset(&lines, 0.9);

    let joined_train = train_set.join("\n");
    let joined_val = val_set.join("\n");

    let embedding_dim = 16;
    let vocab: HashMap<char, usize>;

    // Building vocab/loading vocab
    if !Path::new(vocab_filename).exists() {
        vocab = build_char_vocab(&lines, min_freq);
        println!("New Vocab: {:?}", vocab);
        let result = save_vocab(&vocab, vocab_filename);
        println!("Aaaand the result is: {:?}", result)
    } else {
        vocab = load_vocab(vocab_filename);
        println!("Saved vocab is: {:?}", vocab)
    };

    let vocab_size = vocab.len();

    // Building the Embedding Layer
    let embedding = Embedding::new(vocab_size, embedding_dim, max_len);
    println!(
        "Created embedding layer with vocab size {} and dim {}",
        vocab_size, embedding_dim
    );

    //testing
    let example = "Hello World!";
    let token_ids: Vec<usize> = example
        .chars()
        .filter_map(|ch| vocab.get(&ch).copied())
        .collect();

    println!("Token IDs for '{}': {:?}", example, token_ids);
    let embedded = embedding.forward(&token_ids);
    println!("Embeddings: ");
    for (i, vector) in embedded.iter().enumerate() {
        println!("{}, {:?}", example.chars().nth(i).unwrap(), vector);
    }
}
