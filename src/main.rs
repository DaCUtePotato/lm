mod dataset;
mod model;
mod tokenizer;

use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::Path;

use dataset::{chunk_data, load_token_ids_bin, load_vocab, save_vocab, split_dataset};
use model::embedding::Embedding;
use model::output::OutputProjection;
use model::transformer_block::TransformerBlock;
use tokenizer::build_char_vocab;

fn main() {
    // declaring vars...
    let train = true;
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

    let transformer = TransformerBlock::new(embedding_dim);
    println!("Created Transformer Block :D");

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

    // Build reverse vocab for decoding logits
    let mut index_to_char = vec!['?'; vocab.len()];
    for (ch, idx) in &vocab {
        if *idx < index_to_char.len() {
            index_to_char[*idx] = *ch;
        }
    }

    let vocab_size = vocab.len();

    if !Path::new("tokens.bin").exists() {
        println!("Please run the Python script to pre-tokenise the dataset.");
        return;
    }

    if train {
        let token_ids = load_token_ids_bin("tokens.bin").expect("Failed to load tokens");
        let batches = chunk_data(&token_ids, max_len);
        println!("Loaded the dataset...");
    }

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

    let transformed = transformer.forward(&embedded);
    println!("After Transformer Block: ");
    for (i, vector) in transformed.iter().enumerate() {
        println!("{}, {:?}", example.chars().nth(i).unwrap(), vector);
    }

    let output = OutputProjection::new(embedding_dim, vocab_size);
    let logits = output.forward(transformed);

    println!("Logits: ");
    for (i, logit_vector) in logits.iter().enumerate() {
        println!("{}, {:?}", example.chars().nth(i).unwrap(), logit_vector);
    }
    println!("Predicted Chars:");
    for (i, logit_vector) in logits.iter().enumerate() {
        if let Some((max_idx, _)) = logit_vector
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            let predicted_char = index_to_char.get(max_idx).copied().unwrap_or('?');
            println!(
                "Input: '{}', Predicted: '{}'",
                example.chars().nth(i).unwrap(),
                predicted_char
            );
        }
    }
}
