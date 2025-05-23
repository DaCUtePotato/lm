mod tokenizer;
use std::fs::read_to_string;
use tokenizer::{build_char_vocab, tokenize_char_level};
mod dataset;
use dataset::{load_vocab, save_vocab, split_dataset};
use std::path::Path;

fn main() {
    let min_freq = 10000;
    let dataset = read_to_string("dataset.txt").expect("Failed to read Dataset");
    let lines: Vec<&str> = dataset.lines().collect();
    let vocab_filename = "vocab.txt";

    let (train_set, val_set) = split_dataset(&lines, 0.9);

    let joined_train = train_set.join("\n");
    let joined_val = val_set.join("\n");

    if !Path::new(vocab_filename).exists() {
        let vocab = build_char_vocab(&lines, min_freq);
        println!("New Vocab: {:?}", vocab);
        let result = save_vocab(&vocab, vocab_filename);
        println!("Aaaand the result is: {:?}", result)
    } else {
        let vocab = load_vocab(vocab_filename);
        println!("Saved vocab is: {:?}", vocab)
    };
}
