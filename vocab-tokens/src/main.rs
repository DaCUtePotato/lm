
use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::HashMap;
use std::fs;

const MIN_FREQ: usize = 10000;

fn main() {
    if !fs::exists("dataset.txt").unwrap() {
        println!("You have forgotten to put the dataset into this folder.......... You stupid stupid Balatro Joker");
    } else {
        let content = fs::read_to_string("dataset.txt").unwrap();

        let vocab = build_vocab(&content);

        let mut vocab_txt = String::new();

        for (ch, idx) in &vocab {
            vocab_txt.push_str(&format!("{}\t{}\n", ch, idx));
        }

        fs::write("vocab.txt", vocab_txt).unwrap();

        let mut token_ids = tokenize(content, vocab);

        token_ids.sort();

        let mut buffer = Vec::new(); // Acts like a "binary string"

        for token in token_ids {
            buffer.write_u32::<LittleEndian>(token).unwrap();
        }

        fs::write("tokens.bin", buffer).unwrap();
    }
}

fn build_vocab(text: &str) -> HashMap<char, usize> {
    let mut freq = HashMap::new();

    // Count frequency of each character
    for ch in text.chars() {
        *freq.entry(ch).or_insert(0) += 1;
    }

    // Build vocabulary: character -> index
    let mut vocab = HashMap::new();
    let mut index = 0;

    for (ch, count) in freq {
        if count >= MIN_FREQ {
            vocab.insert(ch, index);
            index += 1;
        }
    }

    vocab
}

fn tokenize(text: String, vocab: HashMap<char, usize>) -> Vec<u32> {
    let mut output = Vec::new();

    for ch in text.chars() {
        if vocab.contains_key(&ch) {
            output.push(vocab[&ch] as u32);
        }
    }

    output
}
