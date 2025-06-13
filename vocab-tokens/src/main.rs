
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

        println!("Builded vocab");

        let mut vocab_txt = String::new();

        let (ch, mut idx): (Vec<char>, Vec<usize>) = vocab.iter().unzip();

        //idx.sort(); // same with the sorting here (if you don't know what i'm talking about look
        //down to the next comment) future me here: nvm it didn't change shit

        for (ch, idx) in ch.iter().zip(idx.iter()) {
            vocab_txt.push_str(&format!("{}\t{}\n", ch, idx));
        }

        println!("Wrote the vocab");

        fs::write("vocab.txt", vocab_txt).unwrap();

        let mut token_ids = tokenize(content, vocab);

        println!("Tokenized dataset");

        //token_ids.sort(); // so that it is sorted, but i think that messes with the learning, as
        //the batches will all have the same letter at the bininging so that means it will learn to
        //spam a letter/token..           future me: same as above

        //token_ids.dedup(); // I put this here so that it would be faster and also easier to see
        //what was going on (definitely knew that it would make our lm not learn)         future
        //me: same as above

        let mut buffer = Vec::new(); // Acts like a "binary string"

        for token in token_ids {
            buffer.write_u32::<LittleEndian>(token).unwrap();
        }

        println!("Wrote the tokens to the buffer");

        fs::write("tokens.bin", buffer).unwrap();

        println!("done");
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
        if count >= MIN_FREQ && ch != '\n' {
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
