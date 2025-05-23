// In this file we are splitting the dataset into the training and control data.
// We are also going to squish the Vocab saving code here because yes :3
use rand::rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use std::io::{BufReader, BufWriter, Write};

pub fn save_vocab(vocab: &HashMap<char, usize>, filename: &str) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    for (ch, idx) in vocab {
        writeln!(writer, "{}\t{}", ch, idx)?;
    }
    Ok(())
}

pub fn load_vocab(filename: &str) -> std::io::Result<HashMap<String, usize>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut vocab = HashMap::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let mut parts = line.split('\t');
        if let (Some(token), Some(idx_str)) = (parts.next(), parts.next()) {
            if let Ok(idx) = idx_str.parse::<usize>() {
                vocab.insert(token.to_string(), idx);
            }
        }
    }
    Ok(vocab)
}

pub fn split_dataset<'a>(lines: &'a [&str], train_ratio: f32) -> (Vec<&'a str>, Vec<&'a str>) {
    let mut rng = rng();
    let mut shuffled = lines.to_vec();
    shuffled.shuffle(&mut rng);

    let split_index = (train_ratio * lines.len() as f32).round() as usize;
    let train_set = shuffled[..split_index].to_vec();
    let val_set = shuffled[split_index..].to_vec();
    (train_set, val_set)
}

fn main() {}
