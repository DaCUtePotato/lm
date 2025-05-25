// In this file we are splitting the dataset into the training and control data.
// We are also going to squish the Vocab saving code here because yes :3
use rand::rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use std::io::{BufReader, BufWriter, Read, Write};

pub fn save_vocab(vocab: &HashMap<char, usize>, filename: &str) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    // collect and sort by index such that the vocab file doesn't start at a random number
    let mut entries: Vec<(&char, &usize)> = vocab.iter().collect();
    entries.sort_by_key(|&(_, idx)| idx);
    for (ch, idx) in entries {
        writeln!(writer, "{}\t{}", ch, idx)?;
    }
    Ok(())
}

pub fn load_vocab(filename: &str) -> HashMap<char, usize> {
    let file = File::open(filename).expect("Failed to open vocab file");
    let reader = BufReader::new(file);
    let mut vocab = HashMap::new();

    for line_result in reader.lines() {
        if let Ok(line) = line_result {
            let mut parts = line.split('\t');
            if let (Some(token), Some(idx_str)) = (parts.next(), parts.next()) {
                if let (Some(ch), Ok(idx)) = (token.chars().next(), idx_str.parse::<usize>()) {
                    vocab.insert(ch, idx);
                }
            }
        }
    }

    vocab
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

pub fn load_token_ids_bin(path: &str) -> std::io::Result<Vec<usize>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let tokens = buffer
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        .collect();
    Ok(tokens)
}

pub fn chunk_data(data: &[usize], max_len: usize) -> Vec<Vec<usize>> {
    data.chunks(max_len).map(|chunk| chunk.to_vec()).collect()
}

fn main() {}
