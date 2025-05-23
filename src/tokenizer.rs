use std::collections::HashMap;

pub fn build_char_vocab(corpus: &[&str], min_freq: usize) -> HashMap<char, usize> {
    let mut char_freq = HashMap::new();

    for line in corpus {
        for ch in line.chars() {
            *char_freq.entry(ch).or_insert(0) += 1;
        }
    }
    let mut vocab = HashMap::new();
    let mut idx = 0;
    for (ch, freq) in char_freq {
        if freq >= min_freq {
            vocab.insert(ch, idx);
            idx += 1;
        }
    }
    vocab
}

pub fn tokenize_char_level(
    text: &str,
    vocab: &HashMap<char, usize>,
) -> (Vec<usize>, HashMap<char, usize>) {
    let mut new_vocab = vocab.clone();
    let mut len = 0;
    let tokens = text
        .chars()
        .map(|character| {
            let c = vocab.get(&character).cloned();
            if c.is_some() {
                c.unwrap()
            } else {
                len += 1;
                let (_, max) = vocab.iter().max_by_key(|entry| entry.1).unwrap();

                new_vocab.insert(character, max + len);

                max + len
            }
        })
        .collect();

    (tokens, new_vocab)
}

//#[cfg(test)]
//mod tests {
//    use super::*;
//
//    #[test]
//    fn test_tokenizer() {
//        let corpus = [r"hello world", r"language model"];
//        let mut vocab = build_char_vocab(&corpus);
//
//        let input = "elhlo!(";
//        let (tokens, new_vocab) = tokenize_char_level(input, &vocab);
//        vocab = new_vocab;
//
//        println!("Char-level tokens: {:?}", tokens);
//    }
//}
