use std::collections::HashMap;

/// Simple Vocabulary Tokenizer
pub struct Tokenizer {
    pub vocab: HashMap<String, usize>,
    pub inv_vocab: HashMap<usize, String>,
}

impl Tokenizer {
    pub fn new() -> Self {
        Tokenizer {
            vocab: HashMap::new(),
            inv_vocab: HashMap::new(),
        }
    }

    pub fn add_token(&mut self, token: &str, id: usize) {
        self.vocab.insert(token.to_string(), id);
        self.inv_vocab.insert(id, token.to_string());
    }

    /// Converts a space-separated text string into vector of token IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| *self.vocab.get(word).unwrap_or(&0)) // 0 = UNK token
            .collect()
    }

    /// Converts token ID back to text
    pub fn decode(&self, token_id: usize) -> String {
        self.inv_vocab
            .get(&token_id)
            .cloned()
            .unwrap_or_else(|| "<UNK>".to_string())
    }
}