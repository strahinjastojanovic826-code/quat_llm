use std::collections::HashMap;

/// Robusniji Vocabulary Tokenizer sa podrškom za unk_token i specijalne znake
#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub vocab: HashMap<String, usize>,
    pub inv_vocab: HashMap<usize, String>,
    pub unk_id: usize,
    pub bos_id: Option<usize>,
    pub eos_id: Option<usize>,
}

impl Tokenizer {
    pub fn new() -> Self {
        let mut tokenizer = Tokenizer {
            vocab: HashMap::new(),
            inv_vocab: HashMap::new(),
            unk_id: 0,
            bos_id: None,
            eos_id: None,
        };
        // Podrazumevani UNK token na indeksu 0
        tokenizer.add_token("<UNK>", 0);
        tokenizer
    }

    pub fn add_token(&mut self, token: &str, id: usize) {
        self.vocab.insert(token.to_string(), id);
        self.inv_vocab.insert(id, token.to_string());
    }

    pub fn set_special_tokens(&mut self, unk_id: usize, bos_id: Option<usize>, eos_id: Option<usize>) {
        self.unk_id = unk_id;
        self.bos_id = bos_id;
        self.eos_id = eos_id;
    }

    /// Enkodovanje teksta u tokene sa osnovnim razdvajanjem interpunkcije
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();

        if let Some(bos) = self.bos_id {
            tokens.push(bos);
        }

        // Osnovno čišćenje i splitovanje koje uzima u obzir reči i interpunkciju
        let cleaned_text = text
            .replace('.', " . ")
            .replace(',', " , ")
            .replace('!', " ! ")
            .replace('?', " ? ");

        for word in cleaned_text.split_whitespace() {
            let id = self.vocab.get(word).copied().unwrap_or(self.unk_id);
            tokens.push(id);
        }

        if let Some(eos) = self.eos_id {
            tokens.push(eos);
        }

        tokens
    }

    /// Dekodovanje niza token ID-eva nazad u string
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut words = Vec::new();

        for &id in tokens {
            // Preskačemo BOS i EOS pri dekodovanju ako nisu poželjni u izlazu
            if Some(id) == self.bos_id || Some(id) == self.eos_id {
                continue;
            }

            let word = self
                .inv_vocab
                .get(&id)
                .cloned()
                .unwrap_or_else(|| "<UNK>".to_string());
            words.push(word);
        }

        words.join(" ")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace(" !", "!")
            .replace(" ?", "?")
    }
}