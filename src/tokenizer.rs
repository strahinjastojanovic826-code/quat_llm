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

    pub fn with_special_tokens(
        unk_token: &str,
        unk_id: usize,
        bos_id: Option<usize>,
        eos_id: Option<usize>,
        pad_id: Option<usize>,
    ) -> Self {
        let mut tok = Self::new();
        tok.vocab.clear();
        tok.inv_vocab.clear();
        tok.add_token(unk_token, unk_id);
        tok.unk_id = unk_id;
        tok.bos_id = bos_id;
        tok.eos_id = eos_id;
        let _ = pad_id;
        tok
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Greedy longest-prefix subword matching inside each whitespace token,
    /// falling back to `unk_id` when no prefix matches.
    pub fn encode_subword(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<usize> {
        let mut tokens = Vec::new();
        if add_bos {
            if let Some(bos) = self.bos_id {
                tokens.push(bos);
            }
        }

        let normalized = text
            .replace('.', " .")
            .replace(',', " ,")
            .replace('!', " !")
            .replace('?', " ?")
            .replace(':', " :")
            .replace(';', " ;");

        for word in normalized.split_whitespace() {
            if let Some(&exact_id) = self.vocab.get(word) {
                tokens.push(exact_id);
                continue;
            }

            let mut idx = 0usize;
            while idx < word.len() {
                let mut matched: Option<(usize, usize)> = None;
                for end in (idx + 1..=word.len()).rev() {
                    if !word.is_char_boundary(end) {
                        continue;
                    }
                    let sub = &word[idx..end];
                    if let Some(&id) = self.vocab.get(sub) {
                        matched = Some((end, id));
                        break;
                    }
                }

                match matched {
                    Some((next_idx, id)) => {
                        tokens.push(id);
                        idx = next_idx;
                    }
                    None => {
                        tokens.push(self.unk_id);
                        let mut chars = word[idx..].char_indices();
                        chars.next();
                        idx = chars
                            .next()
                            .map(|(offset, _)| idx + offset)
                            .unwrap_or(word.len());
                    }
                }
            }
        }

        if add_eos {
            if let Some(eos) = self.eos_id {
                tokens.push(eos);
            }
        }

        tokens
    }

    /// Decodes token IDs into readable text while skipping BOS/EOS and cleaning punctuation spacing.
    pub fn decode_clean(&self, tokens: &[usize], skip_special: bool) -> String {
        let mut parts = Vec::new();
        for &id in tokens {
            if skip_special && (Some(id) == self.bos_id || Some(id) == self.eos_id) {
                continue;
            }
            let piece = self
                .inv_vocab
                .get(&id)
                .cloned()
                .unwrap_or_else(|| "<UNK>".to_string());
            parts.push(piece);
        }

        parts
            .join(" ")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" :", ":")
            .replace(" ;", ";")
    }

    /// Serializes `id\ttoken` lines to a file for simple vocab persistence.
    pub fn save_tsv(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        let mut entries: Vec<(usize, &String)> =
            self.inv_vocab.iter().map(|(k, v)| (*k, v)).collect();
        entries.sort_by_key(|(id, _)| *id);
        for (id, tok) in entries {
            writeln!(file, "{}\t{}", id, tok)?;
        }
        Ok(())
    }

    /// Loads `id\ttoken` lines from a file.
    pub fn load_tsv(path: &std::path::Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut tokenizer = Self::new();
        tokenizer.vocab.clear();
        tokenizer.inv_vocab.clear();
        for line in content.lines() {
            if let Some((id_str, tok)) = line.split_once('\t') {
                if let Ok(id) = id_str.parse::<usize>() {
                    tokenizer.add_token(tok, id);
                }
            }
        }
        Ok(tokenizer)
    }
    
}