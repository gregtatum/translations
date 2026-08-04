//! Encode stdin lines to space-separated SentencePiece ids via the shipping
//! `spm.rs` tokenizer (no eos). Used by the M0 ggml tokenizer-parity gate to
//! produce a reference for vocabs that have no upstream `spm_encode` golden
//! (en-ru). One output line per input line, matching the golden convention.
//!
//! Usage: `cargo run -q -p fxtranslate --example spm_encode_ids -- <path.spm> < corpus.txt`

use std::io::{self, BufRead, Write};

use fxtranslate::spm::SpmVocab;

fn main() {
    let path = std::env::args().nth(1).expect("usage: spm_encode_ids <path.spm>");
    let vocab = SpmVocab::load(&path).expect("vocab parses");

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line.expect("read line");
        let ids = vocab.encode(&line);
        let s: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        writeln!(out, "{}", s.join(" ")).expect("write");
    }
}
