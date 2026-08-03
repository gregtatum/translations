//! Microbenchmark the int8 GEMM kernel (gemmology on aarch64) at the exact decoder and
//! encoder shapes, to isolate kernel efficiency from the rest of the engine — the decision
//! gate in notes/17 for whether the gap to ORT/MLAS is the kernel or the graph.
//!
//! Run: `cargo run --release --example kernel_micro --features fast -p fxtranslate`
//!
//! Reports GFLOP/s (2·m·k·n / time) per shape. NOTE: `PreparedB::matmul_into` includes the
//! dequant+bias epilogue, whereas the ORT `MatMulInteger` microbench (onnx/kernel_micro.py)
//! is the raw integer matmul — so compare with that epilogue caveat in mind.

use std::time::Instant;

use fxtranslate::gemm::PreparedB;

// Cheap deterministic fill (no rand dep) — values in a small int8 range.
fn lcg_fill_i8(buf: &mut [i8], mut s: u64) {
    for b in buf.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = ((s >> 56) as i8) / 2; // roughly [-64, 63]
    }
}
fn lcg_fill_u8(buf: &mut [u8], mut s: u64) {
    for b in buf.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (s >> 56) as u8;
    }
}

fn bench(m: usize, k: usize, n: usize) {
    // B is stored transposed [n, k] (row-major n×k), as PreparedB::new expects.
    let mut b = vec![0i8; n * k];
    lcg_fill_i8(&mut b, 0x1234 ^ ((n as u64) << 20) ^ k as u64);
    let pb = match PreparedB::new(&b, n, k) {
        Some(pb) => pb,
        None => {
            println!("  m={m:<4} k={k} n={n:<6}  (no packed kernel — scalar build)");
            return;
        }
    };
    let mut a = vec![0u8; m * k];
    lcg_fill_u8(&mut a, 0xabcd ^ (m as u64));
    let bias = vec![0.0f32; n];
    let mut out = Vec::new();

    // Warm up, then time a fixed wall-clock budget and count iterations.
    for _ in 0..8 {
        pb.matmul_into(&a, m, 1.0, &bias, &mut out);
    }
    let budget = std::time::Duration::from_millis(400);
    let start = Instant::now();
    let mut iters = 0u64;
    while start.elapsed() < budget {
        for _ in 0..16 {
            pb.matmul_into(&a, m, 1.0, &bias, &mut out);
        }
        iters += 16;
    }
    let secs = start.elapsed().as_secs_f64();
    let per = secs / iters as f64;
    let gflops = (2.0 * m as f64 * k as f64 * n as f64) / per / 1e9;
    println!(
        "  m={m:<4} k={k} n={n:<6}  {per_us:>9.2} us/matmul  {gflops:>7.1} GFLOP/s",
        per_us = per * 1e6
    );
    std::hint::black_box(&out);
}

fn main() {
    println!("gemmology int8 kernel (matmul_into: int matmul + dequant + bias)\n");
    println!("decoder shapes (small m):");
    for &m in &[1usize, 2, 4] {
        bench(m, 512, 512); // attn Q/K/V/O, SSRU, FFN-out
        bench(m, 512, 2048); // FFN W1
        bench(m, 512, 32000); // tied output projection
    }
    println!("\nencoder shapes (large m = batch*seq):");
    for &m in &[64usize, 256] {
        bench(m, 512, 512);
        bench(m, 512, 2048);
    }
}
