#![feature(portable_simd)]

// use packed_simd::f64x4;
use std::simd::{f64x4, num::SimdFloat};

use rand::Rng;
use std::time::Instant;

fn sum_f64_simd_1(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = f64x4::splat(0.0);
    for i in (0..a.len()).step_by(4) {
        let a = f64x4::from_slice(&a[i..]);
        let b = f64x4::from_slice(&b[i..]);
        sum += a * b;
    }
    sum.reduce_sum()
}

fn sum_f64_simd(values: &[f64]) -> f64 {
    let chunks = values.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut total = f64x4::splat(0.0);

    for chunk in chunks {
        total += f64x4::from_slice(chunk);
    }

    let mut sum = total.reduce_sum();

    for &value in remainder {
        sum += value;
    }

    sum
}

fn dot_product_naive(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn main() {
    // let n = 1_000_000_000_000i64;
    let n = 1_000_000_0i64;
    let mut rng = rand::rng();

    let a: Vec<f64> = (0..n).map(|_| rng.random()).collect();
    let b: Vec<f64> = (0..n).map(|_| rng.random()).collect();

    // === Naive ===
    let start = Instant::now();
    let result_naive = dot_product_naive(&a, &b);
    let duration_naive = start.elapsed();
    println!(
        "Naive dot product: {:.4}, Time: {:?}",
        result_naive, duration_naive
    );

    // === SIMD Dot Product ===
    let start = Instant::now();
    let result_simd = sum_f64_simd_1(&a, &b);
    let duration_simd = start.elapsed();
    println!(
        "SIMD dot product:  {:.4}, Time: {:?}",
        result_simd, duration_simd
    );

    // === SIMD Sum on A ===
    let start = Instant::now();
    let result_sum = sum_f64_simd(&a);
    let duration_sum = start.elapsed();
    println!(
        "SIMD sum of A:     {:.4}, Time: {:?}",
        result_sum, duration_sum
    );
}
