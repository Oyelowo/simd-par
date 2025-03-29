#![feature(portable_simd)]

use aligned::A32;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;
use std::simd::{f32x8, f64x4, num::SimdFloat};
// use ::aligned::{Aligned};
//
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm_prefetch;

// unsafe {
//     _mm_prefetch(ptr.add(i + 128) as *const i8, _MM_HINT_T0);
// }


use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::marker::PhantomData;
use std::mem;
use std::slice;

/// A heap-allocated vector aligned to `ALIGN` bytes.
pub struct AlignedVec<T, const ALIGN: usize> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
    _phantom: PhantomData<T>,
}

impl<T, const ALIGN: usize> AlignedVec<T, ALIGN> {
    pub fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len * mem::size_of::<T>(), ALIGN)
            .expect("Invalid layout");
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }

        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut T) },
            len,
            layout,
            _phantom: PhantomData,
        }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T, const ALIGN: usize> Drop for AlignedVec<T, ALIGN> {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

impl<T, const ALIGN: usize> std::ops::Deref for AlignedVec<T, ALIGN> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, const ALIGN: usize> std::ops::DerefMut for AlignedVec<T, ALIGN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}




// use core::arch::x86_64::*
fn generate_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::rng();
    let a: Vec<f64> = (0..n).map(|_| rng.random()).collect();
    let b: Vec<f64> = (0..n).map(|_| rng.random()).collect();
    (a, b)
}

type SimdAlignedVec = AlignedVec<f32, 32>;
// Aligned SIMD-safe data
// fn generate_data_f32_aligned(n: usize) -> (AlignedVec<f32, A32>, AlignedVec<f32, A32>) {
fn generate_data_f32_aligned(n: usize) -> (SimdAlignedVec, SimdAlignedVec) {
    let mut rng = rand::thread_rng();
    let mut a = AlignedVec::new(n);
    let mut b = AlignedVec::new(n);
    for i in 0..n {
        a[i] = rng.r#gen();
        b[i] = rng.r#gen();
    }
    (a, b)
}



fn dot_product_naive(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn sum_f64_simd_1(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = f64x4::splat(0.0);
    let chunks = a.len() / 4 * 4;

    for i in (0..chunks).step_by(4) {
        let va = f64x4::from_slice(&a[i..]);
        let vb = f64x4::from_slice(&b[i..]);
        sum += va * vb;
    }

    let mut remainder = 0.0;
    for i in chunks..a.len() {
        remainder += a[i] * b[i];
    }

    sum.reduce_sum() + remainder
}

fn simd_sum(values: &[f64]) -> f64 {
    let chunks = values.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut acc = f64x4::splat(0.0);
    for chunk in chunks {
        acc += f64x4::from_slice(chunk);
    }

    let mut total = acc.reduce_sum();
    for &val in remainder {
        total += val;
    }

    total
}

use rayon::prelude::*;

// Parallel SIMD dot product
fn dot_product_simd_rayon(a: &[f64], b: &[f64]) -> f64 {
    a.par_chunks(1024)
        .zip(b.par_chunks(1024))
        .map(|(chunk_a, chunk_b)| {
            let len = chunk_a.len().min(chunk_b.len());
            let mut sum = f64x4::splat(0.0);
            let simd_chunks = len / 4 * 4;

            for i in (0..simd_chunks).step_by(4) {
                let va = f64x4::from_slice(&chunk_a[i..]);
                let vb = f64x4::from_slice(&chunk_b[i..]);
                sum += va * vb;
            }

            let mut remainder = 0.0;
            for i in simd_chunks..len {
                remainder += chunk_a[i] * chunk_b[i];
            }

            sum.reduce_sum() + remainder
        })
        .sum()
}

fn dot_product_parallel_rayon_no_simd(a: &[f64], b: &[f64]) -> f64 {
    a.par_iter().zip(b.par_iter()).map(|(x, y)| x * y).sum()
}

fn sum_f64_simd_unrolled(a: &[f64], b: &[f64]) -> f64 {
    let mut sum0 = f64x4::splat(0.0);
    let mut sum1 = f64x4::splat(0.0);
    let mut sum2 = f64x4::splat(0.0);
    let mut sum3 = f64x4::splat(0.0);

    let chunks = a.len() / 16 * 16;
    let mut i = 0;

    while i < chunks {
        sum0 += f64x4::from_slice(&a[i..]) * f64x4::from_slice(&b[i..]);
        sum1 += f64x4::from_slice(&a[i + 4..]) * f64x4::from_slice(&b[i + 4..]);
        sum2 += f64x4::from_slice(&a[i + 8..]) * f64x4::from_slice(&b[i + 8..]);
        sum3 += f64x4::from_slice(&a[i + 12..]) * f64x4::from_slice(&b[i + 12..]);
        i += 16;
    }

    let mut remainder = 0.0;
    for j in i..a.len() {
        remainder += a[j] * b[j];
    }

    (sum0 + sum1 + sum2 + sum3).reduce_sum() + remainder
}

fn dot_product_simd_rayon_unrolled_x4(a: &[f64], b: &[f64]) -> f64 {
    a.par_chunks(1024)
        .zip(b.par_chunks(1024))
        .map(|(chunk_a, chunk_b)| {
            let len = chunk_a.len().min(chunk_b.len());
            let simd_chunks = len / 16 * 16;

            let mut sum0 = f64x4::splat(0.0);
            let mut sum1 = f64x4::splat(0.0);
            let mut sum2 = f64x4::splat(0.0);
            let mut sum3 = f64x4::splat(0.0);

            let mut i = 0;
            while i < simd_chunks {
                sum0 += f64x4::from_slice(&chunk_a[i..]) * f64x4::from_slice(&chunk_b[i..]);
                sum1 += f64x4::from_slice(&chunk_a[i + 4..]) * f64x4::from_slice(&chunk_b[i + 4..]);
                sum2 += f64x4::from_slice(&chunk_a[i + 8..]) * f64x4::from_slice(&chunk_b[i + 8..]);
                sum3 +=
                    f64x4::from_slice(&chunk_a[i + 12..]) * f64x4::from_slice(&chunk_b[i + 12..]);
                i += 16;
            }

            let mut remainder = 0.0;
            for j in i..len {
                remainder += chunk_a[j] * chunk_b[j];
            }

            (sum0 + sum1 + sum2 + sum3).reduce_sum() + remainder
        })
        .sum()
}

fn generate_data32(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::rng();
    let a: Vec<f32> = (0..n).map(|_| rng.random()).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.random()).collect();
    (a, b)
}
fn dot_naive(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = f32x8::splat(0.0);
    let chunks = a.len() / 8 * 8;
    for i in (0..chunks).step_by(8) {
        unsafe {
            let va = f32x8::from_slice(&a[i..]);
            let vb = f32x8::from_slice(&b[i..]);
            acc += va * vb;
        }
    }

    let mut tail = 0.0;
    for i in chunks..a.len() {
        tail += a[i] * b[i];
    }

    acc.reduce_sum() + tail
}

fn dot_simd_unrolled(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = f32x8::splat(0.0);
    let mut s1 = f32x8::splat(0.0);
    let mut s2 = f32x8::splat(0.0);
    let mut s3 = f32x8::splat(0.0);

    let chunks = a.len() / 32 * 32;
    let mut i = 0;
    while i < chunks {
        unsafe {
            s0 += f32x8::from_slice(&a[i..]) * f32x8::from_slice(&b[i..]);
            s1 += f32x8::from_slice(&a[i + 8..]) * f32x8::from_slice(&b[i + 8..]);
            s2 += f32x8::from_slice(&a[i + 16..]) * f32x8::from_slice(&b[i + 16..]);
            s3 += f32x8::from_slice(&a[i + 24..]) * f32x8::from_slice(&b[i + 24..]);
        }
        i += 32;
    }

    let mut tail = 0.0;
    for j in i..a.len() {
        tail += a[j] * b[j];
    }

    (s0 + s1 + s2 + s3).reduce_sum() + tail
}

fn dot_simd_rayon(a: &[f32], b: &[f32]) -> f32 {
    a.par_chunks(8192)
        .zip(b.par_chunks(8192))
        .map(|(ca, cb)| dot_simd(ca, cb))
        .sum()
}

fn dot_simd_rayon_unrolled(a: &[f32], b: &[f32]) -> f32 {
    a.par_chunks(8192)
        .zip(b.par_chunks(8192))
        .map(|(ca, cb)| dot_simd_unrolled(ca, cb))
        .sum()
}

fn dot_rayon_naive(a: &[f32], b: &[f32]) -> f32 {
    a.par_iter().zip(b.par_iter()).map(|(x, y)| x * y).sum()
}

fn simd_benchmark(c: &mut Criterion) {
    let size = 1_000_000_000;
    let (a, b) = generate_data(size);

    // c.bench_function("naive dot", |bench| {
    //     bench.iter(|| dot_product_naive(black_box(&a), black_box(&b)))
    // });

    c.bench_function("SIMD dot", |bench| {
        bench.iter(|| sum_f64_simd_1(black_box(&a), black_box(&b)))
    });

    c.bench_function("SIMD sum A", |b| b.iter(|| simd_sum(black_box(&a))));

    c.bench_function("SIMD + Rayon dot No Simd", |bench| {
        bench.iter(|| dot_product_parallel_rayon_no_simd(black_box(&a), black_box(&b)))
    });

    c.bench_function("SIMD + Rayon dot with Simd", |bench| {
        bench.iter(|| dot_product_simd_rayon(black_box(&a), black_box(&b)))
    });

    c.bench_function("SIMD dot unrolled x4", |bench| {
        bench.iter(|| sum_f64_simd_unrolled(black_box(&a), black_box(&b)))
    });

    c.bench_function("SIMD + Rayon dot unrolled x4", |bench| {
        bench.iter(|| dot_product_simd_rayon_unrolled_x4(black_box(&a), black_box(&b)))
    });

    let size = 100_000_000; // 100 million f32s = 400MB × 2
    let (a, b) = generate_data32(size);
    c.bench_function("Naive f32 dot", |bench| {
        bench.iter(|| dot_naive(black_box(&a), black_box(&b)))
    });
    c.bench_function("SIMD f32 dot", |bench| {
        bench.iter(|| dot_simd(black_box(&a), black_box(&b)))
    });
    c.bench_function("SIMD f32 dot unrolled x4", |bench| {
        bench.iter(|| dot_simd_unrolled(black_box(&a), black_box(&b)))
    });
    c.bench_function("Rayon f32 dot naive", |bench| {
        bench.iter(|| dot_rayon_naive(black_box(&a), black_box(&b)))
    });
    c.bench_function("Rayon + SIMD f32 dot", |bench| {
        bench.iter(|| dot_simd_rayon(black_box(&a), black_box(&b)))
    });
    c.bench_function("Rayon + SIMD f32 dot unrolled x4", |bench| {
        bench.iter(|| dot_simd_rayon_unrolled(black_box(&a), black_box(&b)))
    });



    let (a, b) = generate_data_f32_aligned(size);
    c.bench_function("Naive f32 aligned dot", |bench| {
        bench.iter(|| dot_naive(black_box(&a), black_box(&b)))
    });
    c.bench_function("SIMD f32 aligned dot", |bench| {
        bench.iter(|| dot_simd(black_box(&a), black_box(&b)))
    });
    c.bench_function("SIMD f32 aligned dot unrolled x4", |bench| {
        bench.iter(|| dot_simd_unrolled(black_box(&a), black_box(&b)))
    });
    c.bench_function("Rayon f32 aligned dot naive", |bench| {
        bench.iter(|| dot_rayon_naive(black_box(&a), black_box(&b)))
    });
    c.bench_function("Rayon + SIMD f32 aligned dot", |bench| {
        bench.iter(|| dot_simd_rayon(black_box(&a), black_box(&b)))
    });
    c.bench_function("Rayon + SIMD f32 aligned dot unrolled x4", |bench| {
        bench.iter(|| dot_simd_rayon_unrolled(black_box(&a), black_box(&b)))
    });
}

criterion_group!(benches, simd_benchmark);
criterion_main!(benches);
