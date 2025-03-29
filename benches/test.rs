
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn simple(c: &mut Criterion) {
    c.bench_function("sum", |b| {
        let v: Vec<u64> = (0..1000).collect();
        b.iter(|| black_box(v.iter().sum::<u64>()))
    });
}

criterion_group!(benches, simple);
criterion_main!(benches);
