# simd-par

cargo bench

```csv
,Method,Precision,Execution,Time (ms)
11,Rayon + SIMD f32 dot unrolled x4,f32,parallel,7.98
16,Rayon + SIMD f32 aligned dot,f32 (aligned),parallel,7.99
17,Rayon + SIMD f32 aligned dot unrolled x4,f32 (aligned),parallel,8.59
10,Rayon + SIMD f32 dot,f32,parallel,9.36
14,SIMD f32 aligned dot unrolled x4,f32 (aligned),single,14.32
8,SIMD f32 dot unrolled x4,f32,single,14.54
7,SIMD f32 dot,f32,single,16.44
13,SIMD f32 aligned dot,f32 (aligned),single,16.45
15,Rayon f32 aligned dot naive,f32 (aligned),parallel,16.9
9,Rayon f32 dot naive,f32,parallel,17.5
6,Naive f32 dot,f32,single,125.25
12,Naive f32 aligned dot,f32 (aligned),single,126.28
5,SIMD + Rayon dot unrolled x4,f64,parallel,151.47
3,SIMD + Rayon dot with Simd,f64,parallel,157.98
2,SIMD + Rayon dot No Simd,f64,parallel,201.0
4,SIMD dot unrolled x4,f64,single,285.66
1,SIMD sum A,f64,single,312.79
0,SIMD dot,f64,single,327.03
```

### Benchmark Results (1,000,000,000 elements unless noted otherwise)

### **f64 Benchmarks** (baseline = Naive f64 ≈ 1230ms from your 1B result)

| Method                          | Time (ms) | Speedup × | Notes                        |
|--------------------------------|-----------|-----------|------------------------------|
| Naive f64 dot (baseline)       | 1230      | 1.00×     | Scalar reference             |
| SIMD dot                       | 327.03    | 3.76×     | Portable SIMD (f64x4)        |
| SIMD sum A                     | 312.79    | 3.93×     | Just sum, not dot            |
| SIMD dot unrolled x4           | 285.66    | 4.31×     | Unrolled inner loop          |
| SIMD + Rayon dot (no SIMD)     | 201.00    | 6.12×     | Parallel scalar              |
| SIMD + Rayon dot               | 157.98    | 7.79×     | Parallel + SIMD              |
| SIMD + Rayon dot unrolled x4   | 151.47    | 8.12×     | Parallel + SIMD + Unroll     |

---

### **f32 Benchmarks** (baseline = Naive f32 ≈ 125ms)

#### Non-Aligned

| Method                          | Time (ms) | Speedup × | Notes                        |
|--------------------------------|-----------|-----------|------------------------------|
| Naive f32 dot (baseline)       | 125.25    | 1.00×     | Scalar reference             |
| SIMD f32 dot                   | 16.44     | 7.62×     | Portable SIMD (f32x8)        |
| SIMD f32 dot unrolled x4       | 14.54     | 8.61×     | Manual unrolling             |
| Rayon f32 dot naive            | 17.50     | 7.16×     | Parallel scalar              |
| Rayon + SIMD f32 dot           | 9.36      | 13.38×    | Parallel + SIMD              |
| Rayon + SIMD f32 dot unrolled x4 | 7.98    | 15.70×    | Best speedup overall         |

---

#### Aligned

| Method                              | Time (ms) | Speedup × | Notes                        |
|------------------------------------|-----------|-----------|------------------------------|
| Naive f32 aligned dot              | 126.28    | 1.00×     | Scalar, aligned buffer       |
| SIMD f32 aligned dot               | 16.45     | 7.68×     | SIMD aligned                 |
| SIMD f32 aligned dot unrolled x4   | 14.32     | 8.82×     | SIMD aligned, unrolled       |
| Rayon f32 aligned dot naive        | 16.90     | 7.47×     | Parallel scalar              |
| Rayon + SIMD f32 aligned dot       | 7.99      | 15.80×    | Parallel + SIMD              |
| Rayon + SIMD f32 aligned dot x4    | 8.59      | 14.70×    | Slightly slower than SIMD x1 |

---

### Summary

| Category             | Winner                         | Speedup |
|----------------------|----------------------------------|---------|
| f64 Overall Best     | SIMD + Rayon unrolled x4         | 8.12×   |
| f32 Overall Best     | Rayon + SIMD f32 dot unrolled x4 | 15.80×  |

