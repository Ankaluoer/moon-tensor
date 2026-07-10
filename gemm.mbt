///|
/// Naive GEMM: C = A @ B
/// A: (M, K) row-major, B: (K, N) row-major, C: (M, N) row-major
pub fn gemm(
  a : Array[Double],
  b : Array[Double],
  m : Int,
  k : Int,
  n : Int,
) -> Array[Double] {
  let c = Array::make(m * n, 0.0)
  for i = 0; i < m; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      let mut sum = 0.0
      for p = 0; p < k; p = p + 1 {
        sum = sum + a[i * k + p] * b[p * n + j]
      }
      c[i * n + j] = sum
    }
  }
  c
}
