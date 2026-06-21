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

///|
/// Compute max absolute difference between two arrays.
/// Used for numerical correctness verification.
pub fn max_abs_diff(x : Array[Double], y : Array[Double]) -> Double {
  let mut max_diff = 0.0
  for i = 0; i < x.length(); i = i + 1 {
    let d = x[i] - y[i]
    let abs_d = if d >= 0.0 { d } else { -d }
    if abs_d > max_diff {
      max_diff = abs_d
    }
  }
  max_diff
}

///|
/// Naive 1D convolution (no padding, stride=1).
/// 
/// Input shape:  (C_in, L)        flat layout: input[c * L + l]
/// Kernel shape: (C_out, C_in, K) flat layout: kernel[o * C_in * K + c * K + k]
/// Output shape: (C_out, L - K + 1)
pub fn conv1d(
  input : Array[Double],
  kernel : Array[Double],
  c_in : Int,
  l : Int,
  c_out : Int,
  k : Int,
) -> Array[Double] {
  let l_out = l - k + 1
  let output = Array::make(c_out * l_out, 0.0)
  for o = 0; o < c_out; o = o + 1 {
    for x = 0; x < l_out; x = x + 1 {
      let mut sum = 0.0
      for c = 0; c < c_in; c = c + 1 {
        for kk = 0; kk < k; kk = kk + 1 {
          let in_val = input[c * l + (x + kk)]
          let w_val = kernel[o * c_in * k + c * k + kk]
          sum = sum + in_val * w_val
        }
      }
      output[o * l_out + x] = sum
    }
  }
  output
}
