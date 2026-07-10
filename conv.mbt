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
