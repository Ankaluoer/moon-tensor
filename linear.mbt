///|
/// Linear fully-connected operator: output = input @ weight^T + bias
/// Input shape:  (batch, in_features) row-major
/// Weight shape: (out_features, in_features) row-major
/// Bias shape:   (out_features)
/// Output shape: (batch, out_features) row-major
pub fn linear(
  input : Array[Double],
  weight : Array[Double],
  bias : Array[Double],
  batch : Int,
  in_features : Int,
  out_features : Int,
) -> Array[Double] {
  let output = Array::make(batch * out_features, 0.0)
  for b = 0; b < batch; b = b + 1 {
    for o = 0; o < out_features; o = o + 1 {
      let mut sum = bias[o]
      for i = 0; i < in_features; i = i + 1 {
        sum = sum + input[b * in_features + i] * weight[o * in_features + i]
      }
      output[b * out_features + o] = sum
    }
  }
  output
}
