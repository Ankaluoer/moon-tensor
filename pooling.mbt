///|
/// MaxPool1d over input shape (channels, length).
/// Flat layout: x[c * length + l]
pub fn max_pool1d(
  x : Array[Double],
  channels : Int,
  length : Int,
  kernel_size : Int,
  stride : Int,
) -> Array[Double] {
  let out_length = (length - kernel_size) / stride + 1
  let output = Array::make(channels * out_length, 0.0)
  for c = 0; c < channels; c = c + 1 {
    for pos = 0; pos < out_length; pos = pos + 1 {
      let start = pos * stride
      let mut max_v = x[c * length + start]
      for k = 1; k < kernel_size; k = k + 1 {
        let v = x[c * length + start + k]
        if v > max_v {
          max_v = v
        }
      }
      output[c * out_length + pos] = max_v
    }
  }
  output
}

///|
/// AvgPool1d over input shape (channels, length).
/// Flat layout: x[c * length + l]
pub fn avg_pool1d(
  x : Array[Double],
  channels : Int,
  length : Int,
  kernel_size : Int,
  stride : Int,
) -> Array[Double] {
  let out_length = (length - kernel_size) / stride + 1
  let output = Array::make(channels * out_length, 0.0)
  for c = 0; c < channels; c = c + 1 {
    for pos = 0; pos < out_length; pos = pos + 1 {
      let start = pos * stride
      let mut sum = 0.0
      for k = 0; k < kernel_size; k = k + 1 {
        sum = sum + x[c * length + start + k]
      }
      output[c * out_length + pos] = sum / kernel_size.to_double()
    }
  }
  output
}
