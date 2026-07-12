///|
/// LayerNorm over a 1D array.
/// x, gamma, beta shape: (features)
pub fn layer_norm(
  x : Array[Double],
  gamma : Array[Double],
  beta : Array[Double],
  eps : Double,
) -> Array[Double] {
  let len = x.length()
  let output = Array::make(len, 0.0)
  if len == 0 {
    output
  } else {
    let mut mean = 0.0
    for i = 0; i < len; i = i + 1 {
      mean = mean + x[i]
    }
    mean = mean / len.to_double()
    let mut variance = 0.0
    for i = 0; i < len; i = i + 1 {
      let diff = x[i] - mean
      variance = variance + diff * diff
    }
    variance = variance / len.to_double()
    let scale = 1.0 / (variance + eps).sqrt()
    for i = 0; i < len; i = i + 1 {
      output[i] = (x[i] - mean) * scale * gamma[i] + beta[i]
    }
    output
  }
}

///|
/// RMSNorm over a 1D array.
/// x, gamma shape: (features)
pub fn rms_norm(
  x : Array[Double],
  gamma : Array[Double],
  eps : Double,
) -> Array[Double] {
  let len = x.length()
  let output = Array::make(len, 0.0)
  if len == 0 {
    output
  } else {
    let mut mean_square = 0.0
    for i = 0; i < len; i = i + 1 {
      mean_square = mean_square + x[i] * x[i]
    }
    mean_square = mean_square / len.to_double()
    let scale = 1.0 / (mean_square + eps).sqrt()
    for i = 0; i < len; i = i + 1 {
      output[i] = x[i] * scale * gamma[i]
    }
    output
  }
}
