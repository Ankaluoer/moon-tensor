///|
/// Softmax over a 1D array using numerically stable max subtraction.
/// Output shape is the same as input shape.
pub fn softmax(x : Array[Double]) -> Array[Double] {
  let len = x.length()
  let output = Array::make(len, 0.0)
  if len == 0 {
    output
  } else {
    let mut max_v = x[0]
    for i = 1; i < len; i = i + 1 {
      if x[i] > max_v {
        max_v = x[i]
      }
    }
    let mut sum = 0.0
    for i = 0; i < len; i = i + 1 {
      let v = @math.exp(x[i] - max_v)
      output[i] = v
      sum = sum + v
    }
    for i = 0; i < len; i = i + 1 {
      output[i] = output[i] / sum
    }
    output
  }
}

///|
/// LogSoftmax over a 1D array using numerically stable max subtraction.
/// Output shape is the same as input shape.
pub fn log_softmax(x : Array[Double]) -> Array[Double] {
  let len = x.length()
  let output = Array::make(len, 0.0)
  if len == 0 {
    output
  } else {
    let mut max_v = x[0]
    for i = 1; i < len; i = i + 1 {
      if x[i] > max_v {
        max_v = x[i]
      }
    }
    let mut sum = 0.0
    for i = 0; i < len; i = i + 1 {
      sum = sum + @math.exp(x[i] - max_v)
    }
    let log_sum = @math.ln(sum)
    for i = 0; i < len; i = i + 1 {
      output[i] = x[i] - max_v - log_sum
    }
    output
  }
}
