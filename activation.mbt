///|
/// ReLU activation.
/// Output shape is the same as input shape.
pub fn relu(x : Array[Double]) -> Array[Double] {
  let output = Array::make(x.length(), 0.0)
  for i = 0; i < x.length(); i = i + 1 {
    output[i] = if x[i] > 0.0 { x[i] } else { 0.0 }
  }
  output
}

///|
/// Sigmoid activation.
/// Output shape is the same as input shape.
pub fn sigmoid(x : Array[Double]) -> Array[Double] {
  let output = Array::make(x.length(), 0.0)
  for i = 0; i < x.length(); i = i + 1 {
    output[i] = 1.0 / (1.0 + @math.exp(-x[i]))
  }
  output
}

///|
/// GELU activation using tanh approximation.
/// Output shape is the same as input shape.
pub fn gelu(x : Array[Double]) -> Array[Double] {
  let output = Array::make(x.length(), 0.0)
  let sqrt_2_over_pi = 0.7978845608028654
  for i = 0; i < x.length(); i = i + 1 {
    let v = x[i]
    let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v)
    output[i] = 0.5 * v * (1.0 + @math.tanh(inner))
  }
  output
}
