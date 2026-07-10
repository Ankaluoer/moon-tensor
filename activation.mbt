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
