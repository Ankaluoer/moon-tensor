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
