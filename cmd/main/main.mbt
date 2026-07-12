///|
/// Demo: call core operators so they're included in the wasm output.
fn main {
  println("GEMM result:")
  let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
  let c = @lib.gemm(a, b, 2, 3, 2)
  println(c)

  println("Conv1D result:")
  let input = [1.0, 2.0, 3.0, 4.0, 5.0]
  let kernel = [1.0, 0.0, -1.0]
  let out = @lib.conv1d(input, kernel, 1, 5, 1, 3)
  println(out)

  println("Linear result:")
  let linear_input = [2.0, 3.0]
  let linear_weight = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
  let linear_bias = [10.0, 20.0, 30.0]
  let linear_out = @lib.linear(
    linear_input, linear_weight, linear_bias, 1, 2, 3,
  )
  println(linear_out)

  println("ReLU result:")
  let relu_out = @lib.relu([-2.0, -1.0, 0.0, 1.0, 2.0])
  println(relu_out)

  println("Sigmoid result:")
  let sigmoid_out = @lib.sigmoid([0.0, 10.0, -10.0])
  println(sigmoid_out)

  println("Softmax result:")
  let softmax_out = @lib.softmax([1.0, 2.0, 3.0])
  println(softmax_out)

  println("LayerNorm result:")
  let norm_out = @lib.layer_norm(
    [1.0, 2.0, 3.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0],
    1.0e-12,
  )
  println(norm_out)

  println("MaxPool1d result:")
  let max_pool_out = @lib.max_pool1d([1.0, 3.0, 2.0, 5.0, 4.0], 1, 5, 2, 1)
  println(max_pool_out)

  println("AvgPool1d result:")
  let avg_pool_out = @lib.avg_pool1d([1.0, 3.0, 2.0, 6.0, 4.0], 1, 5, 2, 2)
  println(avg_pool_out)
}
