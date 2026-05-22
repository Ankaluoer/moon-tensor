///|
/// Demo: call GEMM and Conv1D so they're included in the wasm output.
fn main {
  // Small GEMM
  let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
  let c = @spike.gemm(a, b, 2, 3, 2)
  println("GEMM result:")
  println(c)

  // Small Conv1D
  let input = [1.0, 2.0, 3.0, 4.0, 5.0]
  let kernel = [1.0, 0.0, -1.0]
  let out = @spike.conv1d(input, kernel, 1, 5, 1, 3)
  println("Conv1D result:")
  println(out)
}
