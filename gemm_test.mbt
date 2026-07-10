///|
test "gemm 2x2 identity" {
  // I @ I = I
  let i = [1.0, 0.0, 0.0, 1.0]
  let c = gemm(i, i, 2, 2, 2)
  assert_eq(c, [1.0, 0.0, 0.0, 1.0])
}

///|
test "gemm 2x3 times 3x2" {
  // A = [[1,2,3],[4,5,6]]  (2x3)
  // B = [[7,8],[9,10],[11,12]]  (3x2)
  // C = A @ B = [[58, 64], [139, 154]]
  let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
  let c = gemm(a, b, 2, 3, 2)
  assert_eq(c, [58.0, 64.0, 139.0, 154.0])
}

///|
test "gemm zero matrix" {
  let a = Array::make(9, 0.0) // 3x3 zeros
  let b = Array::make(9, 1.0) // 3x3 ones
  let c = gemm(a, b, 3, 3, 3)
  let expected = Array::make(9, 0.0)
  assert_eq(max_abs_diff(c, expected), 0.0)
}
