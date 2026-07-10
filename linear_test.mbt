///|
test "linear single batch" {
  // input:  [2, 3], shape (1, 2)
  // weight: [[1, 0], [0, 1], [1, 1]], shape (3, 2)
  // bias:   [10, 20, 30]
  let input = [2.0, 3.0]
  let weight = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
  let bias = [10.0, 20.0, 30.0]
  let out = linear(input, weight, bias, 1, 2, 3)
  assert_eq(out, [12.0, 23.0, 35.0])
}

///|
test "linear multi batch independent rows" {
  // input shape (2, 3)
  // weight shape (2, 3)
  let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let weight = [1.0, 1.0, 1.0, 2.0, 0.0, -1.0]
  let bias = [0.5, -1.0]
  let out = linear(input, weight, bias, 2, 3, 2)
  let expected = [6.5, -2.0, 15.5, 1.0]
  assert_true(max_abs_diff(out, expected) < 1.0e-9)
}

///|
test "linear identity weight with zero bias" {
  let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let weight = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  let bias = [0.0, 0.0, 0.0]
  let out = linear(input, weight, bias, 2, 3, 3)
  assert_eq(out, input)
}
