///|
test "max_pool1d single channel" {
  let x = [1.0, 3.0, 2.0, 5.0, 4.0]
  let out = max_pool1d(x, 1, 5, 2, 1)
  assert_eq(out, [3.0, 3.0, 5.0, 5.0])
}

///|
test "avg_pool1d single channel stride two" {
  let x = [1.0, 3.0, 2.0, 6.0, 4.0]
  let out = avg_pool1d(x, 1, 5, 2, 2)
  assert_true(max_abs_diff(out, [2.0, 4.0]) < 1.0e-9)
}

///|
test "pool1d multi channel independent" {
  let x = [1.0, 5.0, 2.0, 4.0, 10.0, 20.0, 30.0, 40.0]
  let max_out = max_pool1d(x, 2, 4, 2, 2)
  let avg_out = avg_pool1d(x, 2, 4, 2, 2)
  assert_eq(max_out, [5.0, 4.0, 20.0, 40.0])
  assert_true(max_abs_diff(avg_out, [3.0, 3.0, 15.0, 35.0]) < 1.0e-9)
}
