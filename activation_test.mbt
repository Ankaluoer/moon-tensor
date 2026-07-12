///|
test "relu mixed values" {
  let x = [-2.0, -1.0, 0.0, 1.0, 2.0]
  let out = relu(x)
  assert_eq(out, [0.0, 0.0, 0.0, 1.0, 2.0])
}

///|
test "relu all negative" {
  let x = [-3.0, -2.0, -1.0]
  let out = relu(x)
  assert_eq(out, [0.0, 0.0, 0.0])
}

///|
test "sigmoid zero" {
  let out = sigmoid([0.0])
  assert_true(max_abs_diff(out, [0.5]) < 1.0e-9)
}

///|
test "sigmoid large positive and negative" {
  let out = sigmoid([10.0, -10.0])
  assert_true(
    max_abs_diff(out, [0.9999546021312976, 0.0000453978687024]) < 1.0e-4,
  )
}
