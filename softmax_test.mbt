///|
test "softmax sums to one" {
  let out = softmax([1.0, 2.0, 3.0])
  let sum = out[0] + out[1] + out[2]
  let diff = sum - 1.0
  let abs_diff = if diff >= 0.0 { diff } else { -diff }
  assert_true(abs_diff < 1.0e-9)
}

///|
test "softmax stable for large equal values" {
  let out = softmax([1000.0, 1000.0, 1000.0])
  let expected = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
  assert_true(max_abs_diff(out, expected) < 1.0e-9)
}

///|
test "log_softmax exponent matches softmax" {
  let x = [1.0, 2.0, 3.0]
  let probs = softmax(x)
  let logs = log_softmax(x)
  let exp_logs = [@math.exp(logs[0]), @math.exp(logs[1]), @math.exp(logs[2])]
  assert_true(max_abs_diff(exp_logs, probs) < 1.0e-9)
}
