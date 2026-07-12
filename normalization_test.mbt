///|
test "layer_norm normalizes mean and variance" {
  let x = [1.0, 2.0, 3.0]
  let gamma = [1.0, 1.0, 1.0]
  let beta = [0.0, 0.0, 0.0]
  let out = layer_norm(x, gamma, beta, 1.0e-12)
  let expected = [-1.224744871391589, 0.0, 1.224744871391589]
  assert_true(max_abs_diff(out, expected) < 1.0e-9)
}

///|
test "layer_norm applies gamma and beta" {
  let x = [1.0, 3.0]
  let gamma = [2.0, 3.0]
  let beta = [10.0, 20.0]
  let out = layer_norm(x, gamma, beta, 1.0e-12)
  let expected = [8.0, 23.0]
  assert_true(max_abs_diff(out, expected) < 1.0e-6)
}

///|
test "rms_norm scales by root mean square" {
  let x = [3.0, 4.0]
  let gamma = [1.0, 2.0]
  let out = rms_norm(x, gamma, 1.0e-12)
  let expected = [0.848528137423857, 2.262741699796952]
  assert_true(max_abs_diff(out, expected) < 1.0e-9)
}
