///|
test "conv1d simple: 1 channel in, 1 channel out, kernel size 3" {
  // input: [1, 2, 3, 4, 5], shape (1, 5)
  // kernel: [1, 0, -1], shape (1, 1, 3)  -- edge detector
  // output should be:
  //   x=0: 1*1 + 2*0 + 3*(-1) = -2
  //   x=1: 2*1 + 3*0 + 4*(-1) = -2
  //   x=2: 3*1 + 4*0 + 5*(-1) = -2
  let input = [1.0, 2.0, 3.0, 4.0, 5.0]
  let kernel = [1.0, 0.0, -1.0]
  let out = conv1d(input, kernel, 1, 5, 1, 3)
  let expected = [-2.0, -2.0, -2.0]
  assert_true(max_abs_diff(out, expected) < 1.0e-9)
}

///|
test "conv1d multi-channel: 2 in, 2 out, kernel size 2" {
  // C_in=2, L=4, C_out=2, K=2
  // input shape (2, 4):
  //   ch0: [1, 2, 3, 4]
  //   ch1: [5, 6, 7, 8]
  // kernel shape (2, 2, 2):
  //   out0: [[1, 0], [0, 1]]   -> sums input[0][x] + input[1][x+1]
  //   out1: [[1, 1], [1, 1]]   -> sums of all 4 values in window
  let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  let kernel = [
    1.0, 0.0, 0.0, 1.0, // out_channel 0
     1.0, 1.0, 1.0, 1.0, // out_channel 1
  ]
  let out = conv1d(input, kernel, 2, 4, 2, 2)
  // out shape (2, 3):
  //   out0 at x=0: 1*1+2*0 + 5*0+6*1 = 1+6 = 7
  //   out0 at x=1: 2*1+3*0 + 6*0+7*1 = 2+7 = 9
  //   out0 at x=2: 3*1+4*0 + 7*0+8*1 = 3+8 = 11
  //   out1 at x=0: 1+2+5+6 = 14
  //   out1 at x=1: 2+3+6+7 = 18
  //   out1 at x=2: 3+4+7+8 = 22
  let expected = [7.0, 9.0, 11.0, 14.0, 18.0, 22.0]
  assert_true(max_abs_diff(out, expected) < 1.0e-9)
}
