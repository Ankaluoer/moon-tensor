# moon-tensor

MoonBit neural network inference operators for small, portable deployments.

`moon-tensor` aims to be the first neural network inference operator library in the MoonBit ecosystem. It focuses on simple, readable, dependency-light kernels that compile cleanly to WebAssembly, JavaScript, and native targets.

## Features

- GEMM: row-major matrix multiplication
- Conv1D: no-padding, stride-1 1D convolution
- Linear: fully connected layer with bias
- Activations: ReLU, Sigmoid, GELU
- Softmax: Softmax and LogSoftmax
- Normalization: LayerNorm and RMSNorm
- Pooling: MaxPool1d and AvgPool1d
- Utilities: max absolute difference for numerical checks

## Quick Start

Add `moon-tensor` as a dependency in your MoonBit project, then import it from the package that uses the operators.

```moonbit nocheck
import {
  "Ankaluoer/moon-tensor" @tensor,
}
```

Example:

```moonbit nocheck
///|
fn main {
  let input = [1.0, 2.0, 3.0]
  let activated = @tensor.relu(input)
  println(activated)

  let logits = [1.0, 2.0, 3.0]
  let probs = @tensor.softmax(logits)
  println(probs)
}
```

Run tests:

```bash
moon test
```

Build for WebAssembly GC:

```bash
moon build --target wasm-gc
```

## Size Comparison

| Runtime or library | Approximate artifact size |
| --- | ---: |
| moon-tensor wasm-gc demo | 10 KB |
| ORT-Web | 25 MB |
| TensorFlow.js core-style bundles | 300-500 KB |

The goal is not to replace full ML runtimes. `moon-tensor` targets tiny inference paths where a small set of predictable operators is enough.

## Roadmap

- Quantized kernels for smaller models and faster edge inference
- More inference operators, including Conv2D and attention building blocks
- Keyword spotting demo for an end-to-end small model workflow
- More backend checks across wasm-gc, wasm, js, and native targets

## License

Apache-2.0
