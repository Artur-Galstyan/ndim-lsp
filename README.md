# ndim-lsp

A Rust-based Language Server Protocol (LSP) for statically detecting tensor shape mismatches in Python deep learning code (NumPy, JAX, PyTorch, Equinox, etc.) using `tree-sitter` and type annotations.

## Installation

Build the server and put the `ndim-lsp` binary on your `PATH`:

```sh
cargo install --path .
```

An editor extension starts the server. Both extensions look for `ndim-lsp` on `PATH`.

**VS Code** (extension `ndim-lsp`, in `../ndim-lsp-vscode`). Settings:

- `ndim-lsp.serverPath`: path to the binary (default `ndim-lsp`).
- `ndim-lsp.diagnosticSeverity`: severity of shape mismatches (default `error`).
- `ndim-lsp.approximationSeverity`: severity of approximate rules (default `warning`).

**Zed** (extension `ndim-lsp-zed`, in `../ndim-lsp-zed`). In your Zed settings:

```json
{
  "lsp": {
    "ndim-lsp": {
      "binary": { "path": "/path/to/ndim-lsp" },
      "initialization_options": {
        "diagnostic_severity": "error",
        "approximation_severity": "warning"
      }
    }
  }
}
```

Severity values are `error`, `warning`, `information`, and `hint`. The server reads them at start, so restart it after a change. An invalid value uses the default.

## Scoped shape support

This pass covers selected JAX 0.11/0.12, NumPy 2.4/2.5, and Torch 2.14 shape rules. See the [JAX changelog](https://docs.jax.dev/en/latest/changelog.html), [NumPy 2.5 notes](https://numpy.org/doc/2.5/release/2.5.0-notes.html), and [Torch 2.14 release](https://github.com/pytorch/pytorch/releases/tag/v2.14.0). It is not complete framework coverage or a dependency upgrade.

- `jax.scipy.special.erfcinv`, `jax.lax.log2`, and `jax.lax.one_minus_square` preserve shape; `jax.lax.polynomial` broadcasts scalar or known coefficient sequences.
- `jnp.minmax` and `jnp.top_k` track two unpacked outputs; `jax.random.multinomial` tracks batch dimensions and the category axis.
- Multi-input `atleast_1d`/`atleast_2d`/`atleast_3d` and `broadcast_arrays` bind tuple-unpacked outputs. `atleast_3d` adds a trailing axis to vectors and matrices.
- A tuple returned by these calls and bound to one name has no tensor shape. `np.pad` supports literal axis dictionaries; `jnp.pad` does not.
- `np.cross`/`jnp.cross` require size-3 vector axes. `torch.cross` has distinct rank and axis rules; `torch.linalg.matrix_sqrth` preserves square matrix shapes.
- Concrete split points support negative and clipped indices; `array_split`/`tensor_split` allow uneven chunks. Symbolic points assume positive, ordered, in-bounds values without proof.
- `row_stack` remains a legacy alias rule, not API or version validation. Numerical, compiler, device, and autodiff changes do not change shape rules.
- Generalized SciPy `eigh` and batched SciPy `lu` remain unsupported in this pass.

## For agents working on this repo

The roadmap below tracks broad support and lists important limits. **Authoritative state lives in `src/`.** Before adding any feature, grep for it first:

```sh
grep -rE "KnownFunction::|apply_known_" src/known_functions.rs src/known_functions/   # built-in fns
grep -E "LayerKind::|\"<ClassName>\"" src/layers.rs              # layers
grep -n "<feature>" src/analysis.rs src/main.rs                  # analysis & LSP wiring
```

If the feature is already implemented, **stop and report** — do not add adjacent work to justify the run. See `llm.txt` for the full agent handoff.

## Features & Roadmap

### Type Annotations & Declarations
- [x] Parse `jaxtyping` annotations (e.g., `Float[Array, "batch features"]`) from function arguments
- [x] Function-level shape scoping
- [x] Parse NumPy-style type hints (e.g., `NDArray[Shape["a, b"], Float]`)
- [x] Support concrete integer dimensions (e.g., `Float[Array, "3 4"]`)
- [x] Support return type annotations for cross-function shape inference
- [x] Compare flat symbolic sums and products (e.g., `"hidden*2"` equals `"2*hidden"`)
- [ ] Full symbolic algebra and nested dimension substitution (subtraction, division, and parenthesized forms)
- [ ] Generics and `TypeVar` mapping across function boundaries

### Array Operations & Math
- [x] Matrix multiplication (`@`) with inner dimension validation
- [x] Element-wise operations (`+`, `-`, `*`, `/`) with shape validation
- [x] Transpose via attribute (`.T`) - reverses dimensions
- [x] NumPy-style broadcasting for element-wise operations (e.g., `[batch, 1] + [batch, features]`)

### Framework Functions (JAX / NumPy)
- [x] Import alias resolution (e.g., `import jax.numpy as jnp`)
- [x] `jnp.concatenate` (dimension matching & concatenated axis modification; list and tuple literals only, a variable sequence is unknown)
- [x] `jnp.transpose` (tuple axis permutation)
- [x] `jnp.expand_dims` (multiple axes insertion, bounds checking)
- [x] `jnp.squeeze` (multiple axes removal, `size == 1` validation)
- [x] Reduction ops (`sum`, `mean`, `max`, `min`, `prod`, `std`, `var`) with `keepdims` and multiple axes support
- [x] `jnp.reshape` (dimension flattening/unflattening tracking)
- [x] `jnp.stack` (and `vstack`, `hstack`, `dstack`, `column_stack`; list and tuple literals only)
- [x] `jnp.einsum`
- [x] `jnp.split`, `jnp.tile`, `jnp.repeat`, `jnp.flatten` (including tuple propagation for `split`)
- [x] Dynamic `kwargs` handling in function parsing

### Deep Learning Layers
- [x] Framework-agnostic layer struct and parsing architecture
- [x] **Equinox:** `equinox.nn.Linear` (validates `in_features`, modifies shape to `out_features`, supports `kwargs` and positional args)
- [x] **PyTorch:** `torch.nn.Linear`
- [x] **Flax:** `flax.linen.Dense`
- [x] Convolutional layers (`Conv1d`, `Conv2d`, `Conv3d`) tracking spatial dimension reductions
- [x] Attention mechanisms (`MultiheadAttention` for PyTorch and Equinox, plus functional attention rules)
- [x] Shape-preserving layers (`Dropout{,1d,2d,3d}`, `BatchNorm{,1d,2d,3d}`, `LayerNorm`, `GroupNorm`, `ReLU`, `GELU`, etc.)
- [x] More layers (transposed convolution, pooling, `Embedding`, RNN/LSTM/GRU, `Flatten`/`Unflatten`, `Upsample`, `Sequential`, einops layers): see `TO_IMPLEMENT.md`

### Static Analysis & AST Tracking
- [x] Trace shapes through variable assignments (`z = x @ y`)
- [x] Evaluate chained expressions (`a @ b @ c`)
- [x] Evaluate parenthesized expressions (`(a + b) @ c`)
- [x] Tuple unpacking for known multi-output calls and supported layers (`a, b = func()`)
- [ ] Tuple unpacking for user-defined helper tuple returns
- [x] Basic array slicing and indexing (`x[:, 1:5, ...]`, integer indices, `None`)
- [ ] Precise advanced and boolean array indexing
- [ ] Control flow tracing (handling branches in `if/else`, `for` loops)
- [x] Cross-function tracing for same-file and imported helper functions

### LSP Editor Features
- [x] **Diagnostics:** Error squiggles on invalid matmul or dimension mismatches
- [x] **Hover:** Show shapes of variables and arguments on hover
- [x] **Inlay Hints:** Display inferred dimensions inline directly in the editor
- [x] Show exact shapes and operators in diagnostic error messages
- [x] Point two-operand mismatch diagnostics at both conflicting operands
- [x] Highlight exact conflicting dimension tokens for direct binary operations on annotated variables
- [x] Diagnostic severity configuration — use `diagnostic_severity` for mismatches and `approximation_severity` for approximate rules
- [x] Go-to-definition for simple symbolic dimension names within the current function scope
- [x] Safe code action to append `.T` when transposing the right matmul operand proves valid

### Architecture & Performance
- [x] `tree-sitter-python` for robust, error-tolerant AST parsing
- [x] `tower-lsp` async server backend
- [x] Recursive AST shape resolution
- [ ] Incremental AST parsing (reusing previous trees on document changes for performance)
- [x] Resolve imported helpers from files under workspace roots and from site-packages (`$VIRTUAL_ENV`, `$CONDA_PREFIX`, or `<workspace>/.venv` and `venv`)
- [ ] Track unsaved changes across imported workspace files
- [x] Cache parsed trees and resolved shapes for repeated requests at the same document version
- [x] Cache import resolution and function signatures across edits (limit: a saved change to an imported file stays cached until a workspace folder change or a server restart)
- [ ] Reuse parsed trees and local shape analysis across document edits
