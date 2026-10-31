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

## Stress benchmark

`bench/generate_stress.py` writes `bench/stress/`: a 21,518-line Python file, three imported helper modules, and a 500-line paste payload. `tests/lsp_stress.rs` runs the built server against this project over stdio. Both tests are ignored in the normal suite:

```sh
python3 bench/generate_stress.py   # regenerate bench/stress/
NDIM_STRESS_RUNS=3 NDIM_BENCH_OUT=/tmp/bench.json cargo test --release --test lsp_stress lsp_stress_benchmark -- --ignored --nocapture
NDIM_GOLDEN_OUT=/tmp/golden.json cargo test --release --test lsp_stress golden_dump -- --ignored --nocapture
```

`golden_dump` writes every diagnostic and inlay hint. Run it before and after a refactor and compare the two files with `cmp`: a cleanup must leave them identical.

Baseline (the dark spot commit plus the incremental work after it; release, 5 runs, p50, 21,518-line file): an edit while typing takes about 190 to 225 ms, which is a 150 ms debounce plus about 25 to 60 ms of compute plus about 11 to 18 ms of transport. An edit after an idle pause waits 20 to 150 ms (for example 96 ms in total). The worst case is an edit that moves every position of the file (`edit_char_start`): about 400 ms, 239 ms of it compute. Cold open takes about 305 ms. Before the incremental work an edit took about 410 ms (150 ms debounce plus about 245 ms of full re-analysis). The numbers depend on the machine.

The debounce column of the benchmark is the value the server chose for the edit. Do not time with `NDIM_VERIFY_INCREMENTAL=1`: it adds a fresh parse and analysis to every edit.

`lsp_stress_benchmark` passes with 0 marker failures and 0 scenario failures. Do not loosen a marker or a scenario assert to keep it green.

## Incremental analysis

The server uses `TextDocumentSyncKind::INCREMENTAL`. After a change it reuses the previous syntax tree (a tree with a syntax error is parsed again from scratch, so the tree always equals a fresh parse) and reuses the analysis of the top-level statements that did not change. The result must equal a fresh analysis of the same text.

- The first analysis after open runs in full. A change of the workspace folders drops the memo (it is valid for one set of search roots), so the next analysis redoes the statements, but it still reuses the tree.
- The debounce is at most 150 ms. A change within 400 ms of the previous one waits 150 ms. An isolated change of a document with a known analysis cost waits twice that cost, between 20 and 150 ms.
- A change message that does not fit the stored text is dropped as a whole. The document then gives no diagnostics or hints until the editor sends a full text.
- Limits: an edit in a hub (`__init__`, `forward`, other common names) reuses less. An edit that moves the whole file, an edit of an import line, and a statement with a syntax error recompute more. A file over 4 MiB or 100,000 statements runs without the memo.
- Debug switches (server environment): `NDIM_VERIFY_INCREMENTAL=1` compares every incremental result with a fresh one and uses the fresh result on a difference (it logs `incremental mismatch`). `NDIM_NO_PROPAGATE_MEMO=1` turns off the reuse of the propagation pass.

See `llm.txt` for the design.

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
- [x] Two-operand elementwise functions (`maximum`, `minimum`, `add`, `power`, `arctan2`, ...) broadcast both operands; `clip`/`clamp` broadcast array bounds
- [x] `torch.bmm`, `F.cross_entropy`, `F.nll_loss`
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
- [x] Tuple-literal unpacking (`a, b = x, y`, swaps, starred targets); an array unpacks into rows, a homogeneous tuple carry (`lax.scan`) gives each target the shared shape, and a name that is neither provably is unknown
- [x] Closures read enclosing variables at call time; `nonlocal` rebinding gives unknown
- [x] User submodules (`self.blk = Block(...)`, `Block(...)(x)`, `vmap(self.blk)`) through the class MRO
- [x] Provable scalar operands broadcast in binary operations
- [x] A name whose shape changes across a loop (`x, y = y, x`) is unknown after the loop
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
- [x] Incremental AST parsing (reusing previous trees on document changes for performance)
- [x] Resolve imported helpers from files under workspace roots and from site-packages (`$VIRTUAL_ENV`, `$CONDA_PREFIX`, or `<workspace>/.venv` and `venv`)
- [ ] Track unsaved changes across imported workspace files
- [x] Cache parsed trees and resolved shapes for repeated requests at the same document version
- [x] Cache import resolution and function signatures across edits (a saved change to an imported file, or a watched-file event, drops the affected entries and republishes the open documents)
- [x] Reuse parsed trees and local shape analysis across document edits (per top-level statement memo and propagation replay; see "Incremental analysis")
- [x] Incremental document sync and an adaptive debounce (20 to 150 ms)
