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

`bench/generate_stress.py` writes `bench/stress/`: a 22,323-line Python file, five imported helper modules, and a 500-line paste payload. `tests/lsp_stress.rs` runs the built server against this project over stdio. Both tests are ignored in the normal suite:

```sh
python3 bench/generate_stress.py   # regenerate bench/stress/
NDIM_STRESS_RUNS=3 NDIM_BENCH_OUT=/tmp/bench.json cargo test --release --test lsp_stress lsp_stress_benchmark -- --ignored --nocapture
NDIM_GOLDEN_OUT=/tmp/golden.json cargo test --release --test lsp_stress golden_dump -- --ignored --nocapture
```

`golden_dump` writes every diagnostic and inlay hint. Run it before and after a refactor and compare the two files with `cmp`: a cleanup must leave them identical.

Baseline (the code with free parameter dims, scoped bare names, and tracked import rebinding, release, 5 runs, p50, 22,323-line file): an edit takes about 50 to 80 ms, which is a 20 ms debounce plus about 16 to 44 ms of compute plus about 14 to 16 ms of transport. Typing runs one analysis for each key: `typing_plain` takes 52 ms for each key and gives 12 analyses and 12 publications for 12 keys. The worst case is an edit that moves every position of the file (`edit_char_start`): about 415 ms, 249 ms of it compute, with the full 150 ms debounce. The next edit also waits 150 ms (`edit_char_start_revert`, 189 ms). Cold open takes about 378 ms (366 ms of compute). Before the incremental work an edit took about 410 ms (150 ms debounce plus about 245 ms of full re-analysis).

Control on one input against commit `c6f2860`, two 5-run benchmarks of each code, in turn: cold open compute rises about 2 percent (362 to 368 ms), inside the noise, and small edits and `edit_char_start` do not change. Edits that rebuild layers cost 2 to 4 ms more (`paste_500_lines` 41 to 45 ms). An older control against commit `025e7f2`, the commit before the cost-based debounce, showed `typing_plain` falling from 211 to 58 ms and `edit_char_end` from 195 to 51 ms. The cost is CPU time: at 100 ms for each key, that commit ran 1 analysis for 12 keys and the debounce gives 12. The numbers depend on the machine and on the session, so compare only runs of one session.

The debounce column of the benchmark is the value the server chose for the edit. Do not time with `NDIM_VERIFY_INCREMENTAL=1`: it adds a fresh parse and analysis to every edit.

`lsp_stress_benchmark` passes with 0 marker failures and 0 scenario failures. Do not loosen a marker or a scenario assert to keep it green.

## Incremental analysis

The server uses `TextDocumentSyncKind::INCREMENTAL`. After a change it reuses the previous syntax tree (a tree with a syntax error is parsed again from scratch, so the tree always equals a fresh parse) and reuses the analysis of the top-level statements that did not change. The result must equal a fresh analysis of the same text.

- The first analysis after open runs in full. A change of the workspace folders drops the memo (it is valid for one set of search roots), so the next analysis redoes the statements, but it still reuses the tree.
- The debounce depends on the analysis cost of the document, not on the rhythm of the keys. It is half of the average incremental run or twice the last whole analysis, between 20 and 150 ms. A document with no known cost waits 150 ms, also after a failed analysis or a change of the workspace folders. Each key can start one analysis, so typing uses more CPU than a fixed window, but a change waits less.
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
- [x] `torch.bmm`, `F.cross_entropy`, `F.nll_loss`, and the loss layers (`nn.MSELoss`, `L1Loss`, `SmoothL1Loss`, `HuberLoss`, `BCELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss`, `NLLLoss`)
- [x] `jnp.reshape` (dimension flattening/unflattening tracking)
- [x] `jnp.stack` (and `vstack`, `hstack`, `dstack`, `column_stack`; list and tuple literals, and list comprehensions over a provable array)
- [x] `jnp.einsum`
- [x] `jnp.split`, `jnp.tile`, `jnp.repeat`, `jnp.flatten` (including tuple propagation for `split`)
- [x] Dynamic `kwargs` handling in function parsing

### Deep Learning Layers
- [x] Framework-agnostic layer struct and parsing architecture
- [x] **Equinox:** `equinox.nn.Linear` (validates `in_features`, modifies shape to `out_features`, supports `kwargs` and positional args)
- [x] **PyTorch:** `torch.nn.Linear`
- [x] **Flax:** `flax.linen.Dense`
- [x] Convolutional layers (`Conv1d`, `Conv2d`, `Conv3d`) tracking spatial dimension reductions, with `padding="same"`, `padding="valid"`, and `dilation`. A padding name without an integer proof, and a torch padding string that is not lower case, are unknown
- [x] Attention mechanisms (`MultiheadAttention` for PyTorch and Equinox, plus functional attention rules)
- [x] Shape-preserving layers (`Dropout{,1d,2d,3d}`, `BatchNorm{,1d,2d,3d}`, `LayerNorm`, `GroupNorm`, `ReLU`, `GELU`, etc.)
- [x] More layers (transposed convolution, pooling, `Embedding`, RNN/LSTM/GRU, `Flatten`/`Unflatten`, `Upsample`, `Sequential`, einops layers): see `TO_IMPLEMENT.md`

### Static Analysis & AST Tracking
- [x] Trace shapes through variable assignments (`z = x @ y`)
- [x] Evaluate chained expressions (`a @ b @ c`)
- [x] Evaluate parenthesized expressions (`(a + b) @ c`)
- [x] Tuple unpacking for known multi-output calls and supported layers (`a, b = func()`)
- [x] Tuple-literal unpacking (`a, b = x, y`, swaps, starred targets); an array unpacks into rows, a homogeneous tuple carry (`lax.scan`) gives each target the shared shape, and a name that is neither provably is unknown
- [x] Closures read enclosing variables at call time; `nonlocal` rebinding, also from a method, gives unknown
- [x] User submodules (`self.blk = Block(...)`, `Block(...)(x)`, `vmap(self.blk)`) through the class MRO, including classes imported from other files, `nn.ModuleList` and `nn.Sequential` fields of user modules, and chain loops (`for l in self.layers: x = l(x)`)
- [x] Module globals and integer constants read in a function (known only when all writes come before the reader and have one shape)
- [x] Integer constants and `self.<attr>` constants in layer constructor dims (`nn.Linear(8, OUT)`); a dim with no proven value makes the layer output unknown
- [x] A parameter or a dataclass-field dim conflicts with nothing, because the caller gives its value: `nn.Linear(8, d)` plus a literal `[n, 4]` gives `[n, 4]`, not an error
- [x] Methods resolve by class: `self.m()`, `cls.m()`, and `super().m()` follow the MRO of the class of the call, and an override in the file gives unknown
- [x] Imports follow Python scopes: a function-local import, a parameter, or a local name changes the meaning of a name only in its own scope, and a module-level rebinding applies from its statement on and to function bodies
- [x] Bare names follow Python scopes: a function name that a parameter, a local, or a rebinding hides gives unknown, and a `vf = vmap(f)` name resolves by the scope that owns it. `f = jax.jit(f)`, `jax.checkpoint`, `equinox.filter_jit`, and `functools.partial(f)` with no bound argument keep the function
- [x] Imported names resolve to one top-level definition: a name bound more than once gives unknown, and a method of the same name never replaces it
- [x] Comprehensions and lambdas where the result is used (`stack` of a list comprehension, `sum`, `vmap`, `lax.map`, `lax.scan` and direct calls of a lambda)
- [x] Provable scalar operands broadcast in binary operations
- [x] A name whose shape changes across a loop (`x, y = y, x`) is unknown after the loop, and the hints of the first iteration are taken back
- [ ] Tuple unpacking for user-defined helper tuple returns
- [x] Basic array slicing and indexing (`x[:, 1:5, ...]`, integer indices, `None`). A slice of an integer axis with integer bounds has the size of Python. Other slices that do not keep the whole axis are unknown: `x[:2]` on `n` is `min(2, n)`. A negative step on a torch tensor is unknown
- [x] A list of integers as an index gives its length to the axis. Any other index that is not a proven integer or a slice (an array, a mask, a tuple) is unknown
- [ ] Precise array and boolean indexing
- [x] Control flow tracing: `if`/`elif`/`else`, `match`, `try`, loops and `with` start each arm from the entry state and join the live paths (a name with different shapes is unknown)
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
- [x] Incremental document sync and a cost-based debounce (20 to 150 ms)
