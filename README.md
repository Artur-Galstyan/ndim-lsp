# ndim-lsp

A Rust-based Language Server Protocol (LSP) for statically detecting tensor shape mismatches in Python deep learning code (NumPy, JAX, PyTorch, Equinox, etc.) using `tree-sitter` and type annotations.

## Installation

Build the server and put the `ndim-lsp` binary on your `PATH`:

```sh
cargo install --path .
```

An editor extension starts the server. Both extensions look for `ndim-lsp` on `PATH`.

### Command line

`ndim-lsp check [--shapes] <path>...` analyzes files or directories without an editor and prints one line per diagnostic, like a compiler:

```text
path:line:column: error|warning: variable: message
```

`--shapes` also prints the shape of each assignment. A directory walk skips hidden directories, virtual environments, `build`, `dist`, and caches. Imports resolve from the current directory (and its `src` directory) and its site-packages. The exit code is 1 when an error was found, 2 for a usage or read error, and 0 otherwise. Warnings do not change the exit code. A summary goes to stderr.

### Errors and warnings

Only two different integers are a proven conflict, so only they give an error (code `shape-mismatch`). A conflict that rests on a name, for example `n` against `m` or `n` against `4`, can be equal at run time: it gives a warning (code `shape-approximation`) and no result shape. The approximate rules give warnings too.

**VS Code** (extension `ndim-lsp`, in `../ndim-lsp-vscode`). The setting `ndim-lsp.serverPath` is the path to the binary (default `ndim-lsp`). A change to this setting restarts the server.

**Zed** (extension `ndim-lsp-zed`, in `../ndim-lsp-zed`). In your Zed settings:

```json
{
  "lsp": {
    "ndim-lsp": {
      "binary": { "path": "/path/to/ndim-lsp" },
      "settings": {
        "diagnostic_severity": "error",
        "approximation_severity": "warning",
        "inlay_hints": true,
        "inlay_hints_max_length": 0
      }
    }
  }
}
```

The Zed extension in `../ndim-lsp-zed` is stale (`extension.toml` 0.0.1, `Cargo.toml` 0.0.2, an old `extension.wasm`). Rebuild the Zed extension to get live changes. The `extension.wasm` file must include `language_server_workspace_configuration`. The `initialization_options` object takes the same keys, and Zed uses it for live changes if `settings` is not set.

### Options

| Key | Default | VS Code setting | Zed setting (under `lsp.ndim-lsp.settings`) |
| --- | --- | --- | --- |
| `diagnostic_severity` | `"error"` | `ndim-lsp.diagnosticSeverity` | `diagnostic_severity` |
| `approximation_severity` | `"warning"` | `ndim-lsp.approximationSeverity` | `approximation_severity` |
| `inlay_hints` | `true` | `ndim-lsp.inlayHints` | `inlay_hints` |
| `inlay_hints_max_length` | `0` (no limit) | `ndim-lsp.inlayHintsMaxLength` | `inlay_hints_max_length` |

Severity values are `error`, `warning`, `information`, and `hint`, in any case. `inlay_hints: false` gives no hints. A label longer than `inlay_hints_max_length` chars is cut to that length with an ellipsis, and the tooltip keeps the full shape. An invalid value uses the default.

The server reads the options from `initializationOptions` at start. A `workspace/didChangeConfiguration` notification changes them with no restart. The keys go under `ndim-lsp` or at the top level of `settings`. A key that is not in the notification keeps its value, so a key removed from the editor settings stays in use until the next restart. If the notification has no option, a client with `workspace/configuration` gets a pull of section `ndim-lsp`. A change republishes diagnostics from the cached analysis and sends `workspace/inlayHint/refresh`.

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

Baseline, measured at commit `993040d` (release, 5 runs, p50, 22,323-line file): an edit takes about 50 to 80 ms, which is a 20 ms debounce plus about 16 to 44 ms of compute plus about 14 to 16 ms of transport. Typing runs one analysis for each key. The worst case is the first edit after open (`edit_char_start`): about 415 ms, 249 ms of it compute, because open builds no memo and the first edit analyzes every statement. Cold open takes about 378 ms (366 ms of compute). A memo built at open was measured and rejected: open got 12 to 17 percent slower, and the first edit fell from 396 to 41 ms. The numbers depend on the machine and on the session, so compare only runs of one session.

The benchmark, the golden dump, and the 24-file corpus are written by the same authors as the analyzer. Real projects (`ndim-lsp check .`) found false errors that these measures did not, so real code is the main test now (see `llm.txt`).

The debounce column of the benchmark is the value the server chose for the edit. Do not time with `NDIM_VERIFY_INCREMENTAL=1`: it adds a fresh parse and analysis to every edit.

`lsp_stress_benchmark` passes with 0 marker failures and 0 scenario failures. Do not loosen a marker or a scenario assert to keep it green.

## Incremental analysis

The server uses `TextDocumentSyncKind::INCREMENTAL`. After a change it reuses the previous syntax tree (a tree with a syntax error is parsed again from scratch, so the tree always equals a fresh parse) and reuses the analysis of the top-level statements that did not change. The result must equal a fresh analysis of the same text.

- The first analysis after open runs in full. A change of the workspace folders drops the memo (it is valid for one set of search roots), so the next analysis redoes the statements, but it still reuses the tree.
- The debounce depends on the analysis cost of the document, not on the rhythm of the keys. It is half of the average incremental run or twice the last whole analysis, between 20 and 150 ms. A document with no known cost waits 150 ms, also after a failed analysis or a change of the workspace folders. Each key can start one analysis, so typing uses more CPU than a fixed window, but a change waits less.
- A change message that does not fit the stored text is dropped as a whole. The document then gives no diagnostics or hints until the editor sends a full text.
- Limits: an edit in a hub (`__init__`, `forward`, other common names) reuses less. The first edit after open, an edit of an import line, and a statement with a syntax error recompute more. A file over 4 MiB or 100,000 statements runs without the memo.
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
- [x] Reduction ops (`sum`, `mean`, `max`, `min`, `prod`, `std`, `var`, `median`, `quantile`, `norm`, ...) with `keepdims` and multiple axes, read by the signature of each library (a positional `keepdim` works)
- [x] Two-operand elementwise functions (`maximum`, `minimum`, `add`, `power`, `arctan2`, ...) broadcast both operands; `clip`/`clamp` broadcast array bounds
- [x] `torch.bmm`, `mm`, `mv`, `addmm`, `baddbmm`, `F.linear`, `F.scaled_dot_product_attention`, `F.cross_entropy`, `F.nll_loss`, the functional losses (`F.mse_loss`, ...), and the loss layers (`nn.MSELoss`, `L1Loss`, `SmoothL1Loss`, `HuberLoss`, `BCELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss`, `NLLLoss`)
- [x] `jax.lax.dot_general` with literal dimension numbers; `unique`, `masked_select`, and `bincount` give a 1-D shape of unknown length
- [x] `jnp.reshape` (dimension flattening/unflattening tracking)
- [x] `jnp.stack` (and `vstack`, `hstack`, `dstack`, `column_stack`; list and tuple literals, and list comprehensions over a provable array)
- [x] `jnp.einsum` (spaces in the subscripts are skipped) and `einops.einsum` (space-separated names)
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
- [x] More layers (transposed convolution with `output_padding` and `dilation`, pooling with `ceil_mode` and `dilation`, `Embedding`, RNN/LSTM/GRU with `bidirectional`, `num_layers`, `proj_size`, and `batch_first`, `Flatten`/`Unflatten`, `Upsample`, `Sequential`, einops layers): see `TO_IMPLEMENT.md`

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
- [x] Basic array slicing and indexing (`x[:, 1:5, ...]`, integer indices, `None`). A slice of an integer axis with integer bounds has the size of Python. A slice of a symbolic axis keeps the rank and has an exact form: `x[1:]` and `x[:-1]` are `n-1`, `x[a:-b]` is `n-a-b`, `x[:4]` and `x[-4:]` are `min(4, n)`, `x[a:b]` is `min(b-a, n-a)`, and `x[::2]` is `ceil(n/2)`. The `n-a` forms assume that the axis has at least that many items. A `min` or `ceil` form conflicts with nothing. `n-1` equals `n-1`, does not conflict with an integer or another name, and conflicts with `n`. An unproven bound or step (`x[:, k:]`) gives an axis of size `?` that conflicts with nothing, and the other axes stay known. A slice of `x[:4]` folds the limit: `x[:4][5:]` is 0. A slice of an axis that is not a name (`_`, `#n`, `*b`, `...`, `n=4`) is `?`, and an index after `*b` or `...` is unknown unless an ellipsis places it. A negative step on a torch tensor is unknown
- [x] A variadic dim (`*b`, `...`) makes a rank a lower bound. No rule reports an error for a value with a variadic dim, except a broadcast or a matmul that aligns from the right. `concatenate` and `stack` keep a shape only when the axis does not cross the variadic dim
- [x] A shape argument is a tuple, a list, one size, or a sum of tuples and `x.shape`: `jnp.zeros(x.shape + (3,))` is `[n, m, 3]`. Any other expression (`(2,) * 3`, a call, a conditional, a comprehension) is unknown
- [x] A list of integers as an index gives its length to the axis. Any other index that is not a proven integer or a slice (an array, a mask, a tuple) is unknown
- [ ] Precise array and boolean indexing
- [x] Control flow tracing: `if`/`elif`/`else`, `match`, `try`, loops and `with` start each arm from the entry state and join the live paths (a name with different shapes is unknown)
- [x] Cross-function tracing for same-file and imported helper functions

### LSP Editor Features
- [x] **Diagnostics:** Errors for proven mismatches and warnings for conflicts that rest on a name, with the codes `shape-mismatch` and `shape-approximation`
- [x] **Hover:** Show the shape of a variable or a parameter at the cursor (the assignment that holds there)
- [x] **Inlay Hints:** Display inferred dimensions inline directly in the editor
- [x] Inlay hint options: `inlay_hints` turns hints off, and `inlay_hints_max_length` cuts long labels
- [x] Live settings: `workspace/didChangeConfiguration` applies all options with no restart
- [x] Show exact shapes and operators in diagnostic error messages
- [x] Point two-operand mismatch diagnostics at both conflicting operands
- [x] Report a binary operator error on the statement, with the conflicting annotation dimension of the other operand as related information
- [x] Diagnostic severity configuration — use `diagnostic_severity` for mismatches and `approximation_severity` for approximate rules
- [x] Go-to-definition, references, highlights, and rename for simple symbolic dimension names within the current function scope
- [x] UTF-8 positions when the client offers them (else UTF-16)
- [x] Command-line mode: `ndim-lsp check <path>...`
- [x] Safe code action to append `.T` when transposing the right matmul operand proves valid

### Architecture & Performance
- [x] `tree-sitter-python` for robust, error-tolerant AST parsing
- [x] `tower-lsp` async server backend
- [x] Recursive AST shape resolution
- [x] Incremental AST parsing (reusing previous trees on document changes for performance)
- [x] Resolve imported helpers from files under workspace roots (and a `src` layout) and from site-packages (`$VIRTUAL_ENV`, `$CONDA_PREFIX`, or `<workspace>/.venv` and `venv`), with relative imports of the open document
- [x] Track unsaved changes across imported workspace files (imports read the editor text of open documents)
- [x] Cache parsed trees and resolved shapes for repeated requests at the same document version
- [x] Cache import resolution and function signatures across edits (a saved change to an imported file, or a watched-file event, drops the affected entries and republishes the open documents)
- [x] Reuse parsed trees and local shape analysis across document edits (per top-level statement memo and propagation replay; see "Incremental analysis")
- [x] Incremental document sync and a cost-based debounce (20 to 150 ms)
- [x] One analysis stays bounded on a hostile file: a dim over 256 bytes is `?` and the rank stays, a value over rank 64 has no shape, one statement has a limit of steps, and a syntax tree deeper than 1,000 levels gives no analysis
