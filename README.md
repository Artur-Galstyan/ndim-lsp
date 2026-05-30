# ndim-lsp

A Rust-based Language Server Protocol (LSP) for statically detecting tensor shape mismatches in Python deep learning code (NumPy, JAX, PyTorch, Equinox, etc.) using `tree-sitter` and type annotations.

## For agents working on this repo

The roadmap below is best-effort and lags the codebase. **Authoritative state lives in `src/`.** Before implementing any feature, grep for it first:

```sh
grep -E "KnownFunction::|apply_known_" src/known_functions.rs   # built-in fns
grep -E "LayerKind::|\"<ClassName>\"" src/layers.rs              # layers
grep -n "<feature>" src/analysis.rs src/main.rs                  # analysis & LSP wiring
```

If the feature is already implemented, **stop and report** — don't invent adjacent work to justify the run. See `LLM.txt` for the full agent handoff.

## Features & Roadmap

### Type Annotations & Declarations
- [x] Parse `jaxtyping` annotations (e.g., `Float[Array, "batch features"]`) from function arguments
- [x] Function-level shape scoping
- [ ] Parse numpy-style type hints (`NDArray[Shape["a, b"], Float]`)
- [ ] Support concrete integer dimensions (e.g., `Float[Array, "3 4"]`)
- [ ] Support return type annotations for cross-function shape inference
- [ ] Symbolic math in annotations (e.g., `"batch hidden*2"`, `"a b+1"`)
- [ ] Generics and TypeVars mapping across function boundaries

### Array Operations & Math
- [x] Matrix multiplication (`@`) with inner dimension validation
- [x] Element-wise operations (`+`, `-`, `*`, `/`) with strict exact shape matching
- [x] Transpose via attribute (`.T`) - reverses dimensions
- [ ] Broadcasting semantics for element-wise operations (e.g., `[batch, 1] + [batch, features]`)

### Framework Functions (JAX / NumPy)
- [x] Import alias resolution (e.g., `import jax.numpy as jnp`)
- [x] `jnp.concatenate` (dimension matching & concatenated axis modification)
- [x] `jnp.transpose` (tuple axis permutation)
- [x] `jnp.expand_dims` (multiple axes insertion, bounds checking)
- [x] `jnp.squeeze` (multiple axes removal, `size == 1` validation)
- [x] Reduction ops (`sum`, `mean`, `max`, `min`, `prod`, `std`, `var`) with `keepdims` and multiple axes support
- [x] `jnp.reshape` (dimension flattening/unflattening tracking)
- [x] `jnp.stack` (and `vstack`, `hstack`, `dstack`, `column_stack`)
- [x] `jnp.einsum`
- [x] `jnp.split`, `jnp.tile`, `jnp.repeat`, `jnp.flatten` (note: `split` returns a tuple; tuple-unpacking propagation still pending)
- [x] Dynamic `kwargs` handling in function parsing

### Deep Learning Layers
- [x] Framework-agnostic layer struct and parsing architecture
- [x] **Equinox:** `equinox.nn.Linear` (validates `in_features`, modifies shape to `out_features`, supports `kwargs` and positional args)
- [x] **PyTorch:** `torch.nn.Linear`
- [ ] **Flax:** `flax.linen.Dense`
- [x] Convolutional layers (`Conv1d`, `Conv2d`, `Conv3d`) tracking spatial dimension reductions
- [ ] Attention mechanisms (e.g., MultiheadAttention)
- [x] Shape-preserving layers (`Dropout{,1d,2d,3d}`, `BatchNorm{,1d,2d,3d}`, `LayerNorm`, `GroupNorm`, `ReLU`, `GELU`, etc.)

### Static Analysis & AST Tracking
- [x] Trace shapes through variable assignments (`z = x @ y`)
- [x] Evaluate chained expressions (`a @ b @ c`)
- [x] Evaluate parenthesized expressions (`(a + b) @ c`)
- [ ] Tuple unpacking assignments (`a, b = func()`)
- [ ] Array slicing & indexing (`x[:, 1:5, ...]`, boolean indexing)
- [ ] Control flow tracing (handling branches in `if/else`, `for` loops)
- [ ] Cross-function tracing (evaluating shapes returned from called helper functions)

### LSP Editor Features
- [x] **Diagnostics:** Error squiggles on invalid matmul or dimension mismatches
- [x] **Hover:** Show shapes of variables and arguments on hover
- [x] **Inlay Hints:** Display inferred dimensions inline directly in the editor
- [x] Show exact shapes and operators in diagnostic error messages
- [ ] Highlight specific conflicting dimensions in error output
- [ ] Diagnostic severity configuration (Warning vs Error)
- [ ] Go-to-definition for dimension names (jump to where a dimension size was defined)
- [ ] Code actions (e.g., "Insert `.T` to fix shape mismatch")

### Architecture & Performance
- [x] `tree-sitter-python` for robust, error-tolerant AST parsing
- [x] `tower-lsp` async server backend
- [x] Recursive AST shape resolution
- [ ] Incremental AST parsing (reusing previous trees on document changes for performance)
- [ ] Multi-file workspace support (tracking shapes and imports across modules)
- [ ] Caching parsed trees and resolved shapes across edits