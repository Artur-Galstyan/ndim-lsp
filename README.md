# ndim-lsp

A Rust-based LSP that detects tensor shape mismatches in NumPy/JAX code using jaxtyping annotations and tree-sitter parsing.

## Features

### Shape sources
- [x] Parse jaxtyping annotations (`Float[Array, "a b"]`)
- [ ] Parse numpy type hints (`NDArray[Shape["a, b"], Float]`)
- [ ] Concrete integer dimensions (`Float[Array, "3 4"]`)
- [ ] Return type annotations on functions

### Operations
- [x] Matrix multiplication (`@`) with inner dimension checking
- [x] Element-wise operations (`+`, `-`, `*`, `/`) with exact shape matching
- [x] Transpose (`.T`) — reverses dimensions
- [ ] `jnp.transpose(x, axes=(...))` — permute specific axes
- [ ] `jnp.reshape(x, shape)` — reshape tracking
- [ ] `jnp.expand_dims` / `jnp.squeeze`
- [ ] `jnp.concatenate` / `jnp.stack`
- [ ] `jnp.sum` / `jnp.mean` with axis argument (dimension reduction)
- [ ] Broadcasting rules

### Shape tracing
- [x] Trace shapes through assignments (`z = x @ y`)
- [x] Chained operations in one expression (`a @ b @ c @ d`)
- [x] Parenthesized expressions (`(x @ y) @ z`)
- [ ] Shapes through function calls (`z = f(x)` using return type of `f`)
- [ ] Shapes through tuple unpacking (`a, b = ...`)
- [ ] Shapes through indexing / slicing (`x[0]`, `x[:, 1:]`)

### Diagnostics
- [x] Squiggles on invalid matmul
- [x] Squiggles on invalid element-wise ops
- [x] Error message shows both shapes and operator
- [ ] Show which specific dimensions conflict
- [ ] Warning vs error severity levels

### Editor features
- [x] Hover shows shape of variables
- [x] Per-function scoping of shapes
- [ ] Hover on expressions (not just variables)
- [ ] Go-to-definition for dimension names
- [x] Inlay hints showing shapes inline
- [ ] Code actions (e.g. "insert .T here")

### Architecture
- [x] tree-sitter Python parsing
- [x] tower-lsp server
- [x] Recursive shape resolution
- [ ] Incremental parsing (reuse previous tree)
- [ ] Multi-file support (imports)
- [ ] Caching parsed trees across edits
