# Built-in Shape Semantics To Implement

This file tracks framework/library functions whose shape behavior we want the analyzer to understand.

Legend:

- **Classified**: `classify_known_function` recognizes the resolved function path.
- **Shape rule**: analyzer can actually infer/check output shapes for the function.
- **Tests**: shape-rule behavior has focused tests.

> Current state: many functions are classified as known, but shape rules are mostly not implemented yet. `equinox.nn.Linear` layer handling is implemented separately from this built-in-function registry.

## JAX higher-order functions

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `jax.vmap` / `equinox.filter_vmap` | ✅ | ✅ | ✅ | Free (`vf = vmap(f); vf(x)`), inline (`vmap(f)(x)`), and attribute (`vmap(self.layer)(x)`) forms; scalar `in_axes`/`out_axes` only. |
| `jax.jit` | ❌ | ❌ | ❌ | Usually shape-preserving wrapper. |
| `jax.grad` | ❌ | ❌ | ❌ | Higher-order; output shape depends on differentiated arg. |
| `jax.value_and_grad` | ❌ | ❌ | ❌ | Returns value plus gradient tree. |
| `jax.jacfwd` | ❌ | ❌ | ❌ | Adds Jacobian dimensions. |
| `jax.jacrev` | ❌ | ❌ | ❌ | Adds Jacobian dimensions. |
| `jax.hessian` | ❌ | ❌ | ❌ | Adds two derivative dimension groups. |
| `jax.pmap` | ❌ | ❌ | ❌ | Similar to `vmap`, device axis semantics. |
| `jax.lax.scan` | ✅ | ✅ | ✅ | Tuple LHS: final carry gets init's shape (carry invariant); stacked ys skipped (needs body output inference). |
| `jax.lax.map` | ❌ | ❌ | ❌ | Map over leading axis. |
| `jax.lax.cond` | ❌ | ❌ | ❌ | Branch output shape agreement. |
| `jax.lax.switch` | ❌ | ❌ | ❌ | Branch output shape agreement. |
| `jax.lax.while_loop` | ❌ | ❌ | ❌ | Carry shape invariant. |
| `jax.lax.fori_loop` | ❌ | ❌ | ❌ | Carry shape invariant. |

## JAX NumPy / NumPy array creation

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `jnp.array` / `np.array` | ✅ | ✅ | ✅ | Shape-preserving for known input shape; literal inference later. |
| `jnp.asarray` / `np.asarray` | ✅ | ✅ | ✅ | Shape-preserving conversion. |
| `jnp.zeros` / `np.zeros` | ✅ | ✅ | ✅ | Output shape from shape argument. |
| `jnp.ones` / `np.ones` | ✅ | ✅ | ✅ | Output shape from shape argument. |
| `jnp.full` / `np.full` | ✅ | ✅ | ✅ | Output shape from shape argument. |
| `jnp.empty` / `np.empty` | ✅ | ✅ | ✅ | Output shape from shape argument. |
| `jnp.zeros_like` / `np.zeros_like` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. |
| `jnp.ones_like` / `np.ones_like` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. |
| `jnp.full_like` / `np.full_like` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. |
| `jnp.empty_like` / `np.empty_like` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. |
| `jnp.arange` / `np.arange` | ✅ | ✅ | ✅ | 1D length from args when concrete. |
| `jnp.linspace` / `np.linspace` | ✅ | ✅ | ✅ | 1D length from `num`. |
| `jnp.logspace` / `np.logspace` | ✅ | ✅ | ✅ | 1D length from `num`. |
| `jnp.eye` / `np.eye` | ✅ | ✅ | ✅ | Matrix shape `(N, M?)`. |
| `jnp.identity` / `np.identity` | ✅ | ✅ | ✅ | Matrix shape `(n, n)`. |
| `jnp.diag` / `np.diag` | ✅ | ✅ | ✅ | 1D→2D or 2D→1D. |
| `jnp.diagflat` / `np.diagflat` | ❌ | ❌ | ❌ | Flatten then diagonal matrix. |
| `jnp.tri` / `np.tri` | ❌ | ❌ | ❌ | Matrix shape `(N, M?)`. |
| `jnp.tril` / `np.tril` | ✅ | ✅ | ✅ | Shape-preserving. |
| `jnp.triu` / `np.triu` | ✅ | ✅ | ✅ | Shape-preserving. |
| `jnp.meshgrid` / `np.meshgrid` | ✅ | ✅ | ✅ | Tuple LHS, default 'xy' indexing; keyword args skip. |
| `jnp.indices` / `np.indices` | ❌ | ❌ | ❌ | Prepends rank dimension. |

## JAX NumPy / NumPy shape transforms

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `jnp.reshape` / `np.reshape` | ✅ | ✅ | ✅ | Need size conservation / `-1` inference. |
| `jnp.ravel` / `np.ravel` | ✅ | ✅ | ✅ | Flatten to 1D. |
| `jnp.flatten` / `np.flatten` | ✅ | ✅ | ✅ | NumPy method mostly; function support varies. |
| `jnp.transpose` / `np.transpose` | ✅ | ✅ | ✅ | Permute axes. |
| `jnp.swapaxes` / `np.swapaxes` | ✅ | ✅ | ✅ | Swap two axes. |
| `jnp.moveaxis` / `np.moveaxis` | ✅ | ✅ | ✅ | Move axes preserving order. |
| `jnp.expand_dims` / `np.expand_dims` | ✅ | ✅ | ✅ | Insert size-1 axes. |
| `jnp.squeeze` / `np.squeeze` | ✅ | ✅ | ✅ | Remove size-1 axes. |
| `jnp.atleast_1d` / `np.atleast_1d` | ✅ | ✅ | ✅ | Promote scalar to rank 1. |
| `jnp.atleast_2d` / `np.atleast_2d` | ✅ | ✅ | ✅ | Promote to rank 2. |
| `jnp.atleast_3d` / `np.atleast_3d` | ✅ | ✅ | ✅ | Promote to rank 3. |
| `jnp.broadcast_to` / `np.broadcast_to` | ✅ | ✅ | ✅ | Output explicit target shape. |
| `jnp.broadcast_arrays` / `np.broadcast_arrays` | ✅ | ✅ | ✅ | Common broadcasted shape for all args. |
| `jnp.broadcast_shapes` / `np.broadcast_shapes` | ❌ | ❌ | ❌ | Returns shape tuple, not array. |
| `jnp.pad` / `np.pad` | ✅ | ✅ | ✅ | Modifies dimensions by pad widths. |
| `jnp.roll` / `np.roll` | ✅ | ✅ | ✅ | Shape-preserving. |
| `jnp.flip` / `np.flip` | ✅ | ✅ | ✅ | Shape-preserving. |
| `jnp.fliplr` / `np.fliplr` | ✅ | ✅ | ✅ | Shape-preserving, rank >= 2. |
| `jnp.flipud` / `np.flipud` | ✅ | ✅ | ✅ | Shape-preserving, rank >= 1. |
| `jnp.rot90` / `np.rot90` | ✅ | ✅ | ✅ | Usually swaps two axis lengths when odd `k`. |
| `jnp.rollaxis` / `np.rollaxis` | ❌ | ❌ | ❌ | Axis movement. |
| `jnp.resize` / `np.resize` | ❌ | ❌ | ❌ | Output explicit shape. |

## JAX NumPy / NumPy joining and splitting

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `jnp.concatenate` / `np.concatenate` | ✅ | ✅ | ✅ | Check all dims except concat axis; sum concat axis. |
| `jnp.concat` / `np.concat` | ✅ | ✅ | ✅ | Alias of concatenate in newer APIs. |
| `jnp.stack` / `np.stack` | ✅ | ✅ | ✅ | Check equal shapes; insert new axis. |
| `jnp.vstack` / `np.vstack` | ✅ | ✅ | ✅ | Stack along first axis after at least 2D. |
| `jnp.hstack` / `np.hstack` | ✅ | ✅ | ✅ | Concatenate along axis depending rank. |
| `jnp.dstack` / `np.dstack` | ✅ | ✅ | ✅ | Stack along depth axis. |
| `jnp.column_stack` / `np.column_stack` | ✅ | ✅ | ✅ | Stack 1D as columns. |
| `jnp.row_stack` / `np.row_stack` | ✅ | ✅ | ✅ | Alias-ish for vstack. |
| `jnp.block` / `np.block` | ✅ | ❌ | ❌ | Nested block assembly. |
| `jnp.split` / `np.split` | ✅ | ✅ | ✅ | Per-element shapes via `compute_split_shapes`; `a, b = split(...)` LHS bound in `analyze`. |
| `jnp.array_split` / `np.array_split` / `torch.tensor_split` | ✅ | ✅ | ✅ | Shares `compute_split_shapes` (CASE 1/2). |
| `jnp.hsplit` / `np.hsplit` | ❌ | ❌ | ❌ | Multiple outputs. |
| `jnp.vsplit` / `np.vsplit` | ❌ | ❌ | ❌ | Multiple outputs. |
| `jnp.dsplit` / `np.dsplit` | ❌ | ❌ | ❌ | Multiple outputs. |
| `jnp.tile` / `np.tile` | ✅ | ✅ | ✅ | Repeats dimensions. |
| `jnp.repeat` / `np.repeat` | ✅ | ✅ | ✅ | Repeats along axis or flattened. |

## JAX NumPy / NumPy indexing and selection

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `jnp.take` / `np.take` | ✅ | ✅ | ✅ | Shape based on indices and axis. |
| `jnp.take_along_axis` / `np.take_along_axis` | ❌ | ❌ | ❌ | Output shape matches indices. |
| `jnp.put_along_axis` / `np.put_along_axis` | ❌ | ❌ | ❌ | Mutating-ish; shape-preserving input. |
| `jnp.where` / `np.where` | ✅ | ✅ | ✅ | Broadcast condition/x/y. |
| `jnp.nonzero` / `np.nonzero` | ❌ | ❌ | ❌ | Tuple of index arrays. |
| `jnp.argwhere` / `np.argwhere` | ❌ | ❌ | ❌ | Shape `(N, ndim)`. |
| `jnp.argmax` / `np.argmax` | ✅ | ✅ | ✅ | Reduction via `apply_known_reduction`. |
| `jnp.argmin` / `np.argmin` | ✅ | ✅ | ✅ | Reduction via `apply_known_reduction`. |
| `jnp.argsort` / `np.argsort` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. |
| `jnp.sort` / `np.sort` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. |
| `jnp.searchsorted` / `np.searchsorted` | ❌ | ❌ | ❌ | Shape follows values. |
| `jnp.extract` / `np.extract` | ❌ | ❌ | ❌ | 1D unknown length. |
| `jnp.compress` / `np.compress` | ❌ | ❌ | ❌ | Axis length unknown unless condition concrete. |

## JAX NumPy / NumPy reductions

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `jnp.sum` / `np.sum` | ✅ | ✅ | ✅ | Remove/reduce axes; `keepdims` support. |
| `jnp.mean` / `np.mean` | ✅ | ✅ | ✅ | Same axis semantics. |
| `jnp.max` / `np.max` | ✅ | ✅ | ✅ | Same axis semantics. |
| `jnp.amax` / `np.amax` | ✅ | ✅ | ✅ | Alias max. |
| `jnp.min` / `np.min` | ✅ | ✅ | ✅ | Same axis semantics. |
| `jnp.amin` / `np.amin` | ✅ | ✅ | ✅ | Alias min. |
| `jnp.prod` / `np.prod` | ✅ | ✅ | ✅ | Same axis semantics. |
| `jnp.std` / `np.std` | ✅ | ✅ | ✅ | Same axis semantics. |
| `jnp.var` / `np.var` | ✅ | ✅ | ✅ | Same axis semantics. |
| `jnp.all` / `np.all` | ✅ | ✅ | ✅ | Reduction via `apply_known_reduction`. |
| `jnp.any` / `np.any` | ✅ | ✅ | ✅ | Reduction via `apply_known_reduction`. |
| `jnp.median` / `np.median` | ❌ | ❌ | ❌ | Same axis semantics. |
| `jnp.quantile` / `np.quantile` | ❌ | ❌ | ❌ | Adds quantile dims maybe. |
| `jnp.percentile` / `np.percentile` | ❌ | ❌ | ❌ | Similar to quantile. |
| `jnp.cumsum` / `np.cumsum` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. Does not model `axis=None` flattening (NumPy flattens to 1D); always preserves input shape. |
| `jnp.cumprod` / `np.cumprod` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. Does not model `axis=None` flattening (NumPy flattens to 1D); always preserves input shape. |
| `jnp.trace` / `np.trace` | ✅ | ✅ | ✅ | Removes two axes, optional offset. |

## Linear algebra

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `jnp.matmul` / `np.matmul` / `torch.matmul` | ✅ | ✅ | ✅ | Batch matmul broadcasting + inner dim check. |
| `jnp.dot` / `np.dot` / `torch.dot` | ✅ | ✅ | ✅ | Rank-dependent dot semantics. |
| `jnp.einsum` / `np.einsum` / `torch.einsum` | ✅ | ✅ | ✅ | Explicit-output equations; matmul/transpose/batched cases tested. |
| `jax.lax.dot` | ✅ | ✅ | ✅ | Dot semantics. |
| `jax.lax.dot_general` | ✅ | ✅ | ✅ | Currently approximated as matmul. |
| `jnp.vdot` / `np.vdot` | ✅ | ✅ | ✅ | Flattened dot — scalar output. |
| `jnp.tensordot` / `np.tensordot` / `torch.tensordot` | ✅ | ✅ | ✅ | Contract axes. Only int-axes form supported; tuple-of-lists form is a follow-up. |
| `jnp.outer` / `np.outer` / `torch.outer` | ✅ | ✅ | ✅ | Output `(a, b)`. |
| `jnp.inner` / `np.inner` | ✅ | ✅ | ✅ | Last-axis contraction. |
| `jnp.cross` / `np.cross` / `torch.cross` | ❌ | ❌ | ❌ | Vector axis length 2/3. |
| `jnp.linalg.norm` / `np.linalg.norm` / `torch.linalg.norm` | ❌ | ❌ | ❌ | Axis-dependent. |
| `jnp.linalg.det` / `np.linalg.det` / `torch.linalg.det` | ✅ | ✅ | ✅ | Square matrix -> batch shape; validates last two dims. |
| `jnp.linalg.inv` / `np.linalg.inv` / `torch.linalg.inv` | ✅ | ✅ | ✅ | Shape-preserving for square matrices; validates last two dims. |
| `jnp.linalg.solve` / `np.linalg.solve` / `torch.linalg.solve` | ❌ | ❌ | ❌ | Matrix solve shape rules. |
| `jnp.linalg.eig` / `np.linalg.eig` / `torch.linalg.eig` | ✅ | ✅ | ✅ | Tuple LHS: (n,), (n, n). `eigh` too. |
| `jnp.linalg.qr` / `np.linalg.qr` / `torch.linalg.qr` | ✅ | ✅ | ✅ | Tuple LHS, reduced mode: (m, min(m,n)), (min(m,n), n). |
| `jnp.linalg.svd` / `np.linalg.svd` / `torch.linalg.svd` | ✅ | ✅ | ✅ | Tuple LHS, full_matrices default: (m, m), (min(m,n),), (n, n); keyword args skip. |
| `jnp.linalg.cholesky` / `np.linalg.cholesky` / `torch.linalg.cholesky` | ❌ | ❌ | ❌ | Shape-preserving square matrix. |

## Torch array creation / transforms

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `torch.tensor` | ✅ | ✅ | ✅ | Shape-preserving for known input shape; literal inference later. |
| `torch.as_tensor` | ✅ | ✅ | ✅ | Shape-preserving conversion. |
| `torch.zeros` | ✅ | ✅ | ✅ | Output shape from args. |
| `torch.ones` | ✅ | ✅ | ✅ | Output shape from args. |
| `torch.full` | ✅ | ✅ | ✅ | Output shape from args. |
| `torch.empty` | ✅ | ✅ | ✅ | Output shape from args. |
| `torch.zeros_like` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. |
| `torch.ones_like` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. |
| `torch.full_like` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. |
| `torch.empty_like` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. |
| `torch.arange` | ✅ | ✅ | ✅ | 1D length from args if concrete. |
| `torch.linspace` | ✅ | ✅ | ✅ | 1D length from `steps`. |
| `torch.eye` | ✅ | ✅ | ✅ | Matrix shape. |
| `torch.cat` / `torch.concat` / `torch.concatenate` | ✅ | ✅ | ✅ | Concatenate along dim. |
| `torch.stack` | ✅ | ✅ | ✅ | Stack along new dim. |
| `torch.reshape` | ✅ | ✅ | ✅ | Reshape semantics. |
| `torch.flatten` | ✅ | ✅ | ✅ | Flatten dim range. |
| `torch.ravel` | ✅ | ✅ | ✅ | Flatten all dims. |
| `torch.transpose` | ✅ | ✅ | ✅ | Swap two dims. |
| `torch.permute` | ✅ | ✅ | ✅ | Permute all dims. |
| `torch.unsqueeze` | ✅ | ✅ | ✅ | Insert size-1 dim. |
| `torch.squeeze` | ✅ | ✅ | ✅ | Remove size-1 dims. |
| `torch.broadcast_to` | ✅ | ✅ | ✅ | Output target shape. |
| `torch.broadcast_tensors` | ✅ | ✅ | ✅ | Common broadcast shape. |
| `torch.tile` | ✅ | ✅ | ✅ | Repeat dims. |
| `torch.repeat` | ✅ | ✅ | ✅ | Tensor method and function nuance. |
| `torch.take` | ✅ | ✅ | ✅ | Output follows indices. |
| `torch.where` | ✅ | ✅ | ✅ | Broadcast condition/x/y. |
| `torch.meshgrid` | ✅ | ✅ | ✅ | Same rule as numpy meshgrid. |
| `torch.diag` | ✅ | ✅ | ✅ | 1D↔2D. |
| `torch.diagonal` | ✅ | ✅ | ✅ | Diagonal extraction. |
| `torch.trace` | ✅ | ✅ | ✅ | 2D -> scalar. |
| `torch.triu` | ✅ | ✅ | ✅ | Shape-preserving; shares `KnownFunction::Triu` path with `jnp.triu`. |
| `torch.tril` | ✅ | ✅ | ✅ | Shape-preserving; shares `KnownFunction::Tril` path with `jnp.tril`. |
| `torch.nn.functional.pad` | ✅ | ✅ | ✅ | Classified via `torch.nn.functional` deep module path; reuses `KnownFunction::Pad` and existing `apply_known_pad` pad-width parser. Note: `apply_known_pad` applies pad pairs in dimension order (dim 0 first), unlike PyTorch's reverse-axis convention; see `torch_nn_functional_pad_tests` in integration_tests.rs. |

## Torch reductions

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `torch.sum` | ✅ | ✅ | ✅ | `dim`, `keepdim`. |
| `torch.mean` | ✅ | ✅ | ✅ | `dim`, `keepdim`. |
| `torch.max` | ✅ | ✅ | ✅ | Reduction or elementwise depending args. |
| `torch.min` | ✅ | ✅ | ✅ | Reduction or elementwise depending args. |
| `torch.prod` | ✅ | ✅ | ✅ | `dim`, `keepdim`. |
| `torch.std` | ✅ | ✅ | ✅ | `dim`, `keepdim`. |
| `torch.var` | ✅ | ✅ | ✅ | `dim`, `keepdim`. |
| `torch.all` | ✅ | ✅ | ✅ | Reduction via `apply_known_reduction`. |
| `torch.any` | ✅ | ✅ | ✅ | Reduction via `apply_known_reduction`. |
| `torch.argmax` | ✅ | ✅ | ✅ | Reduction via `apply_known_reduction`. |
| `torch.argmin` | ✅ | ✅ | ✅ | Reduction via `apply_known_reduction`. |
| `torch.cumsum` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. Does not model `dim=None` flattening; always preserves input shape. |
| `torch.cumprod` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. Does not model `dim=None` flattening; always preserves input shape. |

## Neural-network layers / modules

| Function/Class | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `equinox.nn.Linear` | ✅ | ✅ | ✅ | Implemented as `LayerKind::Linear`. |
| `torch.nn.Linear` | ✅ | ✅ | ✅ | Same last-dim transform. |
| `equinox.nn.Embedding` / `torch.nn.Embedding` | ✅ | ✅ | ✅ | Appends the embedding dim to the input shape. |
| `flax.linen.Dense` | ✅ | ✅ | ✅ | Channels-last: last dim → features; input width runtime-inferred, nothing to check. |
| `equinox.nn.Conv1d` | ✅ | ✅ | ✅ | Channels-first layout only; per-axis tuples not yet supported. |
| `equinox.nn.Conv2d` | ✅ | ✅ | ✅ | Channels-first layout only; per-axis tuples not yet supported. |
| `equinox.nn.Conv3d` | ✅ | ✅ | ✅ | Channels-first layout only; per-axis tuples not yet supported. |
| `torch.nn.Conv1d` | ✅ | ✅ | ✅ | Channels-first layout only; per-axis tuples not yet supported; dilation≠1 gives approximate output. |
| `torch.nn.Conv2d` | ✅ | ✅ | ✅ | Channels-first layout only; per-axis tuples not yet supported; dilation≠1 gives approximate output. |
| `torch.nn.Conv3d` | ✅ | ✅ | ✅ | Channels-first layout only; per-axis tuples not yet supported; dilation≠1 gives approximate output. |
| `flax.linen.Conv` | ✅ | ✅ | ✅ | Channels-last, stride-1/SAME only (non-default strides refused); last dim → features. avg_pool/max_pool functions modelled (VALID, concrete dims). |
| `Dropout` variants | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`; Dropout1d/2d/3d enforce min rank (no batch required, matching Conv convention). |
| `BatchNorm` variants | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`; BatchNorm1d/2d/3d enforce min rank (no batch required, matching Conv convention); equinox BatchNorm is rank-agnostic. |
| `LayerNorm` variants | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`; LayerNorm enforces min rank 1. |
| `GroupNorm` variants | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`; GroupNorm enforces min rank 1. |
| activation functions/modules | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`; activations accept any rank. |
| pooling layers | ✅ | ✅ | ✅ | Max/Avg/Adaptive{Max,Avg}Pool 1d/2d/3d, channels-first; conv output formula (adaptive sets spatial dims to output_size). Torch stride default (= kernel_size). |
| `torch.nn.MultiheadAttention` | ✅ | ✅ | ✅ | Tuple LHS: output = query shape, weights = (…, L, S); default average_attn_weights. Other attention layers still ❌. |
| `equinox.nn.MultiheadAttention` | ✅ | ✅ | ✅ | **Bug fix**: `classify_layer_call` previously always read the torch `embed_dim` binding, so equinox ctors (real order `(num_heads, query_size, ...)`) silently failed to classify. `known_layer_signature` now branches on the resolved module path so equinox binds `query_size` (stored in the same `LayerKind::MultiheadAttention.feature_dim` field, renamed from `embed_dim`); torch's `(embed_dim, num_heads)` binding is unchanged — verified against the pre-existing torch MHA tests and `corpus/torch_attention.py`'s `nn.MultiheadAttention(512, 8)` shape, none of which changed. |
| `torch.nn.Flatten` | ✅ | ✅ | ✅ | Collapses `[start_dim, end_dim]` (negative indices resolved at apply time) into one dim; literal product when concrete, else `d0*d1*...`. Defaults `start_dim=1`, `end_dim=-1`. |
| `torch.nn.Unflatten` | ✅ | ✅ | ✅ | Expands `dim` into the parsed components of the `sizes` ctor tuple/list. Non-literal `dim` or unparseable `sizes` → unknown. |
| `equinox.nn.Identity` / `torch.nn.Identity` | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`, any rank. |
| `torch.nn.Upsample` | ✅ | ✅ | ✅ | Rank-agnostic ctor: spatial dims are whatever trails the leading (batch, channel) pair, determined from the actual input rank at apply time. Scales by `scale_factor` (scalar or per-axis tuple) or sets `size` directly; anything unparseable/mismatched-arity is unknown. Requires rank ≥ 3. |
| `equinox.nn.ConvTranspose1d/2d/3d` / `torch.nn.ConvTranspose1d/2d/3d` | ✅ | ✅ | ✅ | Shared `LayerKind::ConvTranspose`; channels-first, inverse of the conv formula: `out = (in-1)*stride - 2*padding + kernel`. `output_padding`/`dilation` not modelled (assumed 0/1); per-axis tuple kernel/stride/padding refused like Conv. |
| `torch.nn.RNN` / `torch.nn.LSTM` / `torch.nn.GRU` | ✅ | ✅ | ✅ | `LayerKind::Rnn`: models only the primary output tensor (last dim → `hidden_size`); the real return value is an `(output, final_state)` tuple, which needs tuple-unpacking support in `analysis.rs`'s `tuple_rhs_shapes` (outside this module's scope) to express — direct/non-tuple application is an approximation. `batch_first` isn't tracked: it doesn't change the formula since the feature dim is always trailing regardless of (seq, batch) order. `bidirectional`/`num_layers` not modelled. |
| `torch.nn.RNNCell` / `torch.nn.GRUCell` / `torch.nn.LSTMCell` / `equinox.nn.LSTMCell` / `equinox.nn.GRUCell` | ✅ | ✅ | ✅ | `LayerKind::RnnCell`: last dim → `hidden_size`, min rank 1 (single step, optionally unbatched). `LSTMCell` genuinely returns `(h, c)`; modelled as just `h` (same approximation reasoning as `Rnn`). |
| `torch.nn.InstanceNorm1d/2d/3d` | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`; min rank enforced like `BatchNorm1d/2d/3d` (channels-first, no batch dim required). |
| `equinox.nn.RMSNorm` / `torch.nn.RMSNorm` | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`; min rank 1, same convention as `LayerNorm`. |
| `torch.nn.PixelShuffle` | ✅ | ✅ | ✅ | `(*, C*r^2, H, W) -> (*, C, H*r, W*r)`; concrete integer `upscale_factor` required (else unknown), errors if channels aren't evenly divisible by `r^2`. |
| `torch.nn.PixelUnshuffle` | ✅ | ✅ | ✅ | Inverse of `PixelShuffle`: `(*, C, H*r, W*r) -> (*, C*r^2, H, W)`. |
| `torch.nn.ConstantPad1d/2d/3d` / `ZeroPad1d/2d/3d` / `ReflectionPad1d/2d/3d` / `ReplicationPad1d/2d/3d` | ✅ | ✅ | ✅ | `LayerKind::Pad`: pads the trailing `spatial_rank` dims. Models a concrete uniform int (every side of every spatial dim) or a fully-literal `2*spatial_rank`-length tuple (torch's reverse-axis pad-pair convention, last spatial dim first); symbolic padding is unknown. |
| `torch.nn.Bilinear` | ✅ | ✅ | ✅ | Only the first tracked positional input (`x1`) is validated/transformed (last dim vs. `in1_features` → `out_features`); `x2`/broadcasting not modelled — this analyzer only ever tracks one positional input per layer call (see `extract_layer_applications`). |
| `torch.nn.CosineSimilarity` | ✅ | ✅ | ✅ | Removes the reduced `dim` axis from `x1`'s shape (default `dim=1`); same single-tracked-input limitation as `Bilinear`. Non-literal `dim` → unknown. |
| `torch.nn.TransformerEncoderLayer` / `torch.nn.TransformerDecoderLayer` | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving` (`d_model` unchanged on the primary input). `TransformerDecoderLayer`'s second (`memory`) input isn't tracked (same single-input limitation as `Bilinear`). |
| `torch.nn.AlphaDropout` | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`, any rank — trivial, same family as `Dropout`. |
| `equinox.nn.MLP` | ✅ | ✅ | ✅ | `LayerKind::Mlp`: last-dim transform `in_size -> out_size`, same rule as `Linear`. |
| `torch.nn.Sequential` | ❌ | ❌ | ❌ | Composite container; needs sub-module propagation (walk the wrapped layer list and thread the shape through each). Not attempted. |
| `torch.nn.Fold` / `torch.nn.Unfold` | ❌ | ❌ | ❌ | im2col-style windowing transform; needs real shape reasoning about window/stride geometry, deliberately skipped. |
| `torch.nn.LocalResponseNorm` | ❌ | ❌ | ❌ | Actually shape-preserving in practice, but left unclassified per this task's explicit scope (conservative — only `AlphaDropout` was called out as the trivial one to do from that group). |
| `equinox.nn.Lambda` | ❌ | ❌ | ❌ | Wraps an arbitrary user callable; no static shape info available, so it's simply left unclassified (unclassified ≡ always-unknown, no behavioral difference from classifying-then-always-returning `Ok(None)`). |
| `equinox.nn.SpectralNorm` / `equinox.nn.WeightNorm` | ❌ | ❌ | ❌ | Wrapper modules around an inner layer; would need recursive sub-layer resolution (resolve the wrapped layer's `LayerKind` and delegate), out of scope here. |

## Method calls

`extract_method_calls` captures `var = receiver.method(args)`. `classify_method_call` maps the method name to a `KnownFunction`. `apply_method_call` synthesises the receiver as positional[0] and dispatches into `apply_known_function`, collapsing multiple positionals into a tuple-string for `Reshape`/`Permute`/`Transpose`. `analyze_layer_shapes` walks free calls + method calls in source order and propagates shapes through them.

| Method | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `x.reshape(...)` | ✅ | ✅ | ✅ | Multi-positional args collapsed to tuple. |
| `x.view(...)` | ✅ | ✅ | ✅ | Torch reshape alias. |
| `x.transpose(...)` | ✅ | ✅ | ✅ | No-args reverses; numpy-style full perm works; torch `transpose(i, j)` only correct for rank 2. |
| `x.permute(...)` | ✅ | ✅ | ✅ | Multi-positional dims collapsed. |
| `x.swapaxes(...)` | ✅ | ✅ | ✅ | Swap two axes. |
| `x.moveaxis(...)` | ✅ | ✅ | ✅ | Move axis preserving order. |
| `x.flatten(...)` | ✅ | ✅ | ✅ | Flatten all dims. |
| `x.ravel(...)` | ✅ | ✅ | ✅ | Flatten all dims. |
| `x.squeeze(...)` | ✅ | ✅ | ✅ | Remove size-1 dims. |
| `x.unsqueeze(...)` | ✅ | ✅ | ✅ | Insert size-1 dim (maps to `ExpandDims`). |
| `x.expand_dims(...)` | ✅ | ✅ | ✅ | Same as `unsqueeze`. |
| `x.sum(...)` | ✅ | ✅ | ✅ | Reduction; supports `axis`/`dim`/`keepdims`/`keepdim`. |
| `x.mean(...)` | ✅ | ✅ | ✅ | Reduction. |
| `x.max(...)` | ✅ | ✅ | ✅ | Reduction. |
| `x.min(...)` | ✅ | ✅ | ✅ | Reduction. |
| `x.prod(...)` | ✅ | ✅ | ✅ | Reduction. |
| `x.std(...)` | ✅ | ✅ | ✅ | Reduction. |
| `x.var(...)` | ✅ | ✅ | ✅ | Reduction. |
| `x.all(...)` | ✅ | ✅ | ✅ | Reduction. |
| `x.any(...)` | ✅ | ✅ | ✅ | Reduction. |
| `x.argmax(...)` | ✅ | ✅ | ✅ | Reduction. |
| `x.argmin(...)` | ✅ | ✅ | ✅ | Reduction. |
| `x.argsort(...)` | ✅ | ✅ | ✅ | Shape-preserving. |
| `x.sort(...)` | ✅ | ✅ | ✅ | Shape-preserving. |
| `x.cumsum(...)` | ✅ | ✅ | ✅ | Shape-preserving. |
| `x.cumprod(...)` | ✅ | ✅ | ✅ | Shape-preserving. |
| `x.repeat(...)` | ✅ | ✅ | ✅ | 1 positional arg → numpy repeat; ≥2 → torch tile semantics. `x.repeat_interleave` and `x.tile` too. |
| `x.expand(...)` | ✅ | ✅ | ✅ | Maps to broadcast_to; -1 keeps the input dim. `x.broadcast_to` too. |
| `x.to(...)` | ✅ | ✅ | ✅ | Shape-preserving (dtype/device cast). |

## Open targets (current)

Earlier suggested-order items (concatenate/stack, reductions, reshape/flatten,
transpose family, expand_dims/squeeze, broadcast_to, matmul/dot, array creation,
method-call equivalents, Linear/Conv layers) are all done. So are: `split`
tuple-unpacking + `L, _ = x.shape` (#30), `self.<field>` lookup (#31), nested/
chained calls (#40), vmap of attribute callables + inline vmap (#35), diagnostic
severity config (#45), and same-file cross-function tracing via `-> Float[...]`
return annotations.

**Prioritization is harness-driven** — run `cargo test corpus_coverage_report
-- --nocapture` (analyzes `corpus/*.py`, ranks un-shapeable assignments by
frequency, and lists each as `file:line kind`). Work the top-ranked dark spots
first; treat the ❌ catalog rows above as low-priority fill-in. Done so far via
this loop: `shape_of_subscript`, symbolic-dim normalization (`self.<attr>` ≡
`<attr>`), direct `self.layer(x)` calls, and `split` factor cancellation
(`d*3` split 3 → `d`). Corpus coverage is **100%** (75/75) across eleven corpus files; the dark spots
below are the current ranked gap list.

Current open work, roughly in impact order:

1. **Grow the corpus again** — 100% on the current eleven files. Known
   remaining approximations: scan's stacked `ys` output, strided flax Conv,
   non-default svd/qr/meshgrid modes.
2. **Cross-*file* return-type tracing** — same-file helpers already propagate;
   imported helpers don't yet.
3. **Remaining single-function shape rules** — see ❌ rows above
   (`diagflat`, `tri`, `indices`, `median`, `cross`, `linalg.solve`/`cholesky`,
   `hsplit`/`vsplit`/`dsplit`, `take_along_axis`, etc.).
