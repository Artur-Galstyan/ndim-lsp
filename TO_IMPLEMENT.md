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
| `jax.vmap` | ✅ | ❌ | ❌ | Higher-order; maps function over batch axes. |
| `jax.jit` | ❌ | ❌ | ❌ | Usually shape-preserving wrapper. |
| `jax.grad` | ❌ | ❌ | ❌ | Higher-order; output shape depends on differentiated arg. |
| `jax.value_and_grad` | ❌ | ❌ | ❌ | Returns value plus gradient tree. |
| `jax.jacfwd` | ❌ | ❌ | ❌ | Adds Jacobian dimensions. |
| `jax.jacrev` | ❌ | ❌ | ❌ | Adds Jacobian dimensions. |
| `jax.hessian` | ❌ | ❌ | ❌ | Adds two derivative dimension groups. |
| `jax.pmap` | ❌ | ❌ | ❌ | Similar to `vmap`, device axis semantics. |
| `jax.lax.scan` | ❌ | ❌ | ❌ | Loop/carry sequence semantics. |
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
| `jnp.empty` / `np.empty` | ❌ | ❌ | ❌ | Output shape from shape argument. |
| `jnp.zeros_like` / `np.zeros_like` | ❌ | ❌ | ❌ | Shape-preserving. |
| `jnp.ones_like` / `np.ones_like` | ❌ | ❌ | ❌ | Shape-preserving. |
| `jnp.full_like` / `np.full_like` | ❌ | ❌ | ❌ | Shape-preserving. |
| `jnp.empty_like` / `np.empty_like` | ❌ | ❌ | ❌ | Shape-preserving. |
| `jnp.arange` / `np.arange` | ✅ | ✅ | ✅ | 1D length from args when concrete. |
| `jnp.linspace` / `np.linspace` | ❌ | ❌ | ❌ | 1D length from `num`. |
| `jnp.logspace` / `np.logspace` | ❌ | ❌ | ❌ | 1D length from `num`. |
| `jnp.eye` / `np.eye` | ✅ | ✅ | ✅ | Matrix shape `(N, M?)`. |
| `jnp.identity` / `np.identity` | ❌ | ❌ | ❌ | Matrix shape `(n, n)`. |
| `jnp.diag` / `np.diag` | ✅ | ✅ | ✅ | 1D→2D or 2D→1D. |
| `jnp.diagflat` / `np.diagflat` | ❌ | ❌ | ❌ | Flatten then diagonal matrix. |
| `jnp.tri` / `np.tri` | ❌ | ❌ | ❌ | Matrix shape `(N, M?)`. |
| `jnp.tril` / `np.tril` | ✅ | ✅ | ✅ | Shape-preserving. |
| `jnp.triu` / `np.triu` | ✅ | ✅ | ✅ | Shape-preserving. |
| `jnp.meshgrid` / `np.meshgrid` | ✅ | ❌ | ❌ | Multiple output arrays; indexing mode matters. |
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
| `jnp.split` / `np.split` | ✅ | ❌ | ❌ | Multiple outputs. |
| `jnp.array_split` / `np.array_split` | ❌ | ❌ | ❌ | Multiple outputs, uneven sizes. |
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
| `jnp.argmax` / `np.argmax` | ❌ | ❌ | ❌ | Reduction over axis. |
| `jnp.argmin` / `np.argmin` | ❌ | ❌ | ❌ | Reduction over axis. |
| `jnp.argsort` / `np.argsort` | ❌ | ❌ | ❌ | Shape-preserving. |
| `jnp.sort` / `np.sort` | ❌ | ❌ | ❌ | Shape-preserving. |
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
| `jnp.all` / `np.all` | ❌ | ❌ | ❌ | Same axis semantics. |
| `jnp.any` / `np.any` | ❌ | ❌ | ❌ | Same axis semantics. |
| `jnp.median` / `np.median` | ❌ | ❌ | ❌ | Same axis semantics. |
| `jnp.quantile` / `np.quantile` | ❌ | ❌ | ❌ | Adds quantile dims maybe. |
| `jnp.percentile` / `np.percentile` | ❌ | ❌ | ❌ | Similar to quantile. |
| `jnp.cumsum` / `np.cumsum` | ❌ | ❌ | ❌ | Shape-preserving unless axis None -> flatten. |
| `jnp.cumprod` / `np.cumprod` | ❌ | ❌ | ❌ | Shape-preserving unless axis None -> flatten. |
| `jnp.trace` / `np.trace` | ✅ | ✅ | ✅ | Removes two axes, optional offset. |

## Linear algebra

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `jnp.matmul` / `np.matmul` / `torch.matmul` | ✅ | ✅ | ✅ | Batch matmul broadcasting + inner dim check. |
| `jnp.dot` / `np.dot` / `torch.dot` | ✅ | ✅ | ✅ | Rank-dependent dot semantics. |
| `jnp.einsum` / `np.einsum` / `torch.einsum` | ✅ | ❌ | ❌ | Equation parser needed. |
| `jax.lax.dot` | ✅ | ✅ | ✅ | Dot semantics. |
| `jax.lax.dot_general` | ✅ | ✅ | ✅ | Currently approximated as matmul. |
| `jnp.vdot` / `np.vdot` | ✅ | ✅ | ✅ | Flattened dot — scalar output. |
| `jnp.tensordot` / `np.tensordot` / `torch.tensordot` | ✅ | ✅ | ✅ | Contract axes. Only int-axes form supported; tuple-of-lists form is a follow-up. |
| `jnp.outer` / `np.outer` / `torch.outer` | ✅ | ✅ | ✅ | Output `(a, b)`. |
| `jnp.inner` / `np.inner` | ✅ | ✅ | ✅ | Last-axis contraction. |
| `jnp.cross` / `np.cross` / `torch.cross` | ❌ | ❌ | ❌ | Vector axis length 2/3. |
| `jnp.linalg.norm` / `np.linalg.norm` / `torch.linalg.norm` | ❌ | ❌ | ❌ | Axis-dependent. |
| `jnp.linalg.det` / `np.linalg.det` / `torch.linalg.det` | ❌ | ❌ | ❌ | Square matrix -> batch shape. |
| `jnp.linalg.inv` / `np.linalg.inv` / `torch.linalg.inv` | ❌ | ❌ | ❌ | Shape-preserving square matrix. |
| `jnp.linalg.solve` / `np.linalg.solve` / `torch.linalg.solve` | ❌ | ❌ | ❌ | Matrix solve shape rules. |
| `jnp.linalg.svd` / `np.linalg.svd` / `torch.linalg.svd` | ❌ | ❌ | ❌ | Multiple outputs. |
| `jnp.linalg.eig` / `np.linalg.eig` / `torch.linalg.eig` | ❌ | ❌ | ❌ | Multiple outputs. |
| `jnp.linalg.qr` / `np.linalg.qr` / `torch.linalg.qr` | ❌ | ❌ | ❌ | Multiple outputs. |
| `jnp.linalg.cholesky` / `np.linalg.cholesky` / `torch.linalg.cholesky` | ❌ | ❌ | ❌ | Shape-preserving square matrix. |

## Torch array creation / transforms

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `torch.tensor` | ✅ | ✅ | ✅ | Shape-preserving for known input shape; literal inference later. |
| `torch.as_tensor` | ✅ | ✅ | ✅ | Shape-preserving conversion. |
| `torch.zeros` | ✅ | ✅ | ✅ | Output shape from args. |
| `torch.ones` | ✅ | ✅ | ✅ | Output shape from args. |
| `torch.full` | ✅ | ✅ | ✅ | Output shape from args. |
| `torch.empty` | ❌ | ❌ | ❌ | Output shape from args. |
| `torch.zeros_like` | ❌ | ❌ | ❌ | Shape-preserving. |
| `torch.ones_like` | ❌ | ❌ | ❌ | Shape-preserving. |
| `torch.empty_like` | ❌ | ❌ | ❌ | Shape-preserving. |
| `torch.arange` | ✅ | ✅ | ✅ | 1D length from args if concrete. |
| `torch.linspace` | ❌ | ❌ | ❌ | 1D length from steps. |
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
| `torch.meshgrid` | ✅ | ❌ | ❌ | Multiple outputs. |
| `torch.diag` | ✅ | ✅ | ✅ | 1D↔2D. |
| `torch.diagonal` | ✅ | ✅ | ✅ | Diagonal extraction. |
| `torch.trace` | ✅ | ✅ | ✅ | 2D -> scalar. |
| `torch.triu` | ✅ | ❌ | ❌ | Shape-preserving. |
| `torch.tril` | ✅ | ❌ | ❌ | Shape-preserving. |
| `torch.nn.functional.pad` | ❌ | ❌ | ❌ | Need deeper module classification. |

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
| `torch.all` | ❌ | ❌ | ❌ | `dim`, `keepdim`. |
| `torch.any` | ❌ | ❌ | ❌ | `dim`, `keepdim`. |
| `torch.argmax` | ❌ | ❌ | ❌ | `dim`, `keepdim`. |
| `torch.argmin` | ❌ | ❌ | ❌ | `dim`, `keepdim`. |
| `torch.cumsum` | ❌ | ❌ | ❌ | Shape-preserving. |
| `torch.cumprod` | ❌ | ❌ | ❌ | Shape-preserving. |

## Neural-network layers / modules

| Function/Class | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `equinox.nn.Linear` | ✅ | ✅ | ✅ | Implemented as `LayerKind::Linear`. |
| `torch.nn.Linear` | ✅ | ✅ | ✅ | Same last-dim transform. |
| `flax.linen.Dense` | ❌ | ❌ | ❌ | Last-dim transform. |
| `equinox.nn.Conv1d` | ✅ | ✅ | ✅ | Channels-first layout only; per-axis tuples not yet supported. |
| `equinox.nn.Conv2d` | ✅ | ✅ | ✅ | Channels-first layout only; per-axis tuples not yet supported. |
| `equinox.nn.Conv3d` | ✅ | ✅ | ✅ | Channels-first layout only; per-axis tuples not yet supported. |
| `torch.nn.Conv1d` | ✅ | ✅ | ✅ | Channels-first layout only; per-axis tuples not yet supported; dilation≠1 gives approximate output. |
| `torch.nn.Conv2d` | ✅ | ✅ | ✅ | Channels-first layout only; per-axis tuples not yet supported; dilation≠1 gives approximate output. |
| `torch.nn.Conv3d` | ✅ | ✅ | ✅ | Channels-first layout only; per-axis tuples not yet supported; dilation≠1 gives approximate output. |
| `flax.linen.Conv` | ❌ | ❌ | ❌ | Spatial formula. |
| `Dropout` variants | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`; Dropout1d/2d/3d enforce min rank (no batch required, matching Conv convention). |
| `BatchNorm` variants | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`; BatchNorm1d/2d/3d enforce min rank (no batch required, matching Conv convention); equinox BatchNorm is rank-agnostic. |
| `LayerNorm` variants | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`; LayerNorm enforces min rank 1. |
| `GroupNorm` variants | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`; GroupNorm enforces min rank 1. |
| activation functions/modules | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`; activations accept any rank. |
| pooling layers | ❌ | ❌ | ❌ | Spatial formula. |
| attention layers | ❌ | ❌ | ❌ | Query/key/value shape rules. |

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
| `x.repeat(...)` | ❌ | ❌ | ❌ | Torch/Numpy semantics differ; out of scope for first method-call pass. |
| `x.expand(...)` | ❌ | ❌ | ❌ | Broadcast view. |
| `x.to(...)` | ❌ | ❌ | ❌ | Shape-preserving conversion; deferred. |

## Suggested implementation order

1. `concatenate` / `cat`
2. `stack`
3. reductions: `sum`, `mean`, `max`, `min`, `prod`, `std`, `var`
4. `reshape`, `flatten`, `ravel`
5. `transpose`, `swapaxes`, `moveaxis`, `permute`
6. `expand_dims` / `unsqueeze`, `squeeze`
7. `broadcast_to`, `broadcast_arrays`
8. `matmul`, `dot`
9. array creation: `zeros`, `ones`, `full`, `arange`, `eye`
10. `vmap`
11. method-call equivalents
12. more layer classes (`torch.nn.Linear`, `flax.linen.Dense`, convolutions)
