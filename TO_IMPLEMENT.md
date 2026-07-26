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
| `jax.jit` | ❌ | ❌ | ❌ | Out of scope: output depends on the wrapped function, not statically derivable with the current design. |
| `jax.grad` | ❌ | ❌ | ❌ | Out of scope: higher-order; output shape depends on the differentiated function, not statically derivable. |
| `jax.value_and_grad` | ❌ | ❌ | ❌ | Out of scope: same reason as `grad` (returns value + gradient tree). |
| `jax.jacfwd` | ❌ | ❌ | ❌ | Out of scope: same reason as `grad` (adds Jacobian dims from the wrapped function). |
| `jax.jacrev` | ❌ | ❌ | ❌ | Out of scope: same reason as `grad`. |
| `jax.hessian` | ❌ | ❌ | ❌ | Out of scope: same reason as `grad` (two derivative dimension groups). |
| `jax.pmap` | ❌ | ❌ | ❌ | Out of scope: device-mesh/axis semantics, not statically derivable with the current design. |
| `jax.lax.scan` | ✅ | ✅ | ✅ | Tuple LHS: final carry gets init's shape (carry invariant); stacked ys skipped (needs body output inference). |
| `jax.lax.map` | ✅ | ✅ | ✅ | Maps over the leading axis like `vmap(f)(xs)` with `in_axes=out_axes=0`; reuses `apply_vmap_call`/`apply_inline_vmap_layer` (bare user function or `self.<attr>` layer). Special-cased in `shape_of_call` (needs the callee-scope lookup machinery, not just `args`+`shapes`). |
| `jax.lax.cond` | ✅ | ❌ | ✅ | Classified; conservatively `Ok(None)` — branch output shape isn't derivable without analyzing both branch functions (out of scope for now). |
| `jax.lax.switch` | ✅ | ❌ | ✅ | Same as `cond` — classified, conservatively `Ok(None)`. |
| `jax.lax.while_loop` | ✅ | ✅ | ✅ | Carry invariant: output = `init_val`'s shape (3rd positional / `init_val` keyword). |
| `jax.lax.fori_loop` | ✅ | ✅ | ✅ | Carry invariant: output = `init_val`'s shape (4th positional / `init_val` keyword). |
| `jax.lax.conv_general_dilated` | ✅ | ✅ | ✅ | Default `dimension_numbers` only (`(batch, in_ch, *spatial)` / `(out_ch, in_ch, *spatial)`); supports `rhs_dilation`, `'SAME'`/`'VALID'`/explicit padding pairs, concrete dims. Any explicit `dimension_numbers`, non-trivial `lhs_dilation`, or symbolic spatial dims → `Ok(None)`. |
| `jax.lax.gather` | ✅ | ❌ | ✅ | Classified; conservatively `Ok(None)` — full `GatherDimensionNumbers`/`slice_sizes` modelling is out of scope for now (too large, like full einsum-path algebra). |
| `jax.lax.scatter` (and `scatter_add`/`scatter_mul`/`scatter_min`/`scatter_max`/`scatter_apply`) | ✅ | ✅ | ✅ | Shape-preserving on the `operand` (first arg). |
| `jax.lax.reduce_window` | ✅ | ✅ | ✅ | Pooling-style output shape from concrete `window_dimensions`/`window_strides`/`padding` (`'VALID'`, explicit `(low, high)` pairs, or omitted); `'SAME'` string and symbolic dims → `Ok(None)`. |
| `jax.lax.top_k` | ✅ | ✅ | ✅ | Tuple LHS (`values, indices`): both get `operand`'s shape with the last axis replaced by `k`. Extends `tuple_rhs_shapes`. |
| `jax.lax.sort` / `jax.lax.sort_key_val` | ✅ | ✅ | ✅ | `sort` (single operand): shape-preserving. `sort_key_val`: tuple LHS, each output preserves its own operand's shape (extends `tuple_rhs_shapes`). Multi-operand `sort` with tuple LHS isn't modelled. |
| `jax.lax.pad` | ✅ | ✅ | ✅ | Explicit per-axis `(low, high, interior)` padding_config triples; numeric dims support `interior != 0`, symbolic dims only when `interior == 0`. |
| `jax.lax.broadcast` / `jax.lax.broadcast_in_dim` | ✅ | ✅ | ✅ | `broadcast`: prepends `sizes` as leading dims. `broadcast_in_dim`: output is the explicit target `shape` arg. |
| `jax.lax.slice` / `jax.lax.dynamic_slice` / `jax.lax.dynamic_update_slice` | ✅ | ✅ | ✅ | `slice`: concrete `start_indices`/`limit_indices`/`strides`. `dynamic_slice`: output = `slice_sizes`. `dynamic_update_slice`: shape-preserving on `operand`. |
| `jax.lax.concatenate` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Concatenate` / `apply_known_concatenate` directly — same `(operands, dimension)` positional shape as `jnp.concatenate(arrays, axis)`. |
| `jax.lax.rev` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Flip`'s shape-preserving rule. |
| `jax.lax.squeeze` / `jax.lax.expand_dims` | ✅ | ✅ | ✅ | Reuse `KnownFunction::Squeeze`/`ExpandDims`; both now also accept the `dimensions` keyword and (for `expand_dims`) a *sequence* of axes, not just one. |
| `jax.lax.transpose` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Transpose`; `permutation` added as an accepted keyword alongside `axes`/`axis`/`dims`. |
| `jax.lax.associative_scan` | ✅ | ✅ | ✅ | Shape-preserving: output = `elems`'s shape (2nd positional / `elems` keyword). |
| `jax.lax.psum` / `jax.lax.pmean` / `jax.lax.all_gather` (collectives) | ❌ | ❌ | ❌ | Out of scope: device-mesh/axis semantics under `pmap`/`shard_map`, not statically derivable with the current design. |

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
| `jnp.diagflat` / `np.diagflat` | ✅ | ✅ | ✅ | Flattens then builds a square `(n, n)` diagonal matrix; `k` offset ignored (v1). |
| `jnp.tri` / `np.tri` | ✅ | ✅ | ✅ | Matrix shape `(N, M or N)`. |
| `jnp.tril` / `np.tril` | ✅ | ✅ | ✅ | Shape-preserving. |
| `jnp.triu` / `np.triu` | ✅ | ✅ | ✅ | Shape-preserving. |
| `jnp.meshgrid` / `np.meshgrid` | ✅ | ✅ | ✅ | Tuple LHS, default 'xy' indexing; keyword args skip. |
| `jnp.indices` / `np.indices` | ✅ | ✅ | ✅ | Prepends rank dimension: `indices((2, 3)).shape == (2, 2, 3)`. |
| `jnp.bincount` / `np.bincount` | ✅ | ✅ | ✅ | Classified; conservatively `Ok(None)` — length is data-dependent (max value / `minlength`), per the "unknown → `Ok(None)`" convention. |
| `jnp.unique` / `np.unique` | ✅ | ✅ | ✅ | Classified; conservatively `Ok(None)` — length is data-dependent. |
| `jnp.select` / `np.select` | ✅ | ✅ | ✅ | Approximated as the shape of `choicelist`'s first array. |

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
| `jnp.broadcast_shapes` / `np.broadcast_shapes` | ❌ | ❌ | ❌ | Genuinely out of scope: returns a plain shape *tuple*, not an array — not applicable to this analyzer's "shape of an array" model. |
| `jnp.pad` / `np.pad` | ✅ | ✅ | ✅ | Modifies dimensions by pad widths. |
| `jnp.roll` / `np.roll` | ✅ | ✅ | ✅ | Shape-preserving. |
| `jnp.flip` / `np.flip` | ✅ | ✅ | ✅ | Shape-preserving. |
| `jnp.fliplr` / `np.fliplr` | ✅ | ✅ | ✅ | Shape-preserving, rank >= 2. |
| `jnp.flipud` / `np.flipud` | ✅ | ✅ | ✅ | Shape-preserving, rank >= 1. |
| `jnp.rot90` / `np.rot90` | ✅ | ✅ | ✅ | Usually swaps two axis lengths when odd `k`. |
| `jnp.rollaxis` / `np.rollaxis` | ✅ | ✅ | ✅ | Rolls `axis` backward until it lies before position `start` (numpy semantics). |
| `jnp.resize` / `np.resize` | ✅ | ✅ | ✅ | Output is exactly the explicit `new_shape` arg. |
| `jnp.insert` / `np.insert` | ✅ | ✅ | ✅ | `axis` given + concrete insertion count (scalar literal → 1, literal list → its length, or a known array's own axis-length); `axis=None` (flattening) → `Ok(None)`. |
| `jnp.delete` / `np.delete` | ✅ | ✅ | ✅ | `axis` given + a single-int or literal-list `obj` (removes 1 / `len(obj)`); mask-based / `axis=None` deletion → `Ok(None)` (data-dependent). |
| `jnp.append` / `np.append` | ✅ | ✅ | ✅ | Like `concatenate` when `axis` given; flattens both operands to 1D and sums lengths when `axis=None`. |

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
| `jnp.block` / `np.block` | ✅ | ✅ | ✅ | Recursively parses the nested-list literal; concatenates the innermost level along the last axis, each level up along the next axis (numpy semantics: axis = `rank - nesting_depth`). |
| `jnp.split` / `np.split` | ✅ | ✅ | ✅ | Per-element shapes via `compute_split_shapes`; `a, b = split(...)` LHS bound in `analyze`. |
| `jnp.array_split` / `np.array_split` / `torch.tensor_split` | ✅ | ✅ | ✅ | Shares `compute_split_shapes` (CASE 1/2). |
| `jnp.hsplit` / `np.hsplit` | ✅ | ✅ | ✅ | Tuple LHS via `compute_fixed_axis_split_shapes` (forces axis=1, or axis=0 for rank-1 input) delegating to `compute_split_shapes`. |
| `jnp.vsplit` / `np.vsplit` | ✅ | ✅ | ✅ | Same mechanism, forces axis=0. |
| `jnp.dsplit` / `np.dsplit` | ✅ | ✅ | ✅ | Same mechanism, forces axis=2; errors if rank < 3. |
| `jnp.tile` / `np.tile` | ✅ | ✅ | ✅ | Repeats dimensions. |
| `jnp.repeat` / `np.repeat` | ✅ | ✅ | ✅ | Repeats along axis or flattened. |
| `jnp.kron` / `np.kron` | ✅ | ✅ | ✅ | Kronecker product; elementwise product of dims after left-padding the shorter shape's rank with 1s. |

## JAX NumPy / NumPy indexing and selection

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `jnp.take` / `np.take` | ✅ | ✅ | ✅ | Shape based on indices and axis. |
| `jnp.take_along_axis` / `np.take_along_axis` | ✅ | ✅ | ✅ | Output shape matches `indices` exactly (2nd positional). |
| `jnp.put_along_axis` / `np.put_along_axis` | ✅ | ✅ | ✅ | Shape-preserving on `arr` (first arg). |
| `jnp.where` / `np.where` | ✅ | ✅ | ✅ | Broadcast condition/x/y. |
| `jnp.nonzero` / `np.nonzero` | ✅ | ✅ | ✅ | Tuple LHS: one output per input dimension, all sharing an opaque `nonzero(<name>)` symbolic length (count is data-dependent; extends `tuple_rhs_shapes`). |
| `jnp.argwhere` / `np.argwhere` | ✅ | ✅ | ✅ | Shape `(nonzero(<name>), ndim)` — `ndim` is known statically from the input's rank; the element count is an opaque symbolic dim (data-dependent). |
| `jnp.argmax` / `np.argmax` | ✅ | ✅ | ✅ | Reduction via `apply_known_reduction`. |
| `jnp.argmin` / `np.argmin` | ✅ | ✅ | ✅ | Reduction via `apply_known_reduction`. |
| `jnp.argsort` / `np.argsort` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. |
| `jnp.sort` / `np.sort` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. |
| `jnp.searchsorted` / `np.searchsorted` | ✅ | ✅ | ✅ | Shape follows `v` (the values arg, 2nd positional), not `a`. |
| `jnp.extract` / `np.extract` | ✅ | ✅ | ✅ | Classified; conservatively `Ok(None)` — 1D length is data-dependent. |
| `jnp.compress` / `np.compress` | ✅ | ✅ | ✅ | Classified; conservatively `Ok(None)` — axis length is data-dependent. |
| `jnp.histogram` / `np.histogram` (and `histogram2d`/`histogramdd`) | ✅ | ✅ | ✅ | Output length = `bins` (int literal), `len(edges) - 1` (literal edge list), or the numpy/jax default (10) when omitted; non-literal `bins` → `Ok(None)`. `histogram2d`/`histogramdd` not classified. |

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
| `jnp.median` / `np.median` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Mean`'s axis/keepdims reduction rule directly (identical shape mechanics). |
| `jnp.quantile` / `np.quantile` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Mean`'s reduction rule (approximation: doesn't model the extra leading dim when `q` is itself an array, only the scalar-`q` case). |
| `jnp.percentile` / `np.percentile` | ✅ | ✅ | ✅ | Same as `quantile` (reuses `Mean`). |
| `jnp.cumsum` / `np.cumsum` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. Does not model `axis=None` flattening (NumPy flattens to 1D); always preserves input shape. |
| `jnp.cumprod` / `np.cumprod` | ✅ | ✅ | ✅ | Shape-preserving via `apply_known_shape_preserving`. Does not model `axis=None` flattening (NumPy flattens to 1D); always preserves input shape. |
| `jnp.trace` / `np.trace` | ✅ | ✅ | ✅ | Removes two axes, optional offset. |
| `jnp.count_nonzero` / `np.count_nonzero` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Sum`'s reduction rule (identical axis semantics). |
| `jnp.ptp` / `np.ptp` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Max`'s reduction rule (identical axis semantics). |
| `jnp.clip` / `np.clip` / `torch.clip` | ✅ | ✅ | — | Already shape-preserving via `is_shape_preserving_call` (elementwise, not a `KnownFunction` table entry). Audit note: do not re-add as a gap. |
| `jnp.nan_to_num` / `np.nan_to_num` | ✅ | ✅ | — | Added to `is_shape_preserving_call`'s numpy elementwise match list (shape-preserving, same path as `clip`). |
| `jnp.fft.*` / `np.fft.*` | ✅ | ✅ | ✅ | `fft`/`ifft`/`fft2`/`ifft2`/`fftn`/`ifftn`/`fftshift`/`ifftshift` classified as shape-preserving (module-path check in `classify_known_function`, reuses the `Copy` shape-preserving arm). `rfft`/`irfft`/`hfft` variants (which change the last-axis length) are intentionally left unclassified. |

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
| `jnp.cross` / `np.cross` / `torch.cross` | ✅ | ✅ | ✅ | Broadcasts like an elementwise binary op (reuses `broadcast_two_shapes`); the cross-product axis length (2 or 3) is unaffected. |
| `jnp.linalg.norm` / `np.linalg.norm` / `torch.linalg.norm` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Sum`'s reduction rule — axis/keepdims mechanics are identical to a plain reduction. |
| `jnp.linalg.det` / `np.linalg.det` / `torch.linalg.det` | ✅ | ✅ | ✅ | Square matrix -> batch shape; validates last two dims. |
| `jnp.linalg.inv` / `np.linalg.inv` / `torch.linalg.inv` | ✅ | ✅ | ✅ | Shape-preserving for square matrices; validates last two dims. |
| `jnp.linalg.solve` / `np.linalg.solve` / `torch.linalg.solve` | ✅ | ✅ | ✅ | Output shape = `b`'s shape (the right-hand side); `a` validated as square. |
| `jnp.linalg.eig` / `np.linalg.eig` / `torch.linalg.eig` | ✅ | ✅ | ✅ | Tuple LHS: (n,), (n, n). `eigh` too. |
| `jnp.linalg.qr` / `np.linalg.qr` / `torch.linalg.qr` | ✅ | ✅ | ✅ | Tuple LHS, reduced mode: (m, min(m,n)), (min(m,n), n). |
| `jnp.linalg.svd` / `np.linalg.svd` / `torch.linalg.svd` | ✅ | ✅ | ✅ | Tuple LHS, full_matrices default: (m, m), (min(m,n),), (n, n); keyword args skip. |
| `jnp.linalg.cholesky` / `np.linalg.cholesky` / `torch.linalg.cholesky` | ✅ | ✅ | ✅ | Reuses `KnownFunction::LinalgInv`'s rule directly — same square-preserving shape (Cholesky just requires SPD, which isn't shape-checkable, but the shape rule is identical). |
| `torch.linalg.lstsq` | ✅ | ✅ | ✅ | Tuple LHS (any prefix of `solution, residuals, rank, sv`): `solution` = `A`'s batch dims + `[n, k]` (matrix rhs) or `+ [n]` (vector rhs); the rest are algorithm/data-dependent → `None`. Extends `tuple_rhs_shapes`. |
| `torch.linalg.pinv` | ✅ | ✅ | ✅ | Swaps the last two dims: `(..., m, n) -> (..., n, m)`. |
| `torch.linalg.matrix_rank` | ✅ | ✅ | ✅ | Reduces last two dims to scalar (batched); no square requirement. |
| `torch.fft.*` | ✅ | ✅ | ✅ | Same module-path check as `jnp`/`np` `fft.*` — `fft`/`ifft`/`fft2`/`ifft2`/`fftn`/`ifftn`/`fftshift`/`ifftshift` shape-preserving; `rfft`/`irfft` (last-axis length change) intentionally unclassified. |

## einops

| Function | Classified | Shape rule | Tests | Notes |
|---|---:|---:|---:|---|
| `einops.rearrange` | ✅ | ✅ | ✅ | `apply_known_einops`; LHS groups bind axis names, RHS groups recompose them (composite groups solve one unknown factor by division). `...` (ellipsis) now supported as a standalone group on both sides — see the ellipsis row below. |
| `einops.reduce` | ✅ | ✅ | ✅ | Same pattern engine as `rearrange`; reduction op string (`"mean"`/`"sum"`/etc.) isn't shape-relevant. Ellipsis supported, same as `rearrange`. |
| `einops.repeat` | ✅ | ✅ | ✅ | Same pattern engine; new axes on the RHS sized from keyword args. Ellipsis supported, same as `rearrange`. |
| `einops.einsum` | ✅ | ✅ | ✅ | Own token-based engine (`apply_known_einops_einsum`) — einops equations use *space-separated* (possibly multi-char) axis names per operand, unlike `torch.einsum`/`jnp.einsum`'s single-char-per-axis notation, so it does not delegate to `apply_known_einsum`. Pattern string is the *last* positional arg. |
| `einops.pack` / `einops.unpack` | ✅ | ✅ | ✅ | `pack`: tuple LHS `(packed, ps)`; restricted to patterns where `*` stands for exactly one axis per tensor (the common single-batch-axis case), under which packing is exactly `concatenate` along the `*` position (`compute_einops_pack_shape`); `ps` is always `None` (a list of shape-tuples, not an array). Patterns with more than one `*`, or a variable-rank `*`, aren't modelled → `Ok(None)`. `unpack`: classified but genuinely dynamic (depends on the runtime `ps` value) — conservatively `None` for every unpacked target. |
| `einops.parse_shape` | ✅ | ✅ | ✅ | Classified; always `Ok(None)` — returns a dict of axis-name → size, not an array shape, so no shape is ever correctly inferable for the assigned variable (this *is* the correct rule, not a gap). |
| `einops.layers.torch.Rearrange` / `einops.layers.torch.Reduce` (and flax/equinox equivalents) | ❌ | ❌ | ❌ | Priority: medium. Layer-wrapper form of `rearrange`/`reduce` used inside `nn.Sequential`; would reuse `apply_known_einops`'s pattern engine once layer dispatch exists. Out of scope for this pass (layer dispatch lives in `src/layers.rs`, owned by a different task). |
| ellipsis (`...`) pattern support for `rearrange`/`reduce`/`repeat` | ✅ | ✅ | ✅ | `...` is parsed as a standalone group (`parse_einops_groups`) and, when present on both LHS and RHS, expands to `N` synthetic per-dim bindings (`__ellipsisK__`) seeded from the input's unmatched leading dims — passed through unchanged and in original order on the output side. Ellipsis on only one side, or nested inside a composite group (e.g. `(... h)`), isn't modelled (`Ok(None)`). `test_ellipsis_pattern_skips` now documents the one-sided case; full support is covered by `einops_ellipsis_tests` in `integration_tests.rs`. |

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
| `torch.gather` | ❌ | ❌ | ❌ | Priority: high. Output shape matches `index` tensor. |
| `torch.scatter` / `torch.scatter_add` | ❌ | ❌ | ❌ | Priority: high. Shape-preserving on `self`/`input`. |
| `torch.take_along_dim` | ❌ | ❌ | ❌ | Priority: high. Output shape matches `indices`. |
| `torch.topk` | ❌ | ❌ | ❌ | Priority: high. Tuple LHS; target dim becomes `k`. |
| `torch.unbind` | ❌ | ❌ | ❌ | Priority: high. Multiple outputs, removes one dim. |
| `torch.chunk` | ❌ | ❌ | ❌ | Priority: high. Multiple outputs, similar to `split`. |
| `torch.narrow` | ❌ | ❌ | ❌ | Priority: high. Slices one dim to `length`. |
| `torch.select` | ❌ | ❌ | ❌ | Priority: high. Removes one dim (index into it). |
| `torch.masked_select` | ❌ | ❌ | ❌ | Priority: high. 1D output, data-dependent length. |
| `torch.index_select` | ❌ | ❌ | ❌ | Priority: high. Dim length becomes `len(index)`. |
| `torch.nn.functional.interpolate` | ❌ | ❌ | ❌ | Priority: high. Spatial dims → `size`/`scale_factor`. |
| `torch.nn.functional.conv1d` / `conv2d` / `conv3d` | ❌ | ❌ | ❌ | Priority: high. Functional counterpart of `torch.nn.ConvNd`; same output formula, weight-derived channels. |
| `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` | ❌ | ❌ | ❌ | Priority: high. Functional counterpart of pooling layers. |
| `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` | ❌ | ❌ | ❌ | Priority: high. Functional counterpart of pooling layers. |
| `torch.nn.functional.softmax` / `log_softmax` | ❌ | ❌ | ❌ | Priority: high. Shape-preserving; only `torch.nn.functional.pad` is classified today. |
| `torch.nn.functional.one_hot` | ❌ | ❌ | ❌ | Priority: high. Appends `num_classes` dim. |
| `torch.combinations` | ❌ | ❌ | ❌ | Priority: medium. `(N choose r, r)` shape. |
| `torch.cartesian_prod` | ❌ | ❌ | ❌ | Priority: medium. Product of input lengths. |
| `torch.block_diag` | ❌ | ❌ | ❌ | Priority: medium. Sums block dims on the diagonal. |
| `torch.nn.functional.normalize` | ❌ | ❌ | ❌ | Priority: medium. Shape-preserving. |
| `torch.nn.functional.embedding` | ❌ | ❌ | ❌ | Priority: medium. Appends embedding dim, like `torch.nn.Embedding`. |
| `torch.nn.utils.rnn.pad_sequence` | ❌ | ❌ | ❌ | Priority: medium. Stacks variable-length sequences to `(max_len, batch, ...)` or `(batch, max_len, ...)`. |

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
| `torch.kthvalue` | ❌ | ❌ | ❌ | Priority: medium. Tuple LHS, reduces one dim (like `max`/`min`). |
| `torch.median` (with `dim`) / `torch.mode` (with `dim`) | ❌ | ❌ | ❌ | Priority: medium. Tuple LHS, reduces one dim; distinct from the no-`dim` scalar form. |
| `torch.unique` | ❌ | ❌ | ❌ | Priority: medium. 1D output, length data-dependent. |

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
| `flax.linen.avg_pool` | ✅ | ✅ | ✅ | `KnownFunction::FlaxPool`; channels-last, VALID padding, concrete dims only (`apply_known_flax_pool`). |
| `flax.linen.max_pool` | ✅ | ✅ | ⚠️ | Same `KnownFunction::FlaxPool` code path and rule as `avg_pool`; only `avg_pool` has a dedicated integration test today. |
| `jax.nn.relu` / `sigmoid` / `gelu` / `silu` / `swish` / `elu` / `leaky_relu` / `selu` / `softmax` / `log_softmax` / `standardize` (free functions) | ✅ | ✅ | ⚠️ | Not a `LayerKind`/`KnownFunction` — handled generically as shape-preserving elementwise via `is_shape_preserving_call` (module `["jax", "nn"]`); same code path also covers `flax.linen.<activation>` free functions. Only `jax.nn.softplus` has a dedicated test (`test_jax_nn_softplus_shape_preserving`), exercising the shared match arm; `test_relu_still_shape_preserving` (in `jax_nn_extra_tests`) is a regression check that carving `one_hot` out below didn't disturb this arm. |
| `jax.nn.one_hot` | ✅ | ✅ | ✅ | Carved out of the generic shape-preserving arm above (it is **not** shape-preserving — it appends `num_classes`): `KnownFunction::OneHot`, `x.shape + [num_classes]` (literal or symbolic `num_classes`). |
| `jax.nn.dot_product_attention` | ✅ | ✅ | ✅ | `KnownFunction::DotProductAttention` — output = `query`'s shape with the last (head) dim replaced by `value`'s head dim. |
| `jax.nn.logsumexp` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Sum`'s reduction rule (identical axis/keepdims mechanics). |
| `equinox.nn.MultiheadAttention` | ⚠️ | ⚠️ | ❌ | Priority: high. `classify_layer_call`/`known_layer_signature` dispatch on class name only, so `equinox.nn.MultiheadAttention` structurally matches the same `LayerKind::MultiheadAttention` arm as torch's — but the built-in binding contract assumes torch's positional order `(embed_dim, num_heads, ...)`, while equinox's real constructor is `(num_heads, query_size, ...)`. Untested; likely mis-binds `embed_dim` for real equinox code. Corpus/tests only use torch's `nn.MultiheadAttention`. |
| `equinox.nn.LSTMCell` / `GRUCell` | ❌ | ❌ | ❌ | Priority: high. Output is a new carry matching hidden size. |
| `equinox.nn.MLP` | ❌ | ❌ | ❌ | Priority: high. Last dim → `out_size`. |
| `equinox.nn.Sequential` | ❌ | ❌ | ❌ | Priority: high. Compose child layer shape rules in order. |
| `equinox.nn.ConvTranspose1d` / `ConvTranspose2d` / `ConvTranspose3d` | ❌ | ❌ | ❌ | Priority: high. Transposed-conv output formula (upsamples spatial dims). |
| `equinox.nn.RMSNorm` | ❌ | ❌ | ❌ | Priority: medium. Shape-preserving. |
| `equinox.nn.SpectralNorm` / `WeightNorm` | ❌ | ❌ | ❌ | Priority: medium. Shape-preserving wrappers. |
| `equinox.nn.Identity` | ❌ | ❌ | ❌ | Priority: medium. Shape-preserving. |
| `equinox.nn.Lambda` | ❌ | ❌ | ❌ | Priority: medium. Shape depends on wrapped function; usually unresolvable statically. |
| `torch.nn.RNN` / `LSTM` / `GRU` | ❌ | ❌ | ❌ | Priority: high. Sequence output plus hidden/cell state tuple. |
| `torch.nn.LSTMCell` / `GRUCell` / `RNNCell` | ❌ | ❌ | ❌ | Priority: high. Output is a new carry matching hidden size. |
| `torch.nn.ConvTranspose1d` / `ConvTranspose2d` / `ConvTranspose3d` | ❌ | ❌ | ❌ | Priority: high. Transposed-conv output formula (upsamples spatial dims). |
| `torch.nn.Flatten` / `Unflatten` | ❌ | ❌ | ❌ | Priority: high. Flatten a dim range / split one dim into many. |
| `torch.nn.Upsample` | ❌ | ❌ | ❌ | Priority: high. Spatial dims scaled by `scale_factor`/`size`. |
| `torch.nn.Identity` | ❌ | ❌ | ❌ | Priority: high. Shape-preserving. |
| `torch.nn.Sequential` | ❌ | ❌ | ❌ | Priority: high. Compose child layer shape rules in order. |
| `torch.nn.TransformerEncoderLayer` / `TransformerDecoderLayer` | ❌ | ❌ | ❌ | Priority: high. Shape-preserving on the sequence input. |
| `torch.nn.InstanceNorm1d` / `InstanceNorm2d` / `InstanceNorm3d` | ❌ | ❌ | ❌ | Priority: medium. Shape-preserving. |
| `torch.nn.RMSNorm` | ❌ | ❌ | ❌ | Priority: medium. Shape-preserving. |
| `torch.nn.PixelShuffle` / `PixelUnshuffle` | ❌ | ❌ | ❌ | Priority: medium. Trades channel dim for spatial resolution (factor `r`). |
| `torch.nn.ConstantPad*` / `ZeroPad*` / `ReflectionPad*` / `ReplicationPad*` | ❌ | ❌ | ❌ | Priority: medium. Same shape math as `torch.nn.functional.pad`. |
| `torch.nn.Bilinear` | ❌ | ❌ | ❌ | Priority: medium. Two inputs → last dim becomes `out_features`. |
| `torch.nn.CosineSimilarity` | ❌ | ❌ | ❌ | Priority: medium. Removes the reduced dim. |
| `torch.nn.Fold` / `Unfold` | ❌ | ❌ | ❌ | Priority: low. Inverse-of-conv patch (un)folding shape math. |
| `torch.nn.LocalResponseNorm` | ❌ | ❌ | ❌ | Priority: low. Shape-preserving. |
| `torch.nn.AlphaDropout` | ❌ | ❌ | ❌ | Priority: low. Shape-preserving (same family as other `Dropout` variants, not yet added to that match arm). |

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
| `x.astype(...)` | ✅ | ✅ | ✅ | Shape-preserving; `KnownFunction::Astype`. |
| `x.copy(...)` | ✅ | ✅ | ✅ | Shape-preserving; `KnownFunction::Copy`. |
| `x.detach(...)` | ✅ | ✅ | ✅ | Shape-preserving; `KnownFunction::Detach`. |
| `x.contiguous(...)` | ✅ | ✅ | ✅ | Shape-preserving; `KnownFunction::Contiguous`. |
| `x.chunk(...)` | ❌ | ❌ | ❌ | Priority: high. Multiple outputs, similar to `split`. |
| `x.unbind(...)` | ❌ | ❌ | ❌ | Priority: high. Multiple outputs, removes one dim. |
| `x.split(...)` | ❌ | ❌ | ❌ | Priority: high. Torch method form of `torch.split`/`compute_split_shapes` not yet wired to `apply_method_call`. |
| `x.gather(...)` | ❌ | ❌ | ❌ | Priority: high. Output shape matches `index`. |
| `x.scatter(...)` | ❌ | ❌ | ❌ | Priority: high. Shape-preserving. |
| `x.masked_select(...)` | ❌ | ❌ | ❌ | Priority: high. 1D output, data-dependent length. |
| `x.masked_fill(...)` | ❌ | ❌ | ❌ | Priority: high. Shape-preserving. |
| `x.index_select(...)` | ❌ | ❌ | ❌ | Priority: high. Dim length becomes `len(index)`. |
| `x.narrow(...)` | ❌ | ❌ | ❌ | Priority: high. Slices one dim to `length`. |
| `x.select(...)` | ❌ | ❌ | ❌ | Priority: high. Removes one dim. |
| `x.topk(...)` | ❌ | ❌ | ❌ | Priority: high. Target dim becomes `k`; multiple outputs. |
| `x.unfold(...)` | ❌ | ❌ | ❌ | Priority: high. Adds a new trailing window dim. |
| `x.view_as(...)` / `x.reshape_as(...)` / `x.expand_as(...)` | ❌ | ❌ | ❌ | Priority: high. Shape taken from the other tensor argument. |
| `x.flip(...)` | ❌ | ❌ | ❌ | Priority: high. Shape-preserving. |
| `x.roll(...)` | ❌ | ❌ | ❌ | Priority: high. Shape-preserving. |
| `x.T` / `x.mT` (properties) | ❌ | ❌ | ❌ | Priority: high. Reverse dims / swap last two dims; needs attribute (non-call) handling. |
| `x.item()` | ❌ | ❌ | ❌ | Priority: high. Scalar (rank-0) output. |
| `x.new_zeros(...)` / `new_ones(...)` / `new_full(...)` / `new_empty(...)` | ❌ | ❌ | ❌ | Priority: medium. Output shape from args, like `torch.zeros` family. |
| `x.clone(...)` | ❌ | ❌ | ❌ | Priority: medium. Shape-preserving. |
| `x.cpu(...)` / `x.cuda(...)` | ❌ | ❌ | ❌ | Priority: medium. Shape-preserving device moves. |
| `x.float(...)` / `long(...)` / `int(...)` / `bool(...)` / `double(...)` / `half(...)` | ❌ | ❌ | ❌ | Priority: medium. Shape-preserving dtype casts. |
| `x.clamp(...)` / `x.clip(...)` | ❌ | ❌ | ❌ | Priority: medium. Shape-preserving; not yet in `classify_method_call` (unlike the free-function `torch.clip`/`jnp.clip`, which are already handled generically). |
| `x.softmax(...)` | ❌ | ❌ | ❌ | Priority: medium. Shape-preserving. |
| `x.norm(...)` | ❌ | ❌ | ❌ | Priority: medium. Axis-dependent reduction. |
| `x.diagonal(...)` | ❌ | ❌ | ❌ | Priority: medium. Method form of `torch.diagonal`, not yet wired to `apply_method_call`. |
| `x.tril(...)` / `x.triu(...)` | ❌ | ❌ | ❌ | Priority: medium. Method form of `torch.tril`/`torch.triu`, not yet wired to `apply_method_call`. |

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
