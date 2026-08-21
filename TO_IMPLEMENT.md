# Built-in Shape Semantics To Implement

This file tracks framework/library functions whose shape behavior we want the analyzer to understand.

Legend:

- **Classified**: `classify_known_function` recognizes the resolved function path.
- **Shape rule**: analyzer can actually infer/check output shapes for the function.
- **Tests**: shape-rule behavior has focused tests.

> Current state: all catalog entries that fit the current array-shape model have shape rules. Tests cover each rule directly or through its shared dispatch path. Remaining ❌ entries need richer control-flow/value tracking or return non-array values. Layer handling uses a separate registry in `layers.rs`.

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
| `jax.lax.scan` | ✅ | ✅ | ✅ | Tuple LHS: final carry gets init's shape (carry invariant). `init` is evaluated as a real expression node, so a homogeneous-shape tuple carry (`(a, b)` with equal element shapes, inline at the call site or via a previously-bound tuple-literal variable — `shape_of_tuple`) resolves the same way as a plain single-array carry; heterogeneous-shape tuple carries stay `Ok(None)`. Stacked `ys` is now modelled too, via lazy call-site parameter seeding (see "Lazy call-site parameter seeding" below): when `body` is a same-file function/closure, its params are seeded (`carry` <- `init`'s shape, the per-step element <- `xs`'s shape minus its leading axis) and its body evaluated on demand (`scan_body_ys_shape` in analysis.rs); `ys` is the body's second return element with `xs`'s leading dim prepended back on. A qualified (`module.func`) or `self.<method>` body still isn't traced this way. |
| `jax.lax.map` | ✅ | ✅ | ✅ | Maps over the leading axis like `vmap(f)(xs)` with `in_axes=out_axes=0`; reuses `apply_vmap_call`/`apply_inline_vmap_layer` (bare user function or `self.<attr>` layer). Special-cased in `shape_of_call` (needs the callee-scope lookup machinery, not just `args`+`shapes`). |
| `jax.lax.cond` | ✅ | ❌ | ✅ | Classified; conservatively `Ok(None)` — branch output shape isn't derivable without analyzing both branch functions (out of scope for now). |
| `jax.lax.switch` | ✅ | ❌ | ✅ | Same as `cond` — classified, conservatively `Ok(None)`. |
| `jax.lax.while_loop` | ✅ | ✅ | ✅ | Carry invariant: output = `init_val`'s shape (3rd positional / `init_val` keyword). |
| `jax.lax.fori_loop` | ✅ | ✅ | ✅ | Carry invariant: output = `init_val`'s shape (4th positional / `init_val` keyword). |
| `jax.lax.conv_general_dilated` | ✅ | ✅ | ✅ | Default `dimension_numbers` only (`(batch, in_ch, *spatial)` / `(out_ch, in_ch, *spatial)`); supports `rhs_dilation`, `'SAME'`/`'VALID'`/explicit padding pairs, concrete dims. Any explicit `dimension_numbers`, non-trivial `lhs_dilation`, or symbolic spatial dims → `Ok(None)`. |
| `jax.lax.gather` | ✅ | ✅ | ✅ | `KnownFunction::LaxGather` / `apply_known_lax_gather`: models the common statically-derivable case — `dimension_numbers` is an inline `jax.lax.GatherDimensionNumbers(offset_dims=(...), collapsed_slice_dims=(...), start_index_map=(...))` literal (all three fields literal int tuples/lists) and `slice_sizes` is itself an inline literal int tuple/list at the call site. Output rank = `len(offset_dims) + (start_indices.rank - 1)`; offset-dims axes take, in order, the `slice_sizes` entries not in `collapsed_slice_dims`; every other output axis takes, in order, one of `start_indices`' leading (batch) dims (which may be symbolic) — `compute_lax_gather_shape`. `start_index_map`'s values aren't used in the shape formula (only which operand axis each index component targets) but are required to be present as a literal, matching the "fully statically-specified" form. Any variable/non-literal `dimension_numbers` or `slice_sizes`, or `operand_batching_dims`/`start_indices_batching_dims` (batched gather), falls back to `Ok(None)`. |
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
| `jnp.array_split` / `np.array_split` / `torch.tensor_split` | ✅ | ✅ | ✅ | Shares `compute_split_shapes` (CASE 1/2), section-*count* semantics. `torch.tensor_split` verified against torch docs to use the same section-count semantics as `jnp`/`np` (unlike `torch.split`, which is a distinct `KnownFunction::TorchSplit` — see the Torch section below). |
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
| `einops.pack` / `einops.unpack` | ✅ | ✅ (pack) / ❌ (unpack) | ✅ | `pack`: tuple LHS `(packed, ps)`; restricted to patterns where `*` stands for exactly one axis per tensor (the common single-batch-axis case), under which packing is exactly `concatenate` along the `*` position (`compute_einops_pack_shape`); `ps` is always `None` (a list of shape-tuples, not an array). Patterns with more than one `*`, or a variable-rank `*`, aren't modelled → `Ok(None)`. `unpack`: investigated for a tuple-LHS `a, b = unpack(x, ps, pattern)` rule (mirroring `pack`'s single-`*`-axis restriction: `ps`'s entries would be literal per-tensor sizes along that axis, output = packed shape with `*` replaced by the entry) but left unimplemented — `ps` is, in essentially all real code, the tuple *returned* by a paired `pack()` call, i.e. a plain identifier at the `unpack` call site, not a literal; resolving it to concrete dims would need a new "track an arbitrary literal Python value bound to a name" mechanism that doesn't exist anywhere in this codebase (only tensor *shapes* are tracked, via `ShapeLookup`/`ctx.resolve_shape`), which is disproportionate for a pattern this narrow. The mechanically-cheap sub-case — `ps` written directly inline in the call, e.g. `unpack(x, [[3], [5]], 'b * c')`, parseable with the existing `parse_nested_int_tuples` helper with no new architecture — isn't real code either (it defeats pack/unpack's whole purpose of avoiding manual shape bookkeeping), so also left unimplemented. Still classified, conservatively `Ok(None)` for every unpacked target. |
| `einops.parse_shape` | ✅ | ✅ | ✅ | Classified; always `Ok(None)` — returns a dict of axis-name → size, not an array shape, so no shape is ever correctly inferable for the assigned variable (this *is* the correct rule, not a gap). |
| `einops.layers.torch.Rearrange` / `einops.layers.torch.Reduce` (and the `einops.layers.flax` equivalents) | ✅ | ✅ | ✅ | `LayerKind::EinopsPattern { name, pattern, kwargs }`: `classify_layer_call` binds `pattern` (and, for `Reduce`, the shape-irrelevant `reduction` string) via a `known_layer_signature` catalog entry gated on `einops.layers.{torch,flax}`; any other keyword ctor args (axis-length bindings like `h=14`) land in `kwargs` for free since `bind_call_arguments` inserts *any* keyword arg regardless of declared param name. Application delegates to the existing `apply_known_einops` pattern engine (marked `pub(crate)`, no logic changes) via a one-entry `SingleShapeLookup` adapter, so the free-function and layer-object forms share one algebra. `einops.layers.chainer` and unrecognised keyword-only patterns are not modelled. |
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
| `torch.gather` | ✅ | ✅ | ✅ | `KnownFunction::Gather`; output shape matches `index` (always the last positional arg in both the free-function and method forms). |
| `torch.scatter` / `torch.scatter_add` | ✅ | ✅ | ✅ | `KnownFunction::Scatter`; shape-preserving on `self`/`input` via `apply_known_shape_preserving`. |
| `torch.take_along_dim` | ✅ | ✅ | ✅ | Reuses `KnownFunction::TakeAlongAxis` directly — identical "output matches `indices`" rule as `jnp.take_along_axis`. |
| `torch.topk` | ✅ | ✅ | ✅ | `KnownFunction::TopK`; tuple LHS via `apply_known_topk_shape` (extends `tuple_rhs_shapes`) — both outputs replace `dim` with `k`. Single-assignment dispatch is conservatively `Ok(None)` (real return is always a 2-tuple). |
| `torch.unbind` | ✅ | ✅ | ✅ | `KnownFunction::Unbind`; tuple LHS via `compute_unbind_shape` — removes `dim`; `n_targets` must match the axis's literal size (mismatch → error), symbolic axis → unknown. |
| `torch.chunk` | ✅ | ✅ | ✅ | `KnownFunction::Chunk`; tuple LHS via `compute_chunk_shapes` — at most `chunks` pieces of `ceil(size/chunks)` (last piece may be smaller), unlike `split`'s exact-division semantics. |
| `torch.split` | ✅ | ✅ | ✅ | `KnownFunction::TorchSplit` (distinct from `KnownFunction::Split`, which `torch.tensor_split` still uses); real semantics via `compute_torch_split_shapes` — the 2nd arg is a chunk **size**, not a section count: `ceil(axis_dim / size)` chunks, last one a smaller remainder when it doesn't divide evenly (mirrors `compute_chunk_shapes` with size/count roles swapped). Also accepts a literal list of explicit per-chunk sizes (at most one `-1` entry infers the remainder). Literal axis dim + literal size → exact chunk list. Symbolic axis dim + literal size → count is derived from the LHS tuple arity when known (`a, b, c = x.split(n)`); with no arity hint (single-LHS validate path) → `Ok(None)`. Non-literal, non-list size → always `Ok(None)` (per-chunk size itself unknown). |
| `torch.narrow` | ✅ | ✅ | ✅ | `KnownFunction::Narrow`; slices `dim` down to `length`. |
| `torch.select` | ✅ | ✅ | ✅ | `KnownFunction::SelectDim` (distinct from `jnp.select`'s choicelist-based `KnownFunction::Select`); removes `dim`. |
| `torch.masked_select` | ✅ | ✅ | ✅ | `KnownFunction::MaskedSelect`; classified, conservatively `Ok(None)` — length is data-dependent. |
| `torch.index_select` | ✅ | ✅ | ✅ | `KnownFunction::IndexSelect`; `dim`'s length becomes `len(index)`. |
| `torch.nn.functional.interpolate` | ✅ | ✅ | ✅ | `KnownFunction::Interpolate`; sets spatial dims from `size`, or scales by `scale_factor` (floor-rounded). |
| `torch.nn.functional.conv1d` / `conv2d` / `conv3d` | ✅ | ✅ | ✅ | `KnownFunction::FunctionalConv1d/2d/3d`; same conv formula as the layer forms, channels/kernel derived from the `weight` arg's shape (grouped-conv channel mismatch not validated, matching `jax.lax.conv_general_dilated`'s existing approximation). |
| `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` | ✅ | ✅ | ✅ | `KnownFunction::FunctionalMaxPool1d/2d/3d`; `stride` defaults to `kernel_size` per torch convention; `dilation` not modelled. |
| `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` | ✅ | ✅ | ✅ | `KnownFunction::FunctionalAvgPool1d/2d/3d`; same pooling formula/helper as the max-pool functional forms. |
| `torch.nn.functional.softmax` / `log_softmax` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Copy`'s shape-preserving rule. |
| `torch.nn.functional.one_hot` | ✅ | ✅ | ✅ | Reuses `KnownFunction::OneHot` directly; the `num_classes=-1` "infer at runtime" sentinel is now treated as unknown (harmless no-op for `jax.nn.one_hot`, which never passes `-1`). |
| `torch.combinations` | ✅ | ✅ | ✅ | `KnownFunction::Combinations`; `(nCr(n, r), r)` from a literal 1D input length; `with_replacement` not modelled. |
| `torch.cartesian_prod` | ✅ | ✅ | ✅ | `KnownFunction::CartesianProd`; `(prod(lengths), num_tensors)`, or the input itself when only one tensor given. |
| `torch.block_diag` | ✅ | ✅ | ✅ | `KnownFunction::BlockDiag`; sums row/col dims onto the diagonal; 1D inputs treated as a single row; rank >= 3 unmodelled. |
| `torch.nn.functional.normalize` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Copy`'s shape-preserving rule. |
| `torch.nn.functional.embedding` | ✅ | ✅ | ✅ | `KnownFunction::FunctionalEmbedding`; appends `weight`'s last-axis (embedding) dim to `input`'s shape. |
| `torch.nn.functional.relu` / `relu6` / `gelu` / `silu` / `sigmoid` / `tanh` / `softplus` / `softsign` / `selu` / `celu` / `elu` / `leaky_relu` / `hardtanh` / `hardswish` / `hardsigmoid` / `mish` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Copy`'s shape-preserving rule, same as `softmax`/`log_softmax`/`normalize` above. |
| `torch.nn.functional.dropout` / `dropout2d` / `dropout3d` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Copy`'s shape-preserving rule. |
| `torch.nn.functional.glu` | ✅ | ✅ | ✅ | `KnownFunction::FunctionalGlu`; halves `dim` (default last axis, must be even-sized): numeric divide-by-2, symbolic `x*2` factor cancellation (via `cancel_product_factor`, shared with `Split`), else opaque `glu(<dim>)` name. Not shape-preserving, unlike the rest of the activation family. |
| `torch.nn.utils.rnn.pad_sequence` | ✅ | ✅ | ✅ | `KnownFunction::PadSequence`; batch count + trailing dims known from the literal sequence list, padded length emitted as opaque symbolic `pad_len` (data-dependent). |

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
| `torch.kthvalue` | ✅ | ✅ | ✅ | `KnownFunction::KthValue`; tuple LHS via `apply_known_kthvalue_shape` — `values`/`indices` share a plain single-axis reduction over `dim` (re-synthesizes `(input, dim=.., keepdim=..)`, dropping `k`, and reuses `apply_known_reduction`). |
| `torch.median` (with `dim`) / `torch.mode` (with `dim`) | ✅ | ✅ | ✅ | `KnownFunction::MedianDim`; tuple LHS reuses `apply_known_reduction` directly (same `(input, dim, keepdim)` layout as a plain reduction); the no-`dim` scalar form (2 targets but no `dim`/`axis` arg) is skipped as not a real 2-tuple. |
| `torch.unique` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Unique` directly (classified for `torch` alongside the existing `jnp`/`np` mapping); conservatively `Ok(None)` — length is data-dependent. |

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
| `torch.nn.MultiheadAttention` | ✅ | ✅ | ✅ | Tuple LHS: output = query shape, weights = (…, L, S); default average_attn_weights. Single-assignment LHS (`out = self.attn(q, k, v)`, no tuple unpacking) also binds — `apply_layer_application` still returns `Ok(None)` for this kind (its `(output, weights)` return can't be expressed as one shape), so `shape_of_call`'s direct self-attr layer-call path special-cases `MultiheadAttention` to bind the primary output (query's shape) directly, bypassing `apply_layer_application`. Other attention layers still ❌. |
| `flax.linen.avg_pool` | ✅ | ✅ | ✅ | `KnownFunction::FlaxPool`; channels-last, VALID padding, concrete dims only (`apply_known_flax_pool`). |
| `flax.linen.max_pool` | ✅ | ✅ | ⚠️ | Same `KnownFunction::FlaxPool` code path and rule as `avg_pool`; only `avg_pool` has a dedicated integration test today. |
| `jax.nn.relu` / `sigmoid` / `gelu` / `silu` / `swish` / `elu` / `leaky_relu` / `selu` / `softmax` / `log_softmax` / `standardize` (free functions) | ✅ | ✅ | ⚠️ | Not a `LayerKind`/`KnownFunction` — handled generically as shape-preserving elementwise via `is_shape_preserving_call` (module `["jax", "nn"]`); same code path also covers `flax.linen.<activation>` free functions. Only `jax.nn.softplus` has a dedicated test (`test_jax_nn_softplus_shape_preserving`), exercising the shared match arm; `test_relu_still_shape_preserving` (in `jax_nn_extra_tests`) is a regression check that carving `one_hot` out below didn't disturb this arm. |
| `jax.nn.one_hot` | ✅ | ✅ | ✅ | Carved out of the generic shape-preserving arm above (it is **not** shape-preserving — it appends `num_classes`): `KnownFunction::OneHot`, `x.shape + [num_classes]` (literal or symbolic `num_classes`). |
| `jax.nn.dot_product_attention` | ✅ | ✅ | ✅ | `KnownFunction::DotProductAttention` — output = `query`'s shape with the last (head) dim replaced by `value`'s head dim. |
| `jax.nn.logsumexp` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Sum`'s reduction rule (identical axis/keepdims mechanics). |
| `equinox.nn.MultiheadAttention` | ✅ | ✅ | ✅ | **Bug fix**: `classify_layer_call` previously always read the torch `embed_dim` binding, so equinox ctors (real order `(num_heads, query_size, ...)`) silently failed to classify. `known_layer_signature` now branches on the resolved module path so equinox binds `query_size` (stored in the same `LayerKind::MultiheadAttention.feature_dim` field, renamed from `embed_dim`); torch's `(embed_dim, num_heads)` binding is unchanged — verified against the pre-existing torch MHA tests and `corpus/torch_attention.py`'s `nn.MultiheadAttention(512, 8)` shape, none of which changed. |
| `torch.nn.Flatten` | ✅ | ✅ | ✅ | Collapses `[start_dim, end_dim]` (negative indices resolved at apply time) into one dim; literal product when concrete, else `d0*d1*...`. Defaults `start_dim=1`, `end_dim=-1`. |
| `torch.nn.Unflatten` | ✅ | ✅ | ✅ | Expands `dim` into the parsed components of the `sizes` ctor tuple/list. Non-literal `dim` or unparseable `sizes` → unknown. |
| `equinox.nn.Identity` / `torch.nn.Identity` | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`, any rank. |
| `torch.nn.Upsample` | ✅ | ✅ | ✅ | Rank-agnostic ctor: spatial dims are whatever trails the leading (batch, channel) pair, determined from the actual input rank at apply time. Scales by `scale_factor` (scalar or per-axis tuple) or sets `size` directly; anything unparseable/mismatched-arity is unknown. Requires rank ≥ 3. |
| `equinox.nn.ConvTranspose1d/2d/3d` / `torch.nn.ConvTranspose1d/2d/3d` | ✅ | ✅ | ✅ | Shared `LayerKind::ConvTranspose`; channels-first, inverse of the conv formula: `out = (in-1)*stride - 2*padding + kernel`. `output_padding`/`dilation` not modelled (assumed 0/1); per-axis tuple kernel/stride/padding refused like Conv. |
| `torch.nn.RNN` / `torch.nn.LSTM` / `torch.nn.GRU` | ✅ | ✅ | ✅ | `LayerKind::Rnn`. Direct/non-tuple application (`out = self.rnn(x)`) approximates the real `(output, final_state)` return as just the primary output (last dim → `hidden_size`). Tuple LHS (`out, h = self.gru(x)`, `out, (h, c) = self.lstm(x)`) is now modelled in `analysis.rs`'s `tuple_rhs_shapes` (the `self.<attr>` → `LayerKind::Rnn` arm): element 0 is the same primary-output rule; element 1 (the state — LSTM's nested `(h, c)` binds both names to this one shape via one level of LHS tuple-pattern nesting, `TupleTarget::Nested`) approximates torch's `h_n`/`c_n` by dropping the leading (sequence) axis and replacing the last dim with `hidden_size`. This assumes `batch_first=False` (torch's default; not tracked) and `num_layers * num_directions == 1` (not tracked), so for a batched input it's missing the real leading `num_layers*num_directions` axis torch's actual state carries — a stated simplification, not the exact torch shape. `batch_first` doesn't change the *output* formula (the feature dim is always trailing). `bidirectional`/`num_layers` not modelled anywhere. |
| `torch.nn.RNNCell` / `torch.nn.GRUCell` / `torch.nn.LSTMCell` / `equinox.nn.LSTMCell` / `equinox.nn.GRUCell` | ✅ | ✅ | ✅ | `LayerKind::RnnCell`: last dim → `hidden_size`, min rank 1 (single step, optionally unbatched). Single-LHS form models just the primary output (`h`, same approximation reasoning as `Rnn`); a tuple LHS (`h, c = self.lstm(x, (h0, c0))`) extends `tuple_rhs_shapes` to bind *both* elements to that same `hidden_size`-last-dim shape (mirrors the `Rnn` arm's `(h, c)` state handling). |
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
| `torch.nn.Sequential` / `equinox.nn.Sequential` | ✅ | ✅ | ✅ | `LayerKind::Sequential { children: Vec<LayerKind> }`. Classification happens at the node level in `layers.rs` (`try_classify_sequential`, hooked into `resolve_layer_kind_for_call` ahead of the normal catalog/disk path): each ctor argument is resolved as its own nested layer-constructor call via `classify_nested_layer_node`, reusing the same catalog-first/disk-fallback resolution recursively — so nested `Sequential`s classify for free. torch's variadic-positional form (`nn.Sequential(l1, l2, ...)`) and equinox's single-list form (`eqx.nn.Sequential([l1, l2, ...])`) are both unwrapped via `sequential_child_nodes`. Any ctor argument that isn't itself a resolvable `call` node (arbitrary callable, `*args` unpacking, an unclassifiable layer) makes the *whole* `Sequential` unclassified (`Ok(None)`), never a partial/guessed result. Application (`apply_layer_kind`'s `Sequential` arm) threads the shape through each child via the same function recursively; an unknown (`Ok(None)`) child makes the chain unknown, and a child's `Err` propagates verbatim with an index-qualified layer label (`"net[1]"`) so a mid-chain mismatch reports as that child's own error. |
| `torch.nn.Unfold` | ✅ | ✅ | ✅ | `LayerKind::Unfold`: channels-first `(C, H, W)` / `(N, C, H, W)` im2col — `(C*kh*kw, L)` / `(N, C*kh*kw, L)`, `L = num_blocks_h * num_blocks_w` via the same block-count formula as `conv_spatial_dim` (dilation folded in as an effective kernel size `d*(k-1)+1`). `kernel_size`/`stride`/`padding`/`dilation` each accept a scalar or an explicit 2-tuple. Exact when H/W and the geometry args are literal ints, or when `dilation` is literal `1` (folds away even against a symbolic kernel size); any other non-literal `dilation` combination is unknown. |
| `torch.nn.Fold` | ✅ | ✅ | ✅ | `LayerKind::Fold`: inverse of `Unfold` — `(C*kh*kw, L)` / `(N, C*kh*kw, L)` → `(C, oh, ow)` / `(N, C, oh, ow)`. Channel dim recovered by dividing the input's channel dim by `kh*kw` (`Err` when literal and not evenly divisible, a symbolic `dim/(kh*kw)` string otherwise); `L` itself isn't cross-checked against `output_size`/kernel geometry. |
| `torch.nn.LocalResponseNorm` | ✅ | ✅ | ✅ | Shape-preserving via `LayerKind::ShapePreserving`, any rank — trivial, same family as `AlphaDropout`/`Dropout`. |
| `equinox.nn.Lambda` | ❌ | ❌ | ❌ | Wraps an arbitrary user callable; no static shape info available, so it's simply left unclassified (unclassified ≡ always-unknown, no behavioral difference from classifying-then-always-returning `Ok(None)`). |
| `equinox.nn.SpectralNorm` / `equinox.nn.WeightNorm` | ✅ | ✅ | ✅ | Transparent delegation, no new `LayerKind` variant: `try_classify_inner_layer_wrapper` (node-level, same recursion helper as `Sequential`) classifies the first ctor argument as its own layer-constructor call and returns *that* `LayerKind` directly — the wrapper itself carries no shape information beyond its inner layer, so there's nothing to store. An unclassifiable inner layer leaves the whole assignment unclassified. |

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
| `x.chunk(...)` | ✅ | ✅ | ✅ | `KnownFunction::Chunk`; method form of `torch.chunk` — see `compute_chunk_shapes`. |
| `x.unbind(...)` | ✅ | ✅ | ✅ | `KnownFunction::Unbind`; method form of `torch.unbind` — see `compute_unbind_shape`. |
| `x.split(...)` | ✅ | ✅ | ✅ | Method form of real `torch.split` — `KnownFunction::TorchSplit`/`compute_torch_split_shapes` (receiver prepended as positional[0]). Fixed: previously reused `KnownFunction::Split`'s "N equal sections" (jnp) approximation; now uses the real chunk-*size* semantics (see the `torch.split` row above). |
| `x.gather(...)` | ✅ | ✅ | ✅ | `KnownFunction::Gather`; output shape matches `index`. |
| `x.scatter(...)` | ✅ | ✅ | ✅ | `KnownFunction::Scatter`; shape-preserving. |
| `x.masked_select(...)` | ✅ | ✅ | ✅ | `KnownFunction::MaskedSelect`; classified, conservatively `Ok(None)` — length is data-dependent. |
| `x.masked_fill(...)` | ✅ | ✅ | ✅ | `KnownFunction::MaskedFill`; shape-preserving. |
| `x.index_select(...)` | ✅ | ✅ | ✅ | `KnownFunction::IndexSelect`; dim length becomes `len(index)`. |
| `x.narrow(...)` | ✅ | ✅ | ✅ | `KnownFunction::Narrow`; slices one dim to `length`. |
| `x.select(...)` | ✅ | ✅ | ✅ | `KnownFunction::SelectDim`; removes one dim. |
| `x.topk(...)` | ✅ | ✅ | ✅ | `KnownFunction::TopK`; target dim becomes `k`; tuple LHS via `apply_known_topk_shape`. |
| `x.unfold(...)` | ✅ | ✅ | ✅ | `KnownFunction::Unfold`; adds a new trailing window dim of length `size`; concrete dims only. |
| `x.view_as(...)` / `x.reshape_as(...)` / `x.expand_as(...)` | ✅ | ✅ | ✅ | `KnownFunction::ShapeAs`; shape taken from the other tensor argument. |
| `x.flip(...)` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Flip`'s shape-preserving rule. |
| `x.roll(...)` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Roll`'s shape-preserving rule. |
| `x.T` / `x.mT` (properties) | ✅ | ✅ | ✅ | Handled in `shape_of_attribute` (not a call): `x.T` reverses every dim (rank >= 1); `x.mT` swaps the last two dims (rank >= 2). |
| `x.item()` | ✅ | ✅ | ✅ | `KnownFunction::Item`; always scalar (rank-0) output, independent of the receiver's shape. |
| `x.new_zeros(...)` / `new_ones(...)` / `new_full(...)` / `new_empty(...)` | ✅ | ✅ | ✅ | `KnownFunction::NewConstructor`; shape from the `size` arg (receiver is only a dtype/device template), like the `torch.zeros` family. |
| `x.clone(...)` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Copy`'s shape-preserving rule. |
| `x.cpu(...)` / `x.cuda(...)` | ✅ | ✅ | ✅ | Reuse `KnownFunction::Copy`'s shape-preserving rule. |
| `x.float(...)` / `long(...)` / `int(...)` / `bool(...)` / `double(...)` / `half(...)` | ✅ | ✅ | ✅ | Reuse `KnownFunction::Copy`'s shape-preserving rule. |
| `x.clamp(...)` / `x.clip(...)` | ✅ | ✅ | ✅ | Reuse `KnownFunction::Copy`'s shape-preserving rule. |
| `x.softmax(...)` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Copy`'s shape-preserving rule. |
| `x.norm(...)` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Sum`'s reduction rule (same convention as `jnp.linalg.norm`/`torch.linalg.norm`); no `dim` → scalar. |
| `x.diagonal(...)` | ✅ | ✅ | ✅ | Reuses `KnownFunction::Diagonal`'s rule (method form of `torch.diagonal`). |
| `x.tril(...)` / `x.triu(...)` | ✅ | ✅ | ✅ | Reuse `KnownFunction::Tril`/`KnownFunction::Triu`'s shape-preserving rules. |

## Annotation inputs

Annotation parsing is outside the built-in-function registry, but it seeds every rule in this tracker.

- `jaxtyping` forms such as `Float[Array, "batch features"]` support whitespace-separated dimensions.
- NumPy-style forms such as `NDArray[Shape["batch, features"], Float]` support qualified names and comma-separated dimensions.
- Parameter and return annotations support symbolic names and concrete integer dimensions.
- Same-file and imported helper calls bind annotated parameter dimensions into annotated return shapes.
- Exact annotation-token ranges drive binary mismatch diagnostics and same-scope definition lookup for simple symbolic dimensions.
- A matmul quick fix appends `.T` only when the analyzer proves that the rank-2 transpose removes the mismatch.
- Flat symbolic sums and products compare canonically. Full algebra, nested substitution, generics, and `TypeVar` mapping remain open.

## Lazy call-site parameter seeding

Closes the architectural limit documented in `llm.txt`'s "Known
architectural limit (the next big thing)": previously, a callee whose
parameters had no shape annotations could never be shaped, even when
every call site passed fully-shaped arguments — `lax.scan` body functions
destructuring their carry, nested closures, and helper functions relying on
inference all went dark.

- **Mechanism** (`analysis.rs`): when a call resolves to a same-file
  function scope (top-level function, nested closure, or `self.<method>`)
  that isn't FULLY annotated (checked against `original_shapes`, an
  immutable snapshot of every scope's annotations taken before any
  assignment/specialization runs — never the live, mutable `scopes[...].
  shapes`), `apply_seeded_user_function` seeds its un-annotated params from
  the call site's resolved argument shapes (`seed_params_from_call`, using
  the new `FunctionShapeScope::all_params` field — every declared param,
  unlike `param_order`, which is annotated-only) and evaluates its body on
  demand (`specialize_callee_call`): re-collects and replays just that
  function's own assignments, computes the return shape via the extended
  return tracer (`trace_specialized_return` — bare identifier, general
  `return <expr>` via `shape_of_expression`, or a bare-tuple/`expression_
  list` return's elements kept distinct), and reports whichever the caller
  needs. A fully-annotated callee is unaffected — it still goes through the
  older `apply_user_function`/`bind_and_substitute`/`trace_user_function_
  return` path unchanged.
- **Specialization vs. global state**: every specialization runs against a
  scope-local shapes map seeded fresh from `original_shapes` (so one call
  site's argument shapes can never leak into another's). First-call-wins:
  the FIRST successful specialization of a given scope writes its computed
  locals + `assignment_shapes` + `errors` back into the shared analysis
  (editor hover/inlay inside the callee body); every later specialization
  of the same scope reverts its own mutations afterward.
- **Recursion/cycles**: an active-specialization stack in `ShapeCtx` makes
  re-entering a scope already being specialized return `None` immediately;
  depth capped at 8 (`MAX_SPECIALIZATION_DEPTH`).
- **Scalar-typed params**: a plain `int`/`float`/`bool`/`complex`-typed
  parameter (e.g. `decay: float = 0.9`) is seeded as rank-0 (`[]`) at
  extraction time (`extract_jaxtyping_shapes`) — sound (never an array by
  Python's own type system), and needed for binops like `mean + (1 -
  decay) * delta` to shape instead of going dark. `apply_elementwise_shape`
  and `binop_operand` broadcast a resolved rank-0 shape like a literal
  scalar.
- **`jax.lax.scan` body seeding** (the flagship consumer, `scan_body_ys_
  shape`): for `carry_out, ys = lax.scan(body, init, xs)` with a same-file
  `body`, seeds `body`'s first param from `init`'s shape and its second
  from `xs`'s shape minus its leading axis, then traces `ys` from the
  body's second return element with `xs`'s leading dim prepended back.
- **Corpus impact**: closed all 9 dark spots from the previous wave —
  `lax_scan_ema.py`, `eqx_lstm_gru_scan.py`, and `scope_soup.py` all reach
  100%; overall corpus is 176/176 (100%), floor raised from 85 to 90.
  `scan_tuple_carry_corpus_mirror_tests` and the wave-6 regression tests in
  `integration_tests.rs` were flipped from asserting darkness to asserting
  the new shaped values. New coverage: `lazy_call_site_parameter_seeding_
  tests` (multi-call-site specialization, recursion guard, closure param
  seeding, `return <expr>` tracing).
- **Deviation from the original design note**: cross-*file* imported
  helpers are still not covered by this mechanism (same as before) — only
  same-file scopes, since `all_params`/`original_shapes`/`scope_function_
  nodes` are all keyed by the in-file `scopes` array.

## Open targets (current)

The catalog pass is complete. The remaining ❌ rows need analysis beyond a single array-shape rule:

- Higher-order wrappers (`jit`, the `grad` family, and `pmap`) need wrapped-function and transform semantics.
- `lax.cond` and `lax.switch` need branch-output analysis and shape unification.
- Device collectives need mesh and named-axis state.
- `broadcast_shapes` returns a shape tuple, not an array.
- `einops.unpack` needs tracked literal values for the `ps` object, while `equinox.nn.Lambda` wraps an arbitrary callable.

Imported helpers with parameter and return annotations now propagate through `resolve_imported_function_shape` and its signature cache. Lazy body seeding remains same-file only, so imported helpers without return annotations stay unknown.

Run `cargo test corpus_coverage_report -- --nocapture` before a new rule. The harness reports **100% coverage (176/176) across 24 corpus files**, so add representative corpus code before work on a low-impact ❌ row.

Current work, in impact order:

1. **Grow the corpus** with Flax-heavy models, `jax.lax.scan` pipelines, and Torch training steps.
2. **Add richer tracing when the corpus needs it** for branch merges, user-helper tuple returns, or tracked non-array values.
3. **Improve expression precision** for symbolic subtraction/division, parenthesized dimensions, and advanced or boolean indexing.
