# JAX release compatibility

This audit targets **JAX 0.11.1**, the latest stable release reported by
[PyPI](https://pypi.org/project/jax/0.11.1/) when this change was made.
The project had no previous pinned JAX compatibility baseline. We compared
public exports with **0.9.2**, the last 0.9 release, to identify additions in
the 0.10 and 0.11 series. This is not an exhaustive audit of older JAX APIs.

## Sources and scope

- [Versioned release notes](https://github.com/jax-ml/jax/blob/jax-v0.11.1/CHANGELOG.md).
  The tag still labels its newest section "Unreleased". The **released wheel**,
  not that heading, determines which APIs this audit includes.
- Explicit public exports in the 0.9.2 and 0.11.1 PyPI wheels.
- Signatures and implementations in the installed 0.11.1 wheel.
- Real CPU calls against `jax==jaxlib==0.11.1`, checked with `check.py`.

The export comparison covers `jax`, `jax.numpy`, `jax.lax`, `jax.nn`,
`jax.scipy.linalg`, `jax.scipy.special`, `jax.image`, and `jax.random`.
It does not cover every SciPy submodule, experimental APIs, methods, or
all signature changes. For example, the manual signature check found the new
`axis` parameter of `jax.lax.top_k`, which an export-name comparison cannot find.
The release-note review also found the new array `byteswap()` method. It now
shares the existing shape-preserving method rule and has a runtime fixture.

## New numerical APIs

All **26 new array-returning functions** in those namespace export differences
now have shape rules. Common shape mechanics share rules rather than copy them.

| Namespace | Added functions | Shape rule |
| --- | --- | --- |
| `jax.numpy` | `top_k` | Two outputs. Replace the selected axis with `k`. |
| `jax.lax` | `broadcast_like` | Broadcast the input to the reference shape, not the reverse. |
| `jax.lax` | `mulhi`, `stage` | Elementwise. `mulhi` permits scalar broadcast but rejects non-scalar rank promotion. |
| `jax.lax` | `stack`, `unstack` | Add an axis for a literal sequence; remove an axis for tuple unpacking. |
| `jax.scipy.linalg` | `hadamard`, `dft`, `invhilbert`, `invpascal` | `(n, n)`. Validate the Hadamard size. Do not mistake dtype, scale, or kind for a column count. |
| `jax.scipy.linalg` | `helmert` | `(n-1, n)` by default, `(n, n)` with `full=True`. |
| `jax.scipy.linalg` | `circulant`, `fiedler` | `(..., n)` becomes `(..., n, n)`. Promote scalar inputs to length one. |
| `jax.scipy.linalg` | `companion`, `fiedler_companion` | `(..., n)` becomes `(..., n-1, n-1)`. Their minimum input lengths differ. |
| `jax.scipy.linalg` | `leslie` | Broadcast batch axes. Require survival length to equal fecundity length minus one. |
| `jax.scipy.linalg` | `convolution_matrix` | `(..., m)` becomes `(..., rows, n)`. `full`: `m+n-1`; `same`: `max(m,n)`; `valid`: `max(m,n)-min(m,n)+1`. |
| `jax.scipy.linalg` | `qr_multiply` | Separate product, R, and optional pivot shapes. Support both modes, vectors, and broadcast batches. JAX ravels vector results, including batch axes. |
| `jax.scipy.special` | `boxcox`, `boxcox1p`, `comb`, `owens_t` | Broadcast both array arguments, including keyword arguments and scalar literals. |
| `jax.scipy.special` | `dawsn`, `erfcx`, `loggamma`, `wofz` | Preserve the input shape. |

`jax.nn` and `jax.image` had no new explicit exports in this comparison.
The AREA resize method changes values, not the output shape.

Other additions do not need direct tensor-shape rules:

- Compiler/configuration and mesh APIs: `CompilerEffortLevel`, `Inline`,
  `auto_pcast`, `get_abstract_mesh`, `get_mesh`, `use_abstract_mesh`.
- Dtypes: `float6_e2m3fn`, `float6_e3m2fn`, `random.key_dtype`.
- Primitive objects: `mulhi_p`, `stack_p`, `unstack_p`, `stage_p`.
- The `lib` module export and the higher-order `custom_remat` wrapper.
  Callable-wrapper inference remains outside this audit.
- `ShapeDtypeStruct.like` constructs abstract metadata, not a tensor.

Removed exports are configuration/internal APIs, not existing shape-rule targets:
`custom_transpose`, `enable_custom_vjp_by_custom_transpose`, `memory`,
`all_gather_done`, `all_gather_start`, `dce_sink_p`.

## Conservative limits

These rules describe shapes, not dtype validity or numerical domains.
They do not validate every Python call signature or every option unrelated to
shape. They do not select behavior based on the JAX version installed in a
user's Python environment.

Unknown axes, unknown shape-changing modes, and unknown `full`/`pivoting`
flags produce unknown shapes rather than default-mode guesses. Symbolic
dimensions propagate where a formula is available. Convolution `same` and
`valid` currently require concrete sizes. Tuple results need direct unpacking
to expose their component shapes. The tuple itself is not a tensor.

Stack inference accepts literal lists/tuples, including nested array
expressions. A variable that holds a sequence remains unknown because the
analyzer does not track general container types. The shared stack fix also
applies to NumPy and PyTorch: homogeneous tuples must not collapse into one
stack operand.

## Repeat the checks

The standard Rust suite consumes `cases.json` without Python or network access:

```sh
cargo test jax_compat_tests
```

Use an isolated environment for the optional runtime oracle:

```sh
uv run --isolated --no-project --python 3.13 --with 'jax[cpu]==0.11.1' \
  python bench/jax_compat/check.py
```

`check.py` only reads the fixtures. It executes every call with concrete
inputs and checks either the expected shapes or a runtime exception.
The Rust test checks the same fixtures through the full Python AST analyzer.
Separate Rust tests cover symbolic dimensions, unknown inputs, and namespaces.

Repeat the public export comparison without installing either JAX version:

```sh
python3 bench/jax_compat/audit.py --baseline 0.9.2 --candidate 0.11.1
```

For the next release, use 0.11.1 as the baseline, read the release notes,
and inspect changed signatures. Add new cases to the shared fixtures, update
the pinned oracle version, and run both checks. Do not treat the export
comparison alone as a full compatibility test.
