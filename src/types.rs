use std::{collections::HashMap, path::PathBuf};

use tree_sitter::Range;

/// Name → shape resolution for the apply helpers. Implemented by plain
/// per-scope `HashMap`s and by `ScopeShapes` in `analysis`, which also sees
/// synthetic bindings without cloning the scope map (#43).
pub trait ShapeLookup {
    fn shape(&self, name: &str) -> Option<&Vec<String>>;
}

impl ShapeLookup for HashMap<String, Vec<String>> {
    fn shape(&self, name: &str) -> Option<&Vec<String>> {
        self.get(name)
    }
}

/// Canonicalize a symbolic-dim expression for equality comparison.
///
/// Dims are bare strings (`"d_state*2"`, `"2*d_state"`, `"batch"`, `"3"`, …)
/// compared across independent call sites (constructor args, jaxtyping
/// annotations, other assignments). Two spellings of the same product/sum
/// commute, but plain `==` treats them as different, producing false
/// mismatches. This is **not** a CAS: it only understands a flat sum of
/// `+`-separated terms, each a flat product of `*`-separated integer
/// literals and identifiers. It:
///
/// * strips whitespace;
/// * folds each term's literal factors into one coefficient (`2*3*d` →
///   `6*d`, `d*1` → `d`);
/// * sorts each term's identifier factors, and sorts the terms themselves,
///   so commutative reorderings compare equal (`d*2` ≡ `2*d`, `a+b*2` ≡
///   `2*b+a`);
/// * drops additive-identity `0` terms (`d+0` → `d`), unless every term is
///   zero.
///
/// Anything containing parens, subtraction, or division is returned
/// unchanged (whitespace-stripped only) — those aren't provably commutative
/// under this simple a token shuffle, so we don't risk a false "equal".
pub fn canonicalize_dim(dim: &str) -> String {
    let stripped: String = dim.chars().filter(|c| !c.is_whitespace()).collect();
    if stripped.is_empty() || stripped.contains(['(', ')', '-', '/']) {
        return stripped;
    }

    let mut terms: Vec<String> = stripped.split('+').map(canonicalize_product_term).collect();
    terms.retain(|t| t != "0");
    if terms.is_empty() {
        terms.push("0".to_string());
    }
    terms.sort_unstable();
    terms.join("+")
}

/// Canonicalize one `+`-separated term: a flat product of int literals and
/// identifiers (`"d_state*2"` → `"2*d_state"`, `"d*1"` → `"d"`).
fn canonicalize_product_term(term: &str) -> String {
    if term.is_empty() {
        return term.to_string();
    }
    let mut literal: i64 = 1;
    let mut idents: Vec<&str> = Vec::new();
    for factor in term.split('*') {
        if let Ok(n) = factor.parse::<i64>() {
            literal = literal.saturating_mul(n);
        } else {
            idents.push(factor);
        }
    }
    idents.sort_unstable();
    if idents.is_empty() {
        return literal.to_string();
    }
    if literal == 1 {
        idents.join("*")
    } else {
        let mut parts = vec![literal.to_string()];
        parts.extend(idents.iter().map(|s| s.to_string()));
        parts.join("*")
    }
}

/// Dim equality under [`canonicalize_dim`] — the shared comparison boundary
/// for every mismatch/broadcast check across `known_functions.rs` and
/// `analysis.rs` (and, via `layers.rs`'s own `canonical_dim`, layer ctor-arg
/// validation).
pub fn dims_canonically_equal(a: &str, b: &str) -> bool {
    canonicalize_dim(a) == canonicalize_dim(b)
}

#[cfg(test)]
mod canonicalize_dim_tests {
    use super::*;

    #[test]
    fn test_commutative_product_matches() {
        assert_eq!(canonicalize_dim("d*2"), canonicalize_dim("2*d"));
        assert_eq!(
            canonicalize_dim("d_state*2"),
            canonicalize_dim("2*d_state")
        );
    }

    #[test]
    fn test_literal_folding() {
        assert_eq!(canonicalize_dim("2*3*d"), "6*d");
        assert_eq!(canonicalize_dim("d*1"), "d");
        assert_eq!(canonicalize_dim("d+0"), "d");
    }

    #[test]
    fn test_commutative_sum_matches() {
        assert_eq!(canonicalize_dim("a+b*2"), canonicalize_dim("2*b+a"));
    }

    #[test]
    fn test_all_zero_sum_collapses_to_zero() {
        assert_eq!(canonicalize_dim("0+0"), "0");
    }

    #[test]
    fn test_plain_identifier_and_literal_unchanged() {
        assert_eq!(canonicalize_dim("batch"), "batch");
        assert_eq!(canonicalize_dim("3"), "3");
    }

    #[test]
    fn test_whitespace_stripped() {
        assert_eq!(canonicalize_dim(" d * 2 "), "2*d");
    }

    #[test]
    fn test_non_commutative_forms_returned_unchanged() {
        // Parens, subtraction, division: not a CAS, returned as-is
        // (whitespace-stripped) rather than risking a false "equal".
        assert_eq!(canonicalize_dim("(d)"), "(d)");
        assert_eq!(canonicalize_dim("d-1"), "d-1");
        assert_eq!(canonicalize_dim("d/2"), "d/2");
    }

    #[test]
    fn test_dims_canonically_equal() {
        assert!(dims_canonically_equal("d*2", "2*d"));
        assert!(!dims_canonically_equal("d*2", "d*3"));
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ImportPath {
    pub dots: usize,
    pub module: Vec<String>,
    pub name: String,
}

#[derive(Debug, PartialEq, Clone)]
pub struct CallInfo {
    pub variable: String,
    pub target: String,
    pub args_node_range: Range,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MethodCallInfo {
    pub variable: String,
    pub receiver: String,
    pub method: String,
    pub args_node_range: Range,
}

#[derive(Debug, PartialEq, Clone)]
pub struct BinaryOpInfo {
    pub variable: String,
    pub left: String,
    pub right: String,
    pub op: BinaryOp,
    pub range: tree_sitter::Range,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BinaryOp {
    MatMul,
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ResolvedTarget {
    pub dots: usize,
    pub parts: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ResolvedModuleTarget {
    pub dots: usize,
    pub module_parts: Vec<String>,
    pub file_path: PathBuf,
    pub symbol_parts: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum PythonSymbol {
    Class { name: String },
    Function { name: String },
    Import { name: String, path: ImportPath },
}

#[derive(Debug, PartialEq, Clone)]
pub struct ResolvedImplementation {
    pub target: ResolvedModuleTarget,
    pub symbol: Option<PythonSymbol>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct PythonCallableSignature {
    pub owner: Option<String>,
    pub name: String,
    pub params: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum CallArgument {
    Positional { value: String },
    Keyword { name: String, value: String },
}

#[derive(Debug, PartialEq, Clone)]
pub struct ResolvedCallSignature {
    pub implementation: ResolvedImplementation,
    pub signature: PythonCallableSignature,
    pub arguments: Vec<CallArgument>,
    pub bindings: HashMap<String, String>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum LayerKind {
    Linear {
        in_features: String,
        out_features: String,
    },
    Conv1d {
        in_channels: String,
        out_channels: String,
        kernel_size: String,
        stride: String,
        padding: String,
    },
    Conv2d {
        in_channels: String,
        out_channels: String,
        kernel_size: String,
        stride: String,
        padding: String,
    },
    Conv3d {
        in_channels: String,
        out_channels: String,
        kernel_size: String,
        stride: String,
        padding: String,
    },
    /// flax.linen.Dense — channels-last: replaces the last dim with
    /// `features`. Input width is runtime-inferred by flax, so there is
    /// nothing to check.
    Dense {
        features: String,
    },
    /// flax.linen.Conv with default stride / SAME padding: channels-last,
    /// spatial dims unchanged, last dim becomes `features`. Non-default
    /// strides are refused at classification time.
    FlaxConv {
        features: String,
        spatial_rank: usize,
    },
    /// Max/Avg pooling, channels-first like Conv: channels preserved, the
    /// trailing `spatial_rank` dims follow the conv output formula.
    Pool {
        name: String,
        spatial_rank: usize,
        kernel_size: String,
        stride: String,
        padding: String,
    },
    /// Adaptive pooling: the trailing `spatial_rank` dims all become
    /// `output_size`, channels preserved.
    AdaptivePool {
        name: String,
        spatial_rank: usize,
        output_size: String,
    },
    /// torch.nn.MultiheadAttention — returns an (output, weights) tuple:
    /// output has the query's shape, weights are (..., L, S) with L/S the
    /// query/key sequence lengths. Only tuple-unpacking LHS is modelled.
    ///
    /// `feature_dim` is the ctor's per-token model dimension: `embed_dim`
    /// for `torch.nn.MultiheadAttention(embed_dim, num_heads)`, or
    /// `query_size` for `equinox.nn.MultiheadAttention(num_heads,
    /// query_size, ...)` — equinox's real positional order is reversed and
    /// uses a different name, so `classify_layer_call` must branch on the
    /// resolved module path rather than reusing one binding key for both
    /// frameworks (previously it always read torch's `embed_dim` key, so
    /// equinox ctors were silently unclassified/misbound).
    MultiheadAttention {
        feature_dim: String,
    },
    /// Index lookup table: appends `embedding_size` to the input shape
    /// (scalar index → `(embedding_size,)`, `(batch, seq)` → `(batch, seq, embedding_size)`).
    Embedding {
        embedding_size: String,
    },
    ShapePreserving {
        name: String,
    },
    /// torch.nn.Flatten — collapses dims `[start_dim, end_dim]` (Python
    /// negative indices resolved against the input rank at apply time) into
    /// a single dim: product of literal ints when all concrete, otherwise a
    /// symbolic `d0*d1*...` string.
    Flatten {
        start_dim: String,
        end_dim: String,
    },
    /// torch.nn.Unflatten(dim, sizes) — expands `dim` into the components of
    /// `sizes` (raw ctor tuple/list text, split at apply time).
    Unflatten {
        dim: String,
        sizes: String,
    },
    /// torch.nn.Upsample — scales the trailing spatial dims (all dims after
    /// the leading batch+channel dims, determined at apply time from the
    /// input rank since `Upsample` itself is rank-agnostic) by
    /// `scale_factor`, or sets them directly to `size`. Exactly one of the
    /// two is expected to be usable; anything else is treated as unknown
    /// (`Ok(None)`).
    Upsample {
        scale_factor: Option<String>,
        size: Option<String>,
    },
    /// ConvTranspose1d/2d/3d (torch + equinox): inverse of the conv output
    /// formula: `out = (in-1)*stride - 2*padding + kernel`. Channels-first
    /// layout, same convention as `Conv1d`/`Conv2d`/`Conv3d`. `output_padding`
    /// and `dilation` are not modelled (assumed 0 / 1).
    ConvTranspose {
        spatial_rank: usize,
        in_channels: String,
        out_channels: String,
        kernel_size: String,
        stride: String,
        padding: String,
    },
    /// torch.nn.RNN/LSTM/GRU full-sequence modules. Models only the primary
    /// output tensor (last dim replaced by `hidden_size`, all other dims
    /// preserved); the true return value is an `(output, final_state)`
    /// tuple, which this analyzer can't express without tuple-unpacking
    /// support in the layer-application pipeline (that lives in
    /// `analysis.rs`, outside this module's scope), so direct/non-tuple
    /// application is an approximation. `batch_first` doesn't change the
    /// formula: the feature dim is always trailing regardless of (seq,
    /// batch) ordering, so it isn't tracked. `bidirectional`/`num_layers`
    /// are not modelled (hidden_size is used as-is).
    Rnn {
        name: String,
        input_size: String,
        hidden_size: String,
    },
    /// torch.nn.RNNCell/GRUCell/LSTMCell and equinox.nn.LSTMCell/GRUCell —
    /// single-step cells. RNNCell/GRUCell genuinely return one tensor of
    /// shape `hidden_size`; LSTMCell returns `(h, c)` — modelled as just the
    /// `h` component (same shape as GRUCell's output), an approximation for
    /// the same tuple-output reason as `Rnn`.
    RnnCell {
        name: String,
        input_size: String,
        hidden_size: String,
    },
    /// torch.nn.PixelShuffle(upscale_factor): `(*, C*r^2, H, W) -> (*, C, H*r, W*r)`.
    PixelShuffle {
        upscale_factor: String,
    },
    /// torch.nn.PixelUnshuffle(downscale_factor): `(*, C, H*r, W*r) -> (*, C*r^2, H, W)`.
    PixelUnshuffle {
        downscale_factor: String,
    },
    /// Constant/Zero/Reflection/Replication Pad Nd: pads the trailing
    /// `spatial_rank` dims. `padding` is the raw ctor arg text: a single int
    /// means uniform padding on every side of every spatial dim; an explicit
    /// `2*spatial_rank`-length tuple follows torch's reverse-axis pad-pair
    /// convention (last spatial dim's (left,right) first).
    Pad {
        name: String,
        spatial_rank: usize,
        padding: String,
    },
    /// torch.nn.Bilinear(in1_features, in2_features, out_features): this
    /// analyzer only tracks the first positional input per layer call (see
    /// `extract_layer_applications`), so this only validates/transforms
    /// against `x1`; `x2` and its broadcasting are not modelled.
    Bilinear {
        in1_features: String,
        in2_features: String,
        out_features: String,
    },
    /// torch.nn.CosineSimilarity(dim): removes the reduced axis from `x1`'s
    /// shape (same single-tracked-input limitation as `Bilinear`).
    CosineSimilarity {
        dim: String,
    },
    /// equinox.nn.MLP: last-dim transform `in_size -> out_size`, same rule as
    /// `Linear`.
    Mlp {
        in_size: String,
        out_size: String,
    },
    /// `torch.nn.Sequential` / `equinox.nn.Sequential`: a composite container
    /// wrapping an ordered list of child layers (torch: variadic positional
    /// ctor args; equinox: a single list-literal positional arg — both forms
    /// are unwrapped into `children` at classification time). Application
    /// threads the input shape through each child's own apply rule in order,
    /// via the shared `apply_layer_kind` helper; a mid-chain error reports as
    /// that child's own error message, and an unknown (`Ok(None)`) child
    /// makes the whole chain unknown. Classification itself is honest: if any
    /// ctor argument isn't itself a resolvable layer-constructor call
    /// (arbitrary callable, `*args` unpacking, etc.), the whole `Sequential`
    /// is left unclassified.
    Sequential { children: Vec<LayerKind> },
    /// `einops.layers.torch.Rearrange` / `Reduce` (and the flax equivalents)
    /// — the layer-object form of the free-function einops pattern algebra.
    /// `name` is `"Rearrange"`/`"Reduce"` (cosmetic only: the reduction op
    /// itself isn't shape-relevant, same as the free-function form).
    /// `pattern` is the raw (still-quoted) ctor pattern string; `kwargs` are
    /// axis-length keyword bindings from the ctor (e.g. `h=14`). Both are fed
    /// verbatim into `known_functions::apply_known_einops`'s existing pattern
    /// engine at apply time.
    EinopsPattern {
        name: String,
        pattern: String,
        kwargs: HashMap<String, String>,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub enum KnownFunction {
    Concatenate,
    Stack,
    Reshape,
    Transpose,
    ExpandDims,
    Squeeze,
    Sum,
    Mean,
    Max,
    Min,
    Prod,
    Std,
    Var,
    Matmul,
    Dot,
    Einsum,
    Split,
    /// Real torch `split`/`Tensor.split` semantics: the 2nd arg is a
    /// `split_size` (or list of sizes), not a section *count* like
    /// `jnp.split`/`np.split`/`torch.tensor_split` (see [`KnownFunction::Split`]).
    TorchSplit,
    Tile,
    Repeat,
    Flatten,
    Ravel,
    MoveAxis,
    SwapAxes,
    Where,
    Zeros,
    Ones,
    Full,
    Arange,
    Eye,
    Vmap,
    Scan,
    FlaxPool,
    EinopsRearrange,
    EinopsReduce,
    EinopsRepeat,
    BroadcastTo,
    BroadcastArrays,
    AtLeast1D,
    AtLeast2D,
    AtLeast3D,
    Pad,
    Roll,
    Flip,
    Rot90,
    Take,
    Diag,
    Diagonal,
    Trace,
    Triu,
    Tril,
    Meshgrid,
    Vstack,
    Hstack,
    Dstack,
    ColumnStack,
    Block,
    Permute,
    Array,
    AsArray,
    Empty,
    ZerosLike,
    OnesLike,
    FullLike,
    EmptyLike,
    Identity,
    Linspace,
    Logspace,
    All,
    Any,
    ArgMax,
    ArgMin,
    Argsort,
    Sort,
    Cumsum,
    Cumprod,
    TensorDot,
    Outer,
    Inner,
    Vdot,
    LinalgInv,
    LinalgDet,
    LinalgSvd,
    LinalgQr,
    LinalgEig,
    Astype,
    Copy,
    Detach,
    Contiguous,
    To,
    // jax.lax higher-order / structured ops
    LaxMap,
    LaxCond,
    LaxSwitch,
    LaxWhileLoop,
    LaxForiLoop,
    LaxConvGeneralDilated,
    LaxGather,
    LaxScatter,
    LaxReduceWindow,
    LaxTopK,
    LaxSort,
    LaxSortKeyVal,
    LaxPad,
    LaxBroadcast,
    LaxBroadcastInDim,
    LaxSlice,
    LaxDynamicSlice,
    LaxDynamicUpdateSlice,
    LaxAssociativeScan,
    // jax.numpy / numpy array creation
    Diagflat,
    Tri,
    Indices,
    BinCount,
    Unique,
    Select,
    // jax.numpy / numpy shape transforms
    RollAxis,
    Resize,
    Insert,
    Delete,
    Append,
    // jax.numpy / numpy joining and splitting
    HSplit,
    VSplit,
    DSplit,
    Kron,
    // jax.numpy / numpy indexing and selection
    TakeAlongAxis,
    PutAlongAxis,
    Nonzero,
    Argwhere,
    SearchSorted,
    Extract,
    Compress,
    Histogram,
    // linear algebra
    Cross,
    LinalgSolve,
    LinalgLstsq,
    LinalgPinv,
    LinalgMatrixRank,
    // einops
    EinopsEinsum,
    EinopsPack,
    EinopsUnpack,
    EinopsParseShape,
    // jax.nn
    OneHot,
    DotProductAttention,
    // torch tensor indexing / selection methods
    Gather,
    Scatter,
    IndexSelect,
    Narrow,
    SelectDim,
    MaskedSelect,
    MaskedFill,
    Unfold,
    ShapeAs,
    Item,
    NewConstructor,
    // torch tuple-output methods
    TopK,
    Chunk,
    Unbind,
    KthValue,
    MedianDim,
    // torch combinatorics
    Combinations,
    CartesianProd,
    BlockDiag,
    // torch.nn.functional
    Interpolate,
    FunctionalConv1d,
    FunctionalConv2d,
    FunctionalConv3d,
    FunctionalMaxPool1d,
    FunctionalMaxPool2d,
    FunctionalMaxPool3d,
    FunctionalAvgPool1d,
    FunctionalAvgPool2d,
    FunctionalAvgPool3d,
    FunctionalEmbedding,
    // torch.nn.utils.rnn
    PadSequence,
}

#[derive(Debug, Clone)]
pub struct LayerApplication {
    pub variable: String,
    pub layer: String,
    pub input: String,
    pub kind: LayerKind,
    pub range: Range,
}

impl PartialEq for LayerApplication {
    fn eq(&self, other: &Self) -> bool {
        self.variable == other.variable
            && self.layer == other.layer
            && self.input == other.input
            && self.kind == other.kind
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct LayerAssignment {
    pub name: String,
    pub kind: LayerKind,
    pub byte_position: usize,
}

/// A `self.<attr> = <layer ctor>` binding together with the byte range of the
/// class it was defined in, so same-named attrs in different classes don't
/// collide at lookup time.
#[derive(Debug, PartialEq, Clone)]
pub struct ScopedSelfAttrLayer {
    pub class_start: usize,
    pub class_end: usize,
    pub kind: LayerKind,
}

/// A `self.<attr> = <ident>` alias binding together with the byte range of
/// the class it was defined in, so same-named attrs in different classes
/// don't collide at lookup time. Mirrors `ScopedSelfAttrLayer`.
#[derive(Debug, PartialEq, Clone)]
pub struct ScopedSelfAttrAlias {
    pub class_start: usize,
    pub class_end: usize,
    pub value: String,
}

/// Classifies a `ShapeError` by how confident the rule that raised it is.
///
/// `Mismatch`: a genuine shape contradiction under the rule's modeled
/// semantics. `Approximation`: raised by a rule that is documented (see
/// `llm.txt`/`TO_IMPLEMENT.md`) as an approximation of the real op, so the
/// "error" may be a false positive rather than an actual bug in the analyzed
/// code. Drives diagnostic severity in `main.rs`.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ShapeErrorKind {
    Mismatch,
    Approximation,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ShapeError {
    pub variable: String,
    pub message: String,
    pub range: Range,
    pub kind: ShapeErrorKind,
    /// Optional related-information location: a second range (e.g. the
    /// *other* operand's node in a binary-op mismatch) plus a short message,
    /// surfaced as LSP `DiagnosticRelatedInformation` by `main.rs`. `None`
    /// for errors that don't have a natural second location.
    pub related: Option<(Range, String)>,
}

impl ShapeError {
    /// A genuine shape contradiction under the rule's modeled semantics.
    pub fn mismatch(variable: impl Into<String>, message: impl Into<String>, range: Range) -> Self {
        ShapeError {
            variable: variable.into(),
            message: message.into(),
            range,
            kind: ShapeErrorKind::Mismatch,
            related: None,
        }
    }

    /// Raised by a rule that is a documented approximation of the real op
    /// (e.g. `jax.lax.dot_general` modeled as matmul), so the error may be a
    /// false positive rather than an actual bug in the analyzed code.
    pub fn approximation(
        variable: impl Into<String>,
        message: impl Into<String>,
        range: Range,
    ) -> Self {
        ShapeError {
            variable: variable.into(),
            message: message.into(),
            range,
            kind: ShapeErrorKind::Approximation,
            related: None,
        }
    }

    /// Attach a related-information location/message, e.g. the *other*
    /// operand's node range in a two-operand shape mismatch. Builder-style:
    /// chain after `mismatch`/`approximation`.
    pub fn with_related(mut self, range: Range, message: impl Into<String>) -> Self {
        self.related = Some((range, message.into()));
        self
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct FunctionShapeScope {
    pub function_name: Option<String>,
    pub start_byte: usize,
    pub end_byte: usize,
    pub shapes: HashMap<String, Vec<String>>,
    pub return_shape: Option<Vec<String>>,
    /// Parameter names with jaxtyping annotations, in declaration order.
    /// Used by cross-function shape propagation to match positional call
    /// arguments to declared parameter shapes.
    pub param_order: Vec<String>,
}

/// One record per (non-annotated) assignment that produced a shape, in the
/// order assignments were processed. Unlike `FunctionShapeScope::shapes`
/// (which keeps only the last shape per name), this preserves the shape at
/// *each* assignment site so inlay hints can be emitted per reassignment.
#[derive(Debug, PartialEq, Clone)]
pub struct AssignmentShape {
    /// 0-based row of the assignment's LHS.
    pub line: u32,
    pub name: String,
    pub shape: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct LayerShapeAnalysis {
    pub scopes: Vec<FunctionShapeScope>,
    pub layers: HashMap<String, LayerKind>,
    pub applications: Vec<LayerApplication>,
    pub errors: Vec<ShapeError>,
    pub assignment_shapes: Vec<AssignmentShape>,
}

impl LayerShapeAnalysis {
    pub fn scope_for_byte(&self, byte: usize) -> Option<&FunctionShapeScope> {
        scope_for_byte(&self.scopes, byte)
    }
}

pub fn scope_for_byte(scopes: &[FunctionShapeScope], byte: usize) -> Option<&FunctionShapeScope> {
    scope_index_for_byte(scopes, byte).map(|i| &scopes[i])
}

pub fn scope_index_for_byte(scopes: &[FunctionShapeScope], byte: usize) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for (i, scope) in scopes.iter().enumerate() {
        if scope.start_byte <= byte && byte < scope.end_byte {
            let size = scope.end_byte - scope.start_byte;
            match best {
                None => best = Some((i, size)),
                Some((_, prev_size)) if size <= prev_size => best = Some((i, size)),
                _ => {}
            }
        }
    }
    best.map(|(i, _)| i)
}
