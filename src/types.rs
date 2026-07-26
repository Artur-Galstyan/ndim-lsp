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

#[derive(Debug, PartialEq, Clone)]
pub struct ShapeError {
    pub variable: String,
    pub message: String,
    pub range: Range,
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
