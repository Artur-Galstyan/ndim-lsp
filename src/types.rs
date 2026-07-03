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
    /// Index lookup table: appends `embedding_size` to the input shape
    /// (scalar index → `(embedding_size,)`, `(batch, seq)` → `(batch, seq, embedding_size)`).
    Embedding {
        embedding_size: String,
    },
    ShapePreserving {
        name: String,
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
