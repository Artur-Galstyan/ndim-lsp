use std::{collections::HashMap, path::PathBuf};

use tree_sitter::Range;

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
}

#[derive(Debug, PartialEq, Clone)]
pub struct LayerShapeAnalysis {
    pub scopes: Vec<FunctionShapeScope>,
    pub layers: HashMap<String, LayerKind>,
    pub applications: Vec<LayerApplication>,
    pub errors: Vec<ShapeError>,
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
