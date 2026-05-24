use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use tree_sitter::{Node, Parser, Query, QueryCursor, Range, StreamingIterator};

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
    pub args_node_range: tree_sitter::Range,
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
pub struct LayerShapeAnalysis {
    pub shapes: HashMap<String, Vec<String>>,
    pub layers: HashMap<String, LayerKind>,
    pub applications: Vec<LayerApplication>,
    pub errors: Vec<ShapeError>,
}

pub fn resolve_python_module_on_disk(
    module: &[String],
    search_roots: &[PathBuf],
) -> Option<PathBuf> {
    if module.is_empty() {
        return None;
    }
    for root in search_roots {
        let module_path = root.join(module.join("/"));
        let file_path = module_path.with_extension("py");
        if file_path.exists() {
            return Some(file_path);
        }

        let package_path = module_path.join("__init__.py");
        if package_path.exists() {
            return Some(package_path);
        }
    }
    None
}

pub fn resolve_call_target(
    target: &str,
    import_map: &HashMap<String, ImportPath>,
) -> ResolvedTarget {
    let target_parts = target
        .split(".")
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>();

    let Some((first, rest)) = target_parts.split_first() else {
        return ResolvedTarget {
            dots: 0,
            parts: target_parts.iter().map(|p| p.to_string()).collect(),
        };
    };

    let imported = import_map.get(*first);

    match imported {
        Some(i) => {
            let mut parts = i.module.clone();
            parts.push(i.name.to_string());
            parts.extend(rest.iter().map(|p| p.to_string()));

            ResolvedTarget {
                dots: i.dots,
                parts,
            }
        }
        None => ResolvedTarget {
            dots: 0,
            parts: target_parts.iter().map(|p| p.to_string()).collect(),
        },
    }
}

pub fn resolve_target_on_disk(
    target: &ResolvedTarget,
    search_roots: &[PathBuf],
) -> Option<ResolvedModuleTarget> {
    if target.dots > 0 || target.parts.is_empty() {
        return None;
    }

    for len in (1..=target.parts.len()).rev() {
        let module_parts = &target.parts[..len];
        let symbol_parts = &target.parts[len..];

        if let Some(file_path) = resolve_python_module_on_disk(module_parts, search_roots) {
            return Some(ResolvedModuleTarget {
                dots: target.dots,
                module_parts: module_parts.to_vec(),
                file_path,
                symbol_parts: symbol_parts.to_vec(),
            });
        }
    }

    None
}

pub fn resolve_import_path_from_package(
    current_package_parts: &[String],
    import_path: &ImportPath,
) -> Option<ResolvedTarget> {
    if import_path.dots == 0 {
        let mut parts = import_path.module.clone();
        parts.push(import_path.name.clone());
        return Some(ResolvedTarget { dots: 0, parts });
    }

    let up = import_path.dots - 1;

    if up >= current_package_parts.len() {
        return None;
    }

    let base_len = current_package_parts.len() - up;
    let mut parts = current_package_parts[..base_len].to_vec();

    parts.extend(import_path.module.iter().cloned());
    parts.push(import_path.name.clone());

    Some(ResolvedTarget { dots: 0, parts })
}

pub fn follow_import_symbol_once(
    current_package_parts: &[String],
    symbol: &PythonSymbol,
) -> Option<ResolvedTarget> {
    match symbol {
        PythonSymbol::Import { path, .. } => {
            resolve_import_path_from_package(current_package_parts, path)
        }
        PythonSymbol::Class { .. } | PythonSymbol::Function { .. } => None,
    }
}

pub fn resolve_reexport_once(
    resolved: &ResolvedModuleTarget,
    node: Node,
    text: &str,
) -> Result<Option<ResolvedTarget>, String> {
    let Some(first) = resolved.symbol_parts.first() else {
        return Ok(None);
    };

    let Some(symbol) = find_top_level_symbol(node, text, first)? else {
        return Ok(None);
    };

    let Some(mut next) = follow_import_symbol_once(&resolved.module_parts, &symbol) else {
        return Ok(None);
    };

    next.parts
        .extend(resolved.symbol_parts.iter().skip(1).cloned());
    Ok(Some(next))
}

pub fn resolve_terminal_symbol_once(
    resolved: &ResolvedModuleTarget,
    node: Node,
    text: &str,
) -> Result<Option<PythonSymbol>, String> {
    let Some(first) = resolved.symbol_parts.first() else {
        return Ok(None);
    };

    let Some(symbol) = find_top_level_symbol(node, text, first)? else {
        return Ok(None);
    };

    match symbol {
        PythonSymbol::Class { .. } | PythonSymbol::Function { .. } => Ok(Some(symbol)),
        PythonSymbol::Import { .. } => Ok(None),
    }
}

pub fn resolve_implementation<F>(
    start: ResolvedTarget,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
) -> Result<Option<ResolvedImplementation>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|e| e.to_string())?;

    let mut current = start;

    let mut visited = HashSet::new();

    for _ in 0..max_depth {
        if !visited.insert(current.parts.clone()) {
            return Ok(None);
        }
        let Some(resolved) = resolve_target_on_disk(&current, search_roots) else {
            return Ok(None);
        };

        if resolved.symbol_parts.is_empty() {
            return Ok(Some(ResolvedImplementation {
                target: resolved,
                symbol: None,
            }));
        }

        let Some(text) = read_file(&resolved.file_path) else {
            return Ok(None);
        };

        let Some(tree) = parser.parse(&text, None) else {
            return Err("failed to parse file".to_string());
        };

        let root = tree.root_node();

        if let Some(symbol) = resolve_terminal_symbol_once(&resolved, root, &text)? {
            return Ok(Some(ResolvedImplementation {
                target: resolved,
                symbol: Some(symbol),
            }));
        }

        if let Some(next) = resolve_reexport_once(&resolved, root, &text)? {
            current = next;
            continue;
        }

        return Ok(None);
    }

    Ok(None)
}

pub fn extract_callable_signature(
    node: Node,
    text: &str,
    symbol: &PythonSymbol,
) -> Result<Option<PythonCallableSignature>, String> {
    fn node_text(node: Node, text: &str) -> Result<String, String> {
        node.utf8_text(text.as_bytes())
            .map(|s| s.to_string())
            .map_err(|e| e.to_string())
    }

    fn definition_name(node: Node, text: &str) -> Result<Option<String>, String> {
        node.child_by_field_name("name")
            .map(|name| node_text(name, text))
            .transpose()
    }

    fn parameter_name(node: Node, text: &str) -> Result<Option<String>, String> {
        if node.kind() == "identifier" {
            return Ok(Some(node_text(node, text)?));
        }

        if let Some(name) = node.child_by_field_name("name") {
            return Ok(Some(node_text(name, text)?));
        }

        for i in 0..node.named_child_count() {
            let Some(child) = node.named_child(i as u32) else {
                continue;
            };
            if child.kind() == "identifier" {
                return Ok(Some(node_text(child, text)?));
            }
        }

        Ok(None)
    }

    fn params_from_function(node: Node, text: &str) -> Result<Option<Vec<String>>, String> {
        let Some(params_node) = node.child_by_field_name("parameters") else {
            return Ok(None);
        };

        let mut params = Vec::new();
        for i in 0..params_node.named_child_count() {
            let Some(child) = params_node.named_child(i as u32) else {
                continue;
            };
            if let Some(name) = parameter_name(child, text)? {
                params.push(name);
            }
        }

        Ok(Some(params))
    }

    fn top_level_function_signature(
        node: Node,
        text: &str,
        name: &str,
    ) -> Result<Option<PythonCallableSignature>, String> {
        for i in 0..node.named_child_count() {
            let Some(child) = node.named_child(i as u32) else {
                continue;
            };
            if child.kind() != "function_definition" {
                continue;
            }
            if definition_name(child, text)?.as_deref() != Some(name) {
                continue;
            }
            let Some(params) = params_from_function(child, text)? else {
                return Ok(None);
            };
            return Ok(Some(PythonCallableSignature {
                owner: None,
                name: name.to_string(),
                params,
            }));
        }

        Ok(None)
    }

    fn class_init_signature(
        node: Node,
        text: &str,
        class_name: &str,
    ) -> Result<Option<PythonCallableSignature>, String> {
        for i in 0..node.named_child_count() {
            let Some(class_node) = node.named_child(i as u32) else {
                continue;
            };
            if class_node.kind() != "class_definition" {
                continue;
            }
            if definition_name(class_node, text)?.as_deref() != Some(class_name) {
                continue;
            }

            let Some(body) = class_node.child_by_field_name("body") else {
                return Ok(None);
            };

            for j in 0..body.named_child_count() {
                let Some(method) = body.named_child(j as u32) else {
                    continue;
                };
                if method.kind() != "function_definition" {
                    continue;
                }
                if definition_name(method, text)?.as_deref() != Some("__init__") {
                    continue;
                }
                let Some(params) = params_from_function(method, text)? else {
                    return Ok(None);
                };
                return Ok(Some(PythonCallableSignature {
                    owner: Some(class_name.to_string()),
                    name: "__init__".to_string(),
                    params,
                }));
            }
        }

        Ok(None)
    }

    match symbol {
        PythonSymbol::Class { name } => class_init_signature(node, text, name),
        PythonSymbol::Function { name } => top_level_function_signature(node, text, name),
        PythonSymbol::Import { .. } => Ok(None),
    }
}

pub fn resolve_call_signature<F>(
    call: &CallInfo,
    source_text: &str,
    import_map: &HashMap<String, ImportPath>,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
) -> Result<Option<ResolvedCallSignature>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let target = resolve_call_target(&call.target, import_map);
    let Some(implementation) = resolve_implementation(target, search_roots, &read_file, max_depth)?
    else {
        return Ok(None);
    };
    let Some(symbol) = &implementation.symbol else {
        return Ok(None);
    };

    let Some(implementation_text) = read_file(&implementation.target.file_path) else {
        return Ok(None);
    };

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|e| e.to_string())?;

    let Some(implementation_tree) = parser.parse(&implementation_text, None) else {
        return Err("failed to parse implementation file".to_string());
    };

    let Some(signature) = extract_callable_signature(
        implementation_tree.root_node(),
        &implementation_text,
        symbol,
    )?
    else {
        return Ok(None);
    };

    let Some(source_tree) = parser.parse(source_text, None) else {
        return Err("failed to parse source file".to_string());
    };
    let Some(args_node) = source_tree.root_node().descendant_for_byte_range(
        call.args_node_range.start_byte,
        call.args_node_range.end_byte,
    ) else {
        return Ok(None);
    };

    let arguments = extract_call_arguments(args_node, source_text)?;
    let bindings = bind_call_arguments(&signature, &arguments);

    Ok(Some(ResolvedCallSignature {
        implementation,
        signature,
        arguments,
        bindings,
    }))
}

pub fn extract_call_arguments(args_node: Node, text: &str) -> Result<Vec<CallArgument>, String> {
    let mut args = Vec::new();

    for i in 0..args_node.named_child_count() {
        let Some(child) = args_node.named_child(i as u32) else {
            continue;
        };

        if child.kind() == "keyword_argument" {
            let Some(name_node) = child.child_by_field_name("name") else {
                return Err("keyword_argument missing name".to_string());
            };
            let Some(value_node) = child.child_by_field_name("value") else {
                return Err("keyword_argument missing value".to_string());
            };
            args.push(CallArgument::Keyword {
                name: name_node
                    .utf8_text(text.as_bytes())
                    .map_err(|e| e.to_string())?
                    .to_string(),
                value: value_node
                    .utf8_text(text.as_bytes())
                    .map_err(|e| e.to_string())?
                    .to_string(),
            });
        } else {
            args.push(CallArgument::Positional {
                value: child
                    .utf8_text(text.as_bytes())
                    .map_err(|e| e.to_string())?
                    .to_string(),
            });
        }
    }

    Ok(args)
}

pub fn bind_call_arguments(
    signature: &PythonCallableSignature,
    args: &[CallArgument],
) -> HashMap<String, String> {
    let mut params = signature.params.as_slice();
    if signature.owner.is_some()
        && matches!(params.first(), Some(first) if first == "self" || first == "cls")
    {
        params = &params[1..];
    }

    let mut bindings = HashMap::new();
    let mut positional_index = 0;

    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if let Some(param) = params.get(positional_index) {
                    bindings.insert((*param).clone(), value.clone());
                }
                positional_index += 1;
            }
            CallArgument::Keyword { name, value } => {
                bindings.insert(name.clone(), value.clone());
            }
        }
    }

    bindings
}

pub fn classify_known_function(target: &ResolvedTarget) -> Option<KnownFunction> {
    let (name, module) = target.parts.split_last()?;

    let is_jax = module == ["jax"];
    let is_jax_numpy = module == ["jax", "numpy"];
    let is_numpy = module == ["numpy"];
    let is_torch = module == ["torch"];
    let is_jax_lax = module == ["jax", "lax"];

    if is_jax {
        return match name.as_str() {
            "vmap" => Some(KnownFunction::Vmap),
            _ => None,
        };
    }

    if is_jax_numpy || is_numpy {
        return match name.as_str() {
            "concatenate" | "concat" => Some(KnownFunction::Concatenate),
            "stack" => Some(KnownFunction::Stack),
            "reshape" => Some(KnownFunction::Reshape),
            "transpose" => Some(KnownFunction::Transpose),
            "expand_dims" => Some(KnownFunction::ExpandDims),
            "squeeze" => Some(KnownFunction::Squeeze),
            "sum" => Some(KnownFunction::Sum),
            "mean" => Some(KnownFunction::Mean),
            "max" | "amax" => Some(KnownFunction::Max),
            "min" | "amin" => Some(KnownFunction::Min),
            "prod" => Some(KnownFunction::Prod),
            "std" => Some(KnownFunction::Std),
            "var" => Some(KnownFunction::Var),
            "matmul" => Some(KnownFunction::Matmul),
            "dot" => Some(KnownFunction::Dot),
            "einsum" => Some(KnownFunction::Einsum),
            "split" => Some(KnownFunction::Split),
            "tile" => Some(KnownFunction::Tile),
            "repeat" => Some(KnownFunction::Repeat),
            "flatten" => Some(KnownFunction::Flatten),
            "ravel" => Some(KnownFunction::Ravel),
            "moveaxis" => Some(KnownFunction::MoveAxis),
            "swapaxes" => Some(KnownFunction::SwapAxes),
            "where" => Some(KnownFunction::Where),
            "zeros" => Some(KnownFunction::Zeros),
            "ones" => Some(KnownFunction::Ones),
            "full" => Some(KnownFunction::Full),
            "arange" => Some(KnownFunction::Arange),
            "eye" => Some(KnownFunction::Eye),
            "broadcast_to" => Some(KnownFunction::BroadcastTo),
            "broadcast_arrays" => Some(KnownFunction::BroadcastArrays),
            "atleast_1d" => Some(KnownFunction::AtLeast1D),
            "atleast_2d" => Some(KnownFunction::AtLeast2D),
            "atleast_3d" => Some(KnownFunction::AtLeast3D),
            "pad" => Some(KnownFunction::Pad),
            "roll" => Some(KnownFunction::Roll),
            "flip" | "fliplr" | "flipud" => Some(KnownFunction::Flip),
            "rot90" => Some(KnownFunction::Rot90),
            "take" => Some(KnownFunction::Take),
            "diag" => Some(KnownFunction::Diag),
            "diagonal" => Some(KnownFunction::Diagonal),
            "trace" => Some(KnownFunction::Trace),
            "triu" => Some(KnownFunction::Triu),
            "tril" => Some(KnownFunction::Tril),
            "meshgrid" => Some(KnownFunction::Meshgrid),
            "vstack" | "row_stack" => Some(KnownFunction::Vstack),
            "hstack" => Some(KnownFunction::Hstack),
            "dstack" => Some(KnownFunction::Dstack),
            "column_stack" => Some(KnownFunction::ColumnStack),
            "block" => Some(KnownFunction::Block),
            "array" => Some(KnownFunction::Array),
            "asarray" => Some(KnownFunction::AsArray),
            _ => None,
        };
    }

    if is_torch {
        return match name.as_str() {
            "cat" | "concat" | "concatenate" => Some(KnownFunction::Concatenate),
            "stack" => Some(KnownFunction::Stack),
            "reshape" => Some(KnownFunction::Reshape),
            "transpose" => Some(KnownFunction::Transpose),
            "unsqueeze" => Some(KnownFunction::ExpandDims),
            "squeeze" => Some(KnownFunction::Squeeze),
            "sum" => Some(KnownFunction::Sum),
            "mean" => Some(KnownFunction::Mean),
            "max" => Some(KnownFunction::Max),
            "min" => Some(KnownFunction::Min),
            "prod" => Some(KnownFunction::Prod),
            "std" => Some(KnownFunction::Std),
            "var" => Some(KnownFunction::Var),
            "matmul" => Some(KnownFunction::Matmul),
            "dot" => Some(KnownFunction::Dot),
            "einsum" => Some(KnownFunction::Einsum),
            "split" => Some(KnownFunction::Split),
            "tile" => Some(KnownFunction::Tile),
            "repeat" => Some(KnownFunction::Repeat),
            "flatten" => Some(KnownFunction::Flatten),
            "ravel" => Some(KnownFunction::Ravel),
            "where" => Some(KnownFunction::Where),
            "zeros" => Some(KnownFunction::Zeros),
            "ones" => Some(KnownFunction::Ones),
            "full" => Some(KnownFunction::Full),
            "arange" => Some(KnownFunction::Arange),
            "eye" => Some(KnownFunction::Eye),
            "broadcast_to" => Some(KnownFunction::BroadcastTo),
            "broadcast_tensors" => Some(KnownFunction::BroadcastArrays),
            "atleast_1d" => Some(KnownFunction::AtLeast1D),
            "atleast_2d" => Some(KnownFunction::AtLeast2D),
            "atleast_3d" => Some(KnownFunction::AtLeast3D),
            "roll" => Some(KnownFunction::Roll),
            "flip" | "fliplr" | "flipud" => Some(KnownFunction::Flip),
            "rot90" => Some(KnownFunction::Rot90),
            "take" => Some(KnownFunction::Take),
            "diag" => Some(KnownFunction::Diag),
            "diagonal" => Some(KnownFunction::Diagonal),
            "trace" => Some(KnownFunction::Trace),
            "triu" => Some(KnownFunction::Triu),
            "tril" => Some(KnownFunction::Tril),
            "meshgrid" => Some(KnownFunction::Meshgrid),
            "vstack" | "row_stack" => Some(KnownFunction::Vstack),
            "hstack" => Some(KnownFunction::Hstack),
            "dstack" => Some(KnownFunction::Dstack),
            "column_stack" => Some(KnownFunction::ColumnStack),
            "permute" => Some(KnownFunction::Permute),
            "tensor" => Some(KnownFunction::Array),
            "as_tensor" => Some(KnownFunction::AsArray),
            _ => None,
        };
    }

    if is_jax_lax {
        return match name.as_str() {
            "dot" => Some(KnownFunction::Dot),
            "dot_general" => Some(KnownFunction::Matmul),
            _ => None,
        };
    }

    None
}

fn parse_simple_sequence_names(value: &str) -> Option<Vec<String>> {
    let trimmed = value.trim();
    let inner = if (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with('(') && trimmed.ends_with(')'))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };

    let names = inner
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| name.to_string())
        .collect::<Vec<_>>();

    if names.is_empty() { None } else { Some(names) }
}

fn parse_axis(value: &str) -> Option<isize> {
    value.trim().parse::<isize>().ok()
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim() {
        "True" | "true" => Some(true),
        "False" | "false" => Some(false),
        _ => None,
    }
}

fn parse_axis_list(value: &str) -> Option<Vec<isize>> {
    let trimmed = value.trim();
    if trimmed == "None" {
        return Some(vec![isize::MIN]);
    }
    if let Some(axis) = parse_axis(trimmed) {
        return Some(vec![axis]);
    }
    let inner = if (trimmed.starts_with('(') && trimmed.ends_with(')'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        return None;
    };
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(parse_axis)
        .collect()
}

fn first_array_arg<'a>(args: &'a [CallArgument]) -> Option<&'a str> {
    for arg in args {
        match arg {
            CallArgument::Positional { value } => return Some(value),
            CallArgument::Keyword { name, value }
                if name == "a" || name == "x" || name == "input" || name == "array" =>
            {
                return Some(value);
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    None
}

fn parse_shape_value(value: &str) -> Option<Vec<String>> {
    let trimmed = value.trim();
    let inner = if (trimmed.starts_with('(') && trimmed.ends_with(')'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };

    let dims = inner
        .split(',')
        .map(str::trim)
        .filter(|dim| !dim.is_empty())
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>();

    if dims.is_empty() { None } else { Some(dims) }
}

fn dim_product(dims: &[String]) -> Option<usize> {
    dims.iter()
        .map(|dim| dim.parse::<usize>())
        .try_fold(1usize, |acc, dim| dim.map(|dim| acc * dim).ok())
}

fn flattened_dim(dims: &[String]) -> String {
    if let Some(product) = dim_product(dims) {
        return product.to_string();
    }
    dims.join("*")
}

fn concat_dim(dims: &[String]) -> String {
    let parsed = dims
        .iter()
        .map(|dim| dim.parse::<usize>())
        .collect::<Result<Vec<_>, _>>();

    if let Ok(values) = parsed {
        return values.iter().sum::<usize>().to_string();
    }

    dims.join("+")
}

pub fn apply_known_function(
    function: &KnownFunction,
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    match function {
        KnownFunction::Concatenate => apply_known_concatenate(args, shapes),
        KnownFunction::Stack => apply_known_stack(args, shapes),
        KnownFunction::Reshape => apply_known_reshape(args, shapes),
        KnownFunction::Flatten | KnownFunction::Ravel => apply_known_flatten(args, shapes),
        KnownFunction::Transpose | KnownFunction::Permute => apply_known_transpose(args, shapes),
        KnownFunction::SwapAxes => apply_known_swapaxes(args, shapes),
        KnownFunction::MoveAxis => apply_known_moveaxis(args, shapes),
        KnownFunction::Sum
        | KnownFunction::Mean
        | KnownFunction::Max
        | KnownFunction::Min
        | KnownFunction::Prod
        | KnownFunction::Std
        | KnownFunction::Var => apply_known_reduction(args, shapes),
        _ => Ok(None),
    }
}

fn sequence_arg_value<'a>(args: &'a [CallArgument]) -> Option<&'a str> {
    let first_arg = args.first()?;
    match first_arg {
        CallArgument::Positional { value } => Some(value),
        CallArgument::Keyword { name, value }
            if name == "arrays" || name == "tensors" || name == "arys" =>
        {
            Some(value)
        }
        CallArgument::Keyword { .. } => None,
    }
}

fn axis_arg(args: &[CallArgument], default: isize) -> isize {
    let mut axis = default;
    for arg in args.iter().skip(1) {
        match arg {
            CallArgument::Positional { value } => {
                if let Some(parsed) = parse_axis(value) {
                    axis = parsed;
                }
            }
            CallArgument::Keyword { name, value } if name == "axis" || name == "dim" => {
                if let Some(parsed) = parse_axis(value) {
                    axis = parsed;
                }
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    axis
}

fn apply_known_concatenate(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(first_value) = sequence_arg_value(args) else {
        return Ok(None);
    };

    let Some(input_names) = parse_simple_sequence_names(first_value) else {
        return Ok(None);
    };

    let axis = axis_arg(args, 0);

    let mut input_shapes = Vec::new();
    for input_name in &input_names {
        let Some(shape) = shapes.get(input_name) else {
            return Ok(None);
        };
        input_shapes.push(shape.clone());
    }

    let Some(first_shape) = input_shapes.first() else {
        return Ok(None);
    };
    if first_shape.is_empty() {
        return Err("concatenate requires rank >= 1 inputs".to_string());
    }

    let rank = first_shape.len();
    let axis = if axis < 0 { rank as isize + axis } else { axis };
    if axis < 0 || axis as usize >= rank {
        return Err(format!(
            "concatenate axis {} out of bounds for rank {}",
            axis, rank
        ));
    }
    let axis = axis as usize;

    for shape in &input_shapes {
        if shape.len() != rank {
            return Err(format!(
                "concatenate expected all inputs to have rank {}, got rank {}",
                rank,
                shape.len()
            ));
        }
        for dim_idx in 0..rank {
            if dim_idx == axis {
                continue;
            }
            if shape[dim_idx] != first_shape[dim_idx] {
                return Err(format!(
                    "concatenate dimension mismatch at axis {}: expected {}, got {}",
                    dim_idx, first_shape[dim_idx], shape[dim_idx]
                ));
            }
        }
    }

    let mut output = first_shape.clone();
    let concat_dims = input_shapes
        .iter()
        .map(|shape| shape[axis].clone())
        .collect::<Vec<_>>();
    output[axis] = concat_dim(&concat_dims);

    Ok(Some(output))
}

fn apply_known_stack(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(first_value) = sequence_arg_value(args) else {
        return Ok(None);
    };
    let Some(input_names) = parse_simple_sequence_names(first_value) else {
        return Ok(None);
    };

    let mut input_shapes = Vec::new();
    for input_name in &input_names {
        let Some(shape) = shapes.get(input_name) else {
            return Ok(None);
        };
        input_shapes.push(shape.clone());
    }

    let Some(first_shape) = input_shapes.first() else {
        return Ok(None);
    };

    for shape in &input_shapes {
        if shape.len() != first_shape.len() {
            return Err(format!(
                "stack expected all inputs to have rank {}, got rank {}",
                first_shape.len(),
                shape.len()
            ));
        }
        for (dim_idx, (expected, got)) in first_shape.iter().zip(shape.iter()).enumerate() {
            if expected != got {
                return Err(format!(
                    "stack dimension mismatch at axis {}: expected {}, got {}",
                    dim_idx, expected, got
                ));
            }
        }
    }

    let output_rank = first_shape.len() + 1;
    let axis = axis_arg(args, 0);
    let axis = if axis < 0 {
        output_rank as isize + axis
    } else {
        axis
    };
    if axis < 0 || axis as usize > first_shape.len() {
        return Err(format!(
            "stack axis {} out of bounds for output rank {}",
            axis, output_rank
        ));
    }

    let mut output = first_shape.clone();
    output.insert(axis as usize, input_shapes.len().to_string());
    Ok(Some(output))
}

fn apply_known_reshape(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };

    let mut shape_value = None;
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                if shape_value.is_none() {
                    shape_value = Some(value.as_str());
                }
            }
            CallArgument::Keyword { name, value }
                if name == "shape" || name == "newshape" || name == "size" =>
            {
                shape_value = Some(value.as_str());
            }
            CallArgument::Keyword { .. } => {}
        }
    }

    let Some(shape_value) = shape_value else {
        return Ok(None);
    };
    let Some(mut output_shape) = parse_shape_value(shape_value) else {
        return Ok(None);
    };

    let minus_one_count = output_shape
        .iter()
        .filter(|dim| dim.as_str() == "-1")
        .count();
    if minus_one_count > 1 {
        return Err("reshape can only infer one -1 dimension".to_string());
    }

    if minus_one_count == 1 {
        let Some(input_product) = dim_product(input_shape) else {
            return Err("reshape cannot infer -1 dimension for symbolic input shape".to_string());
        };
        let known_dims = output_shape
            .iter()
            .filter(|dim| dim.as_str() != "-1")
            .cloned()
            .collect::<Vec<_>>();
        let Some(known_product) = dim_product(&known_dims) else {
            return Err("reshape cannot infer -1 dimension with symbolic target shape".to_string());
        };
        if known_product == 0 || input_product % known_product != 0 {
            return Err(format!(
                "reshape cannot infer -1 dimension: input size {} not divisible by {}",
                input_product, known_product
            ));
        }
        let inferred = (input_product / known_product).to_string();
        for dim in &mut output_shape {
            if dim == "-1" {
                *dim = inferred.clone();
            }
        }
    }

    if minus_one_count == 0 {
        if let (Some(input_product), Some(output_product)) =
            (dim_product(input_shape), dim_product(&output_shape))
        {
            if input_product != output_product {
                return Err(format!(
                    "reshape changes total size from {} to {}",
                    input_product, output_product
                ));
            }
        }
    }

    Ok(Some(output_shape))
}

fn apply_known_flatten(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };

    Ok(Some(vec![flattened_dim(input_shape)]))
}

fn normalize_axis(axis: isize, rank: usize, context: &str) -> Result<usize, String> {
    let normalized = if axis < 0 { rank as isize + axis } else { axis };
    if normalized < 0 || normalized as usize >= rank {
        return Err(format!(
            "{} axis {} out of bounds for rank {}",
            context, normalized, rank
        ));
    }
    Ok(normalized as usize)
}

fn apply_known_transpose(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    let rank = input_shape.len();

    let mut axes = None;
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                axes = parse_axis_list(value);
            }
            CallArgument::Keyword { name, value }
                if name == "axes" || name == "axis" || name == "dims" =>
            {
                axes = parse_axis_list(value);
            }
            CallArgument::Keyword { .. } => {}
        }
    }

    let axes = axes.unwrap_or_else(|| (0..rank).rev().map(|axis| axis as isize).collect());
    if axes.len() != rank {
        return Err(format!(
            "transpose expected {} axes, got {}",
            rank,
            axes.len()
        ));
    }
    let mut normalized = Vec::new();
    for axis in axes {
        let axis = normalize_axis(axis, rank, "transpose")?;
        if normalized.contains(&axis) {
            return Err(format!("transpose duplicate axis {}", axis));
        }
        normalized.push(axis);
    }

    Ok(Some(
        normalized
            .iter()
            .map(|axis| input_shape[*axis].clone())
            .collect(),
    ))
}

fn apply_known_swapaxes(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    let rank = input_shape.len();

    let mut axes = Vec::new();
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                if let Some(axis) = parse_axis(value) {
                    axes.push(axis);
                }
            }
            CallArgument::Keyword { name, value }
                if name == "axis1" || name == "axis2" || name == "dim0" || name == "dim1" =>
            {
                if let Some(axis) = parse_axis(value) {
                    axes.push(axis);
                }
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    if axes.len() != 2 {
        return Ok(None);
    }

    let axis0 = normalize_axis(axes[0], rank, "swapaxes")?;
    let axis1 = normalize_axis(axes[1], rank, "swapaxes")?;
    let mut output = input_shape.clone();
    output.swap(axis0, axis1);
    Ok(Some(output))
}

fn apply_known_moveaxis(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };
    let rank = input_shape.len();

    let mut source = None;
    let mut destination = None;
    let mut positional = Vec::new();
    let mut seen_first_positional = false;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                positional.push(value.as_str());
            }
            CallArgument::Keyword { name, value } if name == "source" => {
                source = parse_axis_list(value)
            }
            CallArgument::Keyword { name, value } if name == "destination" => {
                destination = parse_axis_list(value)
            }
            CallArgument::Keyword { .. } => {}
        }
    }
    if source.is_none() && !positional.is_empty() {
        source = parse_axis_list(positional[0]);
    }
    if destination.is_none() && positional.len() > 1 {
        destination = parse_axis_list(positional[1]);
    }
    let Some(source) = source else {
        return Ok(None);
    };
    let Some(destination) = destination else {
        return Ok(None);
    };
    if source.len() != destination.len() {
        return Err("moveaxis source and destination lengths differ".to_string());
    }

    let source = source
        .into_iter()
        .map(|axis| normalize_axis(axis, rank, "moveaxis"))
        .collect::<Result<Vec<_>, _>>()?;
    let destination = destination
        .into_iter()
        .map(|axis| normalize_axis(axis, rank, "moveaxis"))
        .collect::<Result<Vec<_>, _>>()?;

    let mut order = (0..rank)
        .filter(|axis| !source.contains(axis))
        .collect::<Vec<_>>();
    for (src, dst) in source.iter().zip(destination.iter()) {
        let insert_at = (*dst).min(order.len());
        order.insert(insert_at, *src);
    }

    Ok(Some(
        order
            .iter()
            .map(|axis| input_shape[*axis].clone())
            .collect(),
    ))
}

fn apply_known_reduction(
    args: &[CallArgument],
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_name) = first_array_arg(args) else {
        return Ok(None);
    };
    let Some(input_shape) = shapes.get(input_name) else {
        return Ok(None);
    };

    let mut axes: Option<Vec<isize>> = None;
    let mut keepdims = false;
    let mut seen_first_positional = false;

    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if !seen_first_positional {
                    seen_first_positional = true;
                    continue;
                }
                if axes.is_none() {
                    axes = parse_axis_list(value);
                }
            }
            CallArgument::Keyword { name, value } if name == "axis" || name == "dim" => {
                axes = parse_axis_list(value);
            }
            CallArgument::Keyword { name, value } if name == "keepdims" || name == "keepdim" => {
                if let Some(parsed) = parse_bool(value) {
                    keepdims = parsed;
                }
            }
            CallArgument::Keyword { .. } => {}
        }
    }

    let rank = input_shape.len();
    let Some(axes) = axes else {
        if keepdims {
            return Ok(Some(vec!["1".to_string(); rank]));
        }
        return Ok(Some(Vec::new()));
    };

    if axes.is_empty() {
        return Ok(Some(input_shape.clone()));
    }

    let axes = if axes.contains(&isize::MIN) {
        (0..rank).map(|axis| axis as isize).collect::<Vec<_>>()
    } else {
        axes
    };

    let mut normalized = Vec::new();
    for axis in axes {
        let axis = if axis < 0 { rank as isize + axis } else { axis };
        if axis < 0 || axis as usize >= rank {
            return Err(format!(
                "reduction axis {} out of bounds for rank {}",
                axis, rank
            ));
        }
        let axis = axis as usize;
        if normalized.contains(&axis) {
            return Err(format!("duplicate reduction axis {}", axis));
        }
        normalized.push(axis);
    }
    normalized.sort_unstable();

    if keepdims {
        let mut output = input_shape.clone();
        for axis in normalized {
            output[axis] = "1".to_string();
        }
        return Ok(Some(output));
    }

    let output = input_shape
        .iter()
        .enumerate()
        .filter(|(idx, _)| !normalized.contains(idx))
        .map(|(_, dim)| dim.clone())
        .collect();

    Ok(Some(output))
}

pub fn classify_layer_call(call: &ResolvedCallSignature) -> Option<LayerKind> {
    let is_equinox_module = call.implementation.target.module_parts.len() >= 2
        && call.implementation.target.module_parts[0] == "equinox"
        && call.implementation.target.module_parts[1] == "nn";
    let is_linear_init =
        call.signature.owner.as_deref() == Some("Linear") && call.signature.name == "__init__";

    if !is_equinox_module || !is_linear_init {
        return None;
    }

    Some(LayerKind::Linear {
        in_features: call.bindings.get("in_features")?.clone(),
        out_features: call.bindings.get("out_features")?.clone(),
    })
}

pub fn extract_layer_assignments<F>(
    node: Node,
    text: &str,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
) -> Result<HashMap<String, LayerKind>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let import_map = build_import_map(node, text)?;
    let calls = extract_calls(node, text)?;
    let mut layers = HashMap::new();

    for call in calls {
        let Some(resolved_call) = resolve_call_signature(
            &call,
            text,
            &import_map,
            search_roots,
            &read_file,
            max_depth,
        )?
        else {
            continue;
        };
        let Some(layer) = classify_layer_call(&resolved_call) else {
            continue;
        };
        layers.insert(call.variable, layer);
    }

    Ok(layers)
}

pub fn extract_layer_applications(
    node: Node,
    text: &str,
    layers: &HashMap<String, LayerKind>,
) -> Result<Vec<LayerApplication>, String> {
    let calls = extract_calls(node, text)?;
    let mut applications = Vec::new();

    for call in calls {
        let Some(kind) = layers.get(&call.target) else {
            continue;
        };
        let Some(args_node) = node.descendant_for_byte_range(
            call.args_node_range.start_byte,
            call.args_node_range.end_byte,
        ) else {
            continue;
        };
        let args = extract_call_arguments(args_node, text)?;
        let Some(CallArgument::Positional { value }) = args.first() else {
            continue;
        };
        applications.push(LayerApplication {
            variable: call.variable,
            layer: call.target,
            input: value.clone(),
            kind: kind.clone(),
            range: call.args_node_range,
        });
    }

    Ok(applications)
}

pub fn apply_layer_application(
    app: &LayerApplication,
    shapes: &HashMap<String, Vec<String>>,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_shape) = shapes.get(&app.input) else {
        return Ok(None);
    };

    match &app.kind {
        LayerKind::Linear {
            in_features,
            out_features,
        } => {
            let Some(last_dim) = input_shape.last() else {
                return Err(format!(
                    "Cannot apply linear layer '{}' to scalar input '{}'",
                    app.layer, app.input
                ));
            };

            if last_dim != in_features {
                return Err(format!(
                    "Linear layer '{}' expected input last dim {}, got {} for '{}'",
                    app.layer, in_features, last_dim, app.input
                ));
            }

            let mut output_shape = input_shape.clone();
            let last = output_shape.len() - 1;
            output_shape[last] = out_features.clone();
            Ok(Some(output_shape))
        }
    }
}

pub fn apply_layer_applications(
    apps: &[LayerApplication],
    shapes: &mut HashMap<String, Vec<String>>,
) -> Vec<ShapeError> {
    let mut errors = Vec::new();

    for app in apps {
        match apply_layer_application(app, shapes) {
            Ok(Some(output_shape)) => {
                shapes.insert(app.variable.clone(), output_shape);
            }
            Ok(None) => {}
            Err(message) => errors.push(ShapeError {
                variable: app.variable.clone(),
                message,
                range: app.range,
            }),
        }
    }

    errors
}

pub fn analyze_layer_shapes<F>(
    node: Node,
    text: &str,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
) -> Result<LayerShapeAnalysis, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let mut shapes = extract_jaxtyping_shapes(node, text)?;
    let layers = extract_layer_assignments(node, text, search_roots, read_file, max_depth)?;
    let applications = extract_layer_applications(node, text, &layers)?;
    let errors = apply_layer_applications(&applications, &mut shapes);

    Ok(LayerShapeAnalysis {
        shapes,
        layers,
        applications,
        errors,
    })
}

pub fn extract_jaxtyping_shapes(
    node: Node,
    text: &str,
) -> Result<HashMap<String, Vec<String>>, String> {
    fn node_text(node: Node, text: &str) -> Result<String, String> {
        node.utf8_text(text.as_bytes())
            .map(|s| s.to_string())
            .map_err(|e| e.to_string())
    }

    fn first_identifier(node: Node, text: &str) -> Result<Option<String>, String> {
        if node.kind() == "identifier" {
            return Ok(Some(node_text(node, text)?));
        }
        if let Some(name) = node.child_by_field_name("name") {
            return Ok(Some(node_text(name, text)?));
        }
        let type_node = node.child_by_field_name("type");
        for i in 0..node.named_child_count() {
            let Some(child) = node.named_child(i as u32) else {
                continue;
            };
            if type_node == Some(child) {
                continue;
            }
            if let Some(name) = first_identifier(child, text)? {
                return Ok(Some(name));
            }
        }
        Ok(None)
    }

    fn find_string_literal(node: Node, text: &str) -> Result<Option<String>, String> {
        if node.kind() == "string" {
            return Ok(Some(node_text(node, text)?));
        }
        for i in 0..node.named_child_count() {
            let Some(child) = node.named_child(i as u32) else {
                continue;
            };
            if let Some(value) = find_string_literal(child, text)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    fn contains_array_type(node: Node, text: &str) -> Result<bool, String> {
        let node_text = node_text(node, text)?;
        if node.kind() == "identifier" && node_text == "Array" {
            return Ok(true);
        }
        if node.kind() == "attribute" && node_text.ends_with(".Array") {
            return Ok(true);
        }

        for i in 0..node.named_child_count() {
            let Some(child) = node.named_child(i as u32) else {
                continue;
            };
            if contains_array_type(child, text)? {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn shape_dims(raw_string: &str) -> Vec<String> {
        let trimmed = raw_string.trim();
        let Some(quote_start) = trimmed.find(['"', '\'']) else {
            return Vec::new();
        };
        let prefix = trimmed[..quote_start].to_ascii_lowercase();
        if prefix.chars().any(|c| c != 'r' && c != 'u') {
            return Vec::new();
        }

        let quoted = &trimmed[quote_start..];
        let quote = quoted.chars().next().unwrap();
        let triple = quote.to_string().repeat(3);
        let unquoted = if quoted.starts_with(&triple) && quoted.ends_with(&triple) {
            &quoted[3..quoted.len() - 3]
        } else if quoted.starts_with(quote) && quoted.ends_with(quote) {
            &quoted[1..quoted.len() - 1]
        } else {
            return Vec::new();
        };

        unquoted
            .split_whitespace()
            .filter(|dim| !dim.is_empty())
            .map(|dim| dim.to_string())
            .collect()
    }

    fn visit(
        node: Node,
        text: &str,
        shapes: &mut HashMap<String, Vec<String>>,
    ) -> Result<(), String> {
        if node.kind() == "function_definition" {
            if let Some(parameters) = node.child_by_field_name("parameters") {
                for i in 0..parameters.named_child_count() {
                    let Some(parameter) = parameters.named_child(i as u32) else {
                        continue;
                    };
                    let Some(type_node) = parameter.child_by_field_name("type") else {
                        continue;
                    };
                    if !contains_array_type(type_node, text)? {
                        continue;
                    }
                    let Some(raw_shape) = find_string_literal(type_node, text)? else {
                        continue;
                    };
                    let dims = shape_dims(&raw_shape);
                    if dims.is_empty() {
                        continue;
                    }
                    let Some(name) = first_identifier(parameter, text)? else {
                        continue;
                    };
                    shapes.insert(name, dims);
                }
            }
        }

        for i in 0..node.named_child_count() {
            let Some(child) = node.named_child(i as u32) else {
                continue;
            };
            visit(child, text, shapes)?;
        }

        Ok(())
    }

    let mut shapes = HashMap::new();
    visit(node, text, &mut shapes)?;
    Ok(shapes)
}

pub fn find_top_level_symbol(
    node: Node,
    text: &str,
    name: &str,
) -> Result<Option<PythonSymbol>, String> {
    let query_string = r#"
        (module (class_definition name: (_) @cls_def))
        (module (function_definition name: (_) @fn_def))
        (module (import_statement) @import)
        (module (import_from_statement) @import)
    "#;

    let query = Query::new(&tree_sitter_python::LANGUAGE.into(), query_string)
        .map_err(|e| e.to_string())?;
    let Some(class_idx) = query.capture_index_for_name("cls_def") else {
        return Err("Failed to find capture index for 'cls_def'".to_string());
    };

    let Some(fn_idx) = query.capture_index_for_name("fn_def") else {
        return Err("Failed to find capture index for 'fn_def'".to_string());
    };
    let Some(import_idx) = query.capture_index_for_name("import") else {
        return Err("Failed to find capture index for 'import'".to_string());
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, node, text.as_bytes());
    let mut found = None;

    while let Some(match_) = matches.next() {
        match match_.pattern_index {
            0 => {
                for capture in match_.captures {
                    if capture.index == class_idx {
                        let class_name = capture
                            .node
                            .utf8_text(text.as_bytes())
                            .map_err(|e| e.to_string())?;

                        if class_name == name {
                            found = Some(PythonSymbol::Class {
                                name: class_name.to_string(),
                            });
                        }
                    }
                }
            }
            1 => {
                for capture in match_.captures {
                    if capture.index == fn_idx {
                        let fn_name = capture
                            .node
                            .utf8_text(text.as_bytes())
                            .map_err(|e| e.to_string())?;

                        if fn_name == name {
                            found = Some(PythonSymbol::Function {
                                name: fn_name.to_string(),
                            });
                        }
                    }
                }
            }
            2 | 3 => {
                for capture in match_.captures {
                    if capture.index == import_idx {
                        let import_map = build_import_map(capture.node, text)?;
                        if let Some(path) = import_map.get(name) {
                            found = Some(PythonSymbol::Import {
                                name: name.to_string(),
                                path: path.clone(),
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(found)
}

pub fn build_import_map(node: Node, text: &str) -> Result<HashMap<String, ImportPath>, String> {
    fn plain_import_dotted_name_to_vec(
        dotted_name: &str,
    ) -> Result<(Vec<String>, String, String), String> {
        let parts: Vec<&str> = dotted_name.split(".").collect();
        let module: Vec<String> = parts[..parts.len() - 1]
            .iter()
            .map(|p| p.to_string())
            .collect();
        let Some(name) = parts.last() else {
            return Err("Failed to fetch the last part of the import path".to_string());
        };
        let Some(first) = parts.first() else {
            return Err("Empty import path".to_string());
        };
        Ok((module, name.to_string(), first.to_string()))
    }

    let query_string = r#"
        (import_statement name: (_) @import)
        (import_from_statement module_name: (_) @module_name name: (_) @name)
    "#;
    let query = Query::new(&tree_sitter_python::LANGUAGE.into(), query_string)
        .map_err(|e| e.to_string())?;

    let Some(import_statement_idx) = query.capture_index_for_name("import") else {
        return Err("Failed to capture index for name 'import'".to_string());
    };

    let Some(module_name_idx) = query.capture_index_for_name("module_name") else {
        return Err("Failed to capture index for name 'module_name'".to_string());
    };
    let Some(name_idx) = query.capture_index_for_name("name") else {
        return Err("Failed to capture index for name 'name'".to_string());
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, node, text.as_bytes());

    let mut import_map: HashMap<String, ImportPath> = HashMap::new();

    while let Some(match_) = matches.next() {
        if match_.pattern_index == 0 {
            for capture in match_.captures {
                if capture.index == import_statement_idx {
                    match capture.node.kind() {
                        "dotted_name" => {
                            let dotted_name = capture
                                .node
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?;
                            let (module, name, first) =
                                plain_import_dotted_name_to_vec(dotted_name)
                                    .map_err(|e| e.to_string())?;
                            import_map.insert(
                                first.to_string(),
                                ImportPath {
                                    dots: 0,
                                    module,
                                    name: name.to_string(),
                                },
                            );
                        }
                        "aliased_import" => {
                            let Some(name_child) = capture.node.child_by_field_name("name") else {
                                return Err("aliased_import missing name".to_string());
                            };
                            let dotted_name = name_child
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?;
                            let (module, name, _) = plain_import_dotted_name_to_vec(dotted_name)?;

                            let Some(alias_node) = capture.node.child_by_field_name("alias") else {
                                return Err("aliased_import missing alias".to_string());
                            };
                            let alias = alias_node
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?
                                .to_string();

                            import_map.insert(
                                alias,
                                ImportPath {
                                    dots: 0,
                                    module,
                                    name,
                                },
                            );
                        }
                        _ => {}
                    }
                }
            }
        } else if match_.pattern_index == 1 {
            let mut module: Vec<String> = Vec::new();
            let mut name = String::new();
            let mut alias = String::new();
            let mut dots = 0;
            for capture in match_.captures {
                if capture.index == module_name_idx {
                    let n = capture.node;
                    match n.kind() {
                        "dotted_name" => {
                            let identifier =
                                n.utf8_text(text.as_bytes()).map_err(|e| e.to_string())?;
                            module = identifier.split(".").map(|f| f.to_string()).collect();
                        }
                        "relative_import" => {
                            let mut child_cursor = n.walk();
                            for child in n.named_children(&mut child_cursor) {
                                match child.kind() {
                                    "import_prefix" => {
                                        let prefix = child
                                            .utf8_text(text.as_bytes())
                                            .map_err(|e| e.to_string())?;
                                        dots = prefix.len();
                                    }
                                    "dotted_name" => {
                                        let dotted = child
                                            .utf8_text(text.as_bytes())
                                            .map_err(|e| e.to_string())?;
                                        module = dotted.split(".").map(|f| f.to_string()).collect();
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                } else if capture.index == name_idx {
                    let n = capture.node;
                    match n.kind() {
                        "dotted_name" => {
                            name = n
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?
                                .to_string();
                            alias = name.clone();
                        }
                        "aliased_import" => {
                            let Some(alias_child_node) = n.child_by_field_name("alias") else {
                                return Err("aliased_import has no alias field".to_string());
                            };
                            alias = alias_child_node
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?
                                .to_string();

                            let Some(dotted_name_idx) = n.child_by_field_name("name") else {
                                return Err("aliased_import has no name field".to_string());
                            };
                            let dotted_name = dotted_name_idx
                                .utf8_text(text.as_bytes())
                                .map_err(|e| e.to_string())?
                                .to_string();
                            name = dotted_name;
                        }
                        _ => {}
                    }
                }
            }

            import_map.insert(alias, ImportPath { dots, module, name });
        }
    }

    Ok(import_map)
}

pub fn extract_calls(node: Node, text: &str) -> Result<Vec<CallInfo>, String> {
    let mut result: Vec<CallInfo> = Vec::new();
    let query_string = r#"
        (assignment
          left: (identifier) @var
          right: (call
            function: [
              (identifier) @fn_id
              (attribute) @fn_attr
            ]
          )
        ) @assignment
    "#;

    let query = Query::new(&tree_sitter_python::LANGUAGE.into(), query_string)
        .map_err(|e| e.to_string())?;

    let Some(var_idx) = query.capture_index_for_name("var") else {
        return Err("var not found".to_string());
    };

    let Some(fn_id_idx) = query.capture_index_for_name("fn_id") else {
        return Err("fn_id not found".to_string());
    };

    let Some(fn_attr_idx) = query.capture_index_for_name("fn_attr") else {
        return Err("fn_attr not found".to_string());
    };

    let Some(assignment_idx) = query.capture_index_for_name("assignment") else {
        return Err("assignment not found".to_string());
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, node, text.as_bytes());

    while let Some(match_) = matches.next() {
        let mut variable = String::new();
        let mut target = String::new();
        let mut args_node_range: Option<Range> = None;

        for capture in match_.captures {
            if capture.index == var_idx {
                variable = capture
                    .node
                    .utf8_text(text.as_bytes())
                    .map_err(|e| e.to_string())?
                    .to_string();
            } else if capture.index == assignment_idx {
                let Some(right_child_node) = capture.node.child_by_field_name("right") else {
                    return Err("right child not found".to_string());
                };

                let Some(arguments_node) = right_child_node.child_by_field_name("arguments") else {
                    return Err("arguments child not found".to_string());
                };
                args_node_range = Some(arguments_node.range());
            } else if capture.index == fn_id_idx || capture.index == fn_attr_idx {
                target = capture
                    .node
                    .utf8_text(text.as_bytes())
                    .map_err(|e| e.to_string())?
                    .to_string();
            }
        }

        let args_node_range = args_node_range.ok_or("args_node_range not found".to_string())?;
        result.push(CallInfo {
            variable,
            target,
            args_node_range,
        });
    }

    Ok(result)
}

#[cfg(test)]
mod resolve_import_path_from_package_tests {
    use super::*;

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: parts(module),
            name: name.to_string(),
        }
    }

    fn target(parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots: 0,
            parts: self::parts(parts),
        }
    }

    #[test]
    fn test_absolute_import_ignores_current_package() {
        let found = resolve_import_path_from_package(
            &parts(&["equinox", "nn"]),
            &ip(0, &["jax"], "random"),
        );

        assert_eq!(found, Some(target(&["jax", "random"])));
    }

    #[test]
    fn test_relative_import_same_package_with_module() {
        let found = resolve_import_path_from_package(
            &parts(&["equinox", "nn"]),
            &ip(1, &["_linear"], "Linear"),
        );

        assert_eq!(found, Some(target(&["equinox", "nn", "_linear", "Linear"])));
    }

    #[test]
    fn test_relative_import_same_package_without_module() {
        let found =
            resolve_import_path_from_package(&parts(&["equinox", "nn"]), &ip(1, &[], "layers"));

        assert_eq!(found, Some(target(&["equinox", "nn", "layers"])));
    }

    #[test]
    fn test_relative_import_parent_package() {
        let found = resolve_import_path_from_package(&parts(&["pkg", "sub"]), &ip(2, &["x"], "Y"));

        assert_eq!(found, Some(target(&["pkg", "x", "Y"])));
    }

    #[test]
    fn test_relative_import_grandparent_package() {
        let found =
            resolve_import_path_from_package(&parts(&["pkg", "sub", "inner"]), &ip(3, &["x"], "Y"));

        assert_eq!(found, Some(target(&["pkg", "x", "Y"])));
    }

    #[test]
    fn test_relative_import_too_many_dots_returns_none() {
        let found = resolve_import_path_from_package(&parts(&["pkg", "sub"]), &ip(3, &["x"], "Y"));

        assert_eq!(found, None);
    }

    #[test]
    fn test_relative_import_from_empty_package_returns_none() {
        let found = resolve_import_path_from_package(&[], &ip(1, &["x"], "Y"));

        assert_eq!(found, None);
    }

    #[test]
    fn test_empty_absolute_import_name_is_kept() {
        let found = resolve_import_path_from_package(&parts(&["pkg"]), &ip(0, &[], "foo"));

        assert_eq!(found, Some(target(&["foo"])));
    }
}

#[cfg(test)]
mod extract_jaxtyping_shapes_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_extracts_single_jaxtyping_shape() {
        let code = "def f(x: Float[Array, \"batch features\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_extracts_concrete_integer_dimension() {
        let code = "def f(x: Float[Array, \"batch 3\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "3"])));
    }

    #[test]
    fn test_extracts_multiple_parameters() {
        let code = "def f(x: Float[Array, \"b d\"], y: Int[Array, \"b\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["b", "d"])));
        assert_eq!(shapes.get("y"), Some(&shape(&["b"])));
    }

    #[test]
    fn test_extracts_typed_default_parameter() {
        let code = "def f(x: Float[Array, \"b d\"] = default): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["b", "d"])));
    }

    #[test]
    fn test_skips_unannotated_parameter() {
        let code = "def f(x, y: Float[Array, \"b d\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(!shapes.contains_key("x"));
        assert_eq!(shapes.get("y"), Some(&shape(&["b", "d"])));
    }

    #[test]
    fn test_skips_annotation_without_shape_string() {
        let code = "def f(x: int): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(shapes.is_empty());
    }

    #[test]
    fn test_extracts_shapes_inside_nested_function() {
        let code = "def outer():\n    def inner(x: Float[Array, \"b d\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["b", "d"])));
    }

    #[test]
    fn test_extracts_single_quoted_shape() {
        let code = "def f(x: Float[Array, 'batch features']): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_extracts_multiline_function_signature() {
        let code = "def f(\n    x: Float[Array, \"batch features\"],\n    y: Float[Array, \"batch\"],\n): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
        assert_eq!(shapes.get("y"), Some(&shape(&["batch"])));
    }

    #[test]
    fn test_extracts_method_parameter_but_skips_self() {
        let code = "class M:\n    def __call__(self, x: Float[Array, \"batch features\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(!shapes.contains_key("self"));
        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_return_annotation_is_ignored() {
        let code = "def f(x) -> Float[Array, \"batch features\"]: pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(shapes.is_empty());
    }

    #[test]
    fn test_non_array_string_annotation_is_skipped() {
        let code = "def f(x: Literal[\"batch features\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(shapes.is_empty());
    }

    #[test]
    fn test_notarray_identifier_is_not_treated_as_array() {
        let code = "def f(x: Float[NotArray, \"batch features\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(shapes.is_empty());
    }

    #[test]
    fn test_qualified_jax_array_is_accepted() {
        let code = "def f(x: Float[jax.Array, \"batch features\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_qualified_jaxtyping_array_is_accepted() {
        let code = "def f(x: Float[jaxtyping.Array, \"batch features\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_shaped_array_annotation_is_accepted() {
        let code = "def f(x: Shaped[Array, \"batch features\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_optional_wrapped_array_annotation_is_accepted() {
        let code = "def f(x: Optional[Float[Array, \"batch features\"]]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_raw_shape_string_is_accepted() {
        let code = "def f(x: Float[Array, r\"batch features\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_triple_quoted_shape_string_is_accepted() {
        let code = "def f(x: Float[Array, \"\"\"batch features\"\"\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_f_string_shape_is_skipped() {
        let code = "def f(x: Float[Array, f\"batch {features}\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(shapes.is_empty());
    }

    #[test]
    fn test_annotated_varargs_are_extracted() {
        let code = "def f(*xs: Float[Array, \"batch features\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("xs"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_annotated_kwargs_are_extracted() {
        let code = "def f(**kwargs: Float[Array, \"batch features\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("kwargs"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_keyword_only_annotated_parameter_is_extracted() {
        let code = "def f(*, x: Float[Array, \"batch features\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_positional_only_annotated_parameter_is_extracted() {
        let code = "def f(x: Float[Array, \"batch features\"], /): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_deeply_nested_union_annotation_is_extracted() {
        let code = "def f(x: Union[None, Float[Array, \"batch features\"]]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_bytes_shape_string_is_skipped() {
        let code = "def f(x: Float[Array, b\"batch features\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(shapes.is_empty());
    }

    #[test]
    fn test_comment_near_annotation_does_not_affect_shape() {
        let code = "def f(\n    x: Float[Array, \"batch features\"],  # important\n): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_empty_shape_string_is_skipped() {
        let code = "def f(x: Float[Array, \"\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(shapes.is_empty());
    }

    #[test]
    fn test_extra_spaces_in_shape_string_are_ignored() {
        let code = "def f(x: Float[Array, \"  batch   features  \"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_preserves_punctuation_inside_dimension_names() {
        let code = "def f(x: Float[Array, \"batch hidden*2\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "hidden*2"])));
    }

    #[test]
    fn test_later_parameter_name_wins() {
        let code = "def f(x: Float[Array, \"a b\"]): pass\ndef g(x: Float[Array, \"c d\"]): pass";
        let tree = parse(code);

        let shapes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(shapes.get("x"), Some(&shape(&["c", "d"])));
    }
}

#[cfg(test)]
mod apply_layer_application_tests {
    use super::*;

    fn linear(in_features: &str, out_features: &str) -> LayerKind {
        LayerKind::Linear {
            in_features: in_features.to_string(),
            out_features: out_features.to_string(),
        }
    }

    fn dummy_range() -> Range {
        Range {
            start_byte: 0,
            end_byte: 0,
            start_point: tree_sitter::Point::new(0, 0),
            end_point: tree_sitter::Point::new(0, 0),
        }
    }

    fn app(input: &str, kind: LayerKind) -> LayerApplication {
        LayerApplication {
            variable: "y".to_string(),
            layer: "layer".to_string(),
            input: input.to_string(),
            kind,
            range: dummy_range(),
        }
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_applies_linear_to_rank_2_shape() {
        let app = app("x", linear("3", "5"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let output = apply_layer_application(&app, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "5"])));
    }

    #[test]
    fn test_applies_linear_to_rank_1_shape() {
        let app = app("x", linear("3", "5"));
        let shapes = HashMap::from([("x".to_string(), shape(&["3"]))]);

        let output = apply_layer_application(&app, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["5"])));
    }

    #[test]
    fn test_applies_linear_to_symbolic_shape() {
        let app = app("x", linear("features", "hidden"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_layer_application(&app, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "hidden"])));
    }

    #[test]
    fn test_preserves_leading_dimensions() {
        let app = app("x", linear("features", "hidden"));
        let shapes = HashMap::from([("x".to_string(), shape(&["time", "batch", "features"]))]);

        let output = apply_layer_application(&app, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["time", "batch", "hidden"])));
    }

    #[test]
    fn test_missing_input_shape_returns_none() {
        let app = app("x", linear("3", "5"));
        let shapes = HashMap::new();

        let output = apply_layer_application(&app, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_empty_input_shape_returns_error() {
        let app = app("x", linear("3", "5"));
        let shapes = HashMap::from([("x".to_string(), Vec::new())]);

        let error = apply_layer_application(&app, &shapes).unwrap_err();

        assert!(error.contains("scalar input"));
    }

    #[test]
    fn test_last_dim_mismatch_returns_error() {
        let app = app("x", linear("3", "5"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "4"]))]);

        let error = apply_layer_application(&app, &shapes).unwrap_err();

        assert!(error.contains("expected input last dim 3"));
        assert!(error.contains("got 4"));
    }

    #[test]
    fn test_symbolic_mismatch_returns_error() {
        let app = app("x", linear("features", "hidden"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "other"]))]);

        let error = apply_layer_application(&app, &shapes).unwrap_err();

        assert!(error.contains("expected input last dim features"));
        assert!(error.contains("got other"));
    }

    #[test]
    fn test_numeric_and_symbolic_dims_do_not_match() {
        let app = app("x", linear("features", "hidden"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let error = apply_layer_application(&app, &shapes).unwrap_err();

        assert!(error.contains("expected input last dim features"));
        assert!(error.contains("got 3"));
    }

    #[test]
    fn test_same_in_and_out_features_is_allowed() {
        let app = app("x", linear("features", "features"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_layer_application(&app, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_input_shape_map_is_not_mutated() {
        let app = app("x", linear("3", "5"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let output = apply_layer_application(&app, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "5"])));
        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "3"])));
    }

    #[test]
    fn test_error_mentions_layer_and_input_names() {
        let mut app = app("input", linear("3", "5"));
        app.layer = "projection".to_string();
        let shapes = HashMap::from([("input".to_string(), shape(&["batch", "4"]))]);

        let error = apply_layer_application(&app, &shapes).unwrap_err();

        assert!(error.contains("projection"));
        assert!(error.contains("input"));
    }
}

#[cfg(test)]
mod apply_layer_applications_tests {
    use super::*;

    fn linear(in_features: &str, out_features: &str) -> LayerKind {
        LayerKind::Linear {
            in_features: in_features.to_string(),
            out_features: out_features.to_string(),
        }
    }

    fn dummy_range() -> Range {
        Range {
            start_byte: 0,
            end_byte: 0,
            start_point: tree_sitter::Point::new(0, 0),
            end_point: tree_sitter::Point::new(0, 0),
        }
    }

    fn app(variable: &str, layer: &str, input: &str, kind: LayerKind) -> LayerApplication {
        LayerApplication {
            variable: variable.to_string(),
            layer: layer.to_string(),
            input: input.to_string(),
            kind,
            range: dummy_range(),
        }
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    #[test]
    fn test_applies_single_application_into_shape_map() {
        let apps = vec![app("y", "layer", "x", linear("3", "5"))];
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert!(errors.is_empty());
        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "3"])));
        assert_eq!(shapes.get("y"), Some(&shape(&["batch", "5"])));
    }

    #[test]
    fn test_applies_chained_applications_in_order() {
        let apps = vec![
            app("y", "l1", "x", linear("3", "5")),
            app("z", "l2", "y", linear("5", "7")),
        ];
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert!(errors.is_empty());
        assert_eq!(shapes.get("y"), Some(&shape(&["batch", "5"])));
        assert_eq!(shapes.get("z"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_missing_input_is_skipped_without_error() {
        let apps = vec![app("y", "layer", "missing", linear("3", "5"))];
        let mut shapes = HashMap::new();

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert!(errors.is_empty());
        assert!(!shapes.contains_key("y"));
    }

    #[test]
    fn test_mismatch_records_error_and_continues() {
        let apps = vec![
            app("bad", "bad_layer", "x", linear("4", "5")),
            app("good", "good_layer", "x", linear("3", "7")),
        ];
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "bad");
        assert!(errors[0].message.contains("bad_layer"));
        assert!(!shapes.contains_key("bad"));
        assert_eq!(shapes.get("good"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_order_matters_for_chains() {
        let apps = vec![
            app("z", "l2", "y", linear("5", "7")),
            app("y", "l1", "x", linear("3", "5")),
        ];
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert!(errors.is_empty());
        assert_eq!(shapes.get("y"), Some(&shape(&["batch", "5"])));
        assert!(!shapes.contains_key("z"));
    }

    #[test]
    fn test_later_assignment_overwrites_existing_output_shape() {
        let apps = vec![app("y", "layer", "x", linear("3", "5"))];
        let mut shapes = HashMap::from([
            ("x".to_string(), shape(&["batch", "3"])),
            ("y".to_string(), shape(&["old"])),
        ]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert!(errors.is_empty());
        assert_eq!(shapes.get("y"), Some(&shape(&["batch", "5"])));
    }

    #[test]
    fn test_collects_multiple_errors() {
        let apps = vec![
            app("a", "l1", "x", linear("4", "5")),
            app("b", "l2", "x", linear("6", "7")),
        ];
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].variable, "a");
        assert!(errors[0].message.contains("l1"));
        assert_eq!(errors[1].variable, "b");
        assert!(errors[1].message.contains("l2"));
        assert!(!shapes.contains_key("a"));
        assert!(!shapes.contains_key("b"));
    }

    #[test]
    fn test_scalar_error_does_not_stop_later_valid_application() {
        let apps = vec![
            app("bad", "l1", "scalar", linear("3", "5")),
            app("good", "l2", "x", linear("3", "7")),
        ];
        let mut shapes = HashMap::from([
            ("scalar".to_string(), Vec::new()),
            ("x".to_string(), shape(&["batch", "3"])),
        ]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "bad");
        assert!(errors[0].message.contains("scalar input"));
        assert_eq!(shapes.get("good"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_shape_error_points_to_output_variable_not_input_variable() {
        let apps = vec![app("projected", "projection", "x", linear("4", "5"))];
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "projected");
        assert!(errors[0].message.contains("projection"));
        assert!(errors[0].message.contains("x"));
    }

    #[test]
    fn test_missing_input_does_not_create_shape_error() {
        let apps = vec![app("y", "layer", "unknown", linear("3", "5"))];
        let mut shapes = HashMap::new();

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_empty_applications_preserve_existing_shapes() {
        let apps = Vec::new();
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert!(errors.is_empty());
        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "3"])));
    }

    #[test]
    fn test_failed_application_does_not_overwrite_existing_output_shape() {
        let apps = vec![app("y", "bad_layer", "x", linear("4", "5"))];
        let mut shapes = HashMap::from([
            ("x".to_string(), shape(&["batch", "3"])),
            ("y".to_string(), shape(&["old", "shape"])),
        ]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "y");
        assert_eq!(shapes.get("y"), Some(&shape(&["old", "shape"])));
    }

    #[test]
    fn test_dependent_application_is_skipped_after_failed_producer() {
        let apps = vec![
            app("bad", "bad_layer", "x", linear("4", "5")),
            app("z", "next_layer", "bad", linear("5", "7")),
        ];
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "bad");
        assert!(!shapes.contains_key("bad"));
        assert!(!shapes.contains_key("z"));
    }

    #[test]
    fn test_successful_application_after_unrelated_error_can_use_known_input() {
        let apps = vec![
            app("bad", "bad_layer", "x", linear("4", "5")),
            app("good", "good_layer", "x", linear("3", "7")),
        ];
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "bad");
        assert_eq!(shapes.get("good"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_multiple_errors_for_same_output_variable_are_preserved() {
        let apps = vec![
            app("y", "l1", "x", linear("4", "5")),
            app("y", "l2", "x", linear("6", "7")),
        ];
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].variable, "y");
        assert!(errors[0].message.contains("l1"));
        assert_eq!(errors[1].variable, "y");
        assert!(errors[1].message.contains("l2"));
        assert!(!shapes.contains_key("y"));
    }

    #[test]
    fn test_error_order_follows_application_order_with_successes_between() {
        let apps = vec![
            app("a", "l1", "x", linear("4", "5")),
            app("good", "good_layer", "x", linear("3", "9")),
            app("b", "l2", "x", linear("6", "7")),
        ];
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].variable, "a");
        assert!(errors[0].message.contains("l1"));
        assert_eq!(errors[1].variable, "b");
        assert!(errors[1].message.contains("l2"));
        assert_eq!(shapes.get("good"), Some(&shape(&["batch", "9"])));
    }

    #[test]
    fn test_shape_error_preserves_application_range() {
        let mut failed = app("y", "bad_layer", "x", linear("4", "5"));
        failed.range = Range {
            start_byte: 10,
            end_byte: 13,
            start_point: tree_sitter::Point::new(1, 2),
            end_point: tree_sitter::Point::new(1, 5),
        };
        let apps = vec![failed];
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].range.start_byte, 10);
        assert_eq!(errors[0].range.end_byte, 13);
        assert_eq!(errors[0].range.start_point, tree_sitter::Point::new(1, 2));
        assert_eq!(errors[0].range.end_point, tree_sitter::Point::new(1, 5));
    }

    #[test]
    fn test_multiple_shape_errors_preserve_their_own_ranges() {
        let mut first = app("a", "l1", "x", linear("4", "5"));
        first.range = Range {
            start_byte: 1,
            end_byte: 4,
            start_point: tree_sitter::Point::new(0, 1),
            end_point: tree_sitter::Point::new(0, 4),
        };
        let mut second = app("b", "l2", "x", linear("6", "7"));
        second.range = Range {
            start_byte: 10,
            end_byte: 13,
            start_point: tree_sitter::Point::new(1, 1),
            end_point: tree_sitter::Point::new(1, 4),
        };
        let apps = vec![first, second];
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].range.start_byte, 1);
        assert_eq!(errors[1].range.start_byte, 10);
    }
}

#[cfg(test)]
mod extract_layer_applications_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn linear(in_features: &str, out_features: &str) -> LayerKind {
        LayerKind::Linear {
            in_features: in_features.to_string(),
            out_features: out_features.to_string(),
        }
    }

    fn dummy_range() -> Range {
        Range {
            start_byte: 0,
            end_byte: 0,
            start_point: tree_sitter::Point::new(0, 0),
            end_point: tree_sitter::Point::new(0, 0),
        }
    }

    fn app(variable: &str, layer: &str, input: &str, kind: LayerKind) -> LayerApplication {
        LayerApplication {
            variable: variable.to_string(),
            layer: layer.to_string(),
            input: input.to_string(),
            kind,
            range: dummy_range(),
        }
    }

    #[test]
    fn test_extracts_simple_layer_application() {
        let code = "y = layer(x)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(applications, vec![app("y", "layer", "x", linear("3", "5"))]);
    }

    #[test]
    fn test_extracts_multiple_layer_applications() {
        let code = "y = l1(x)\nz = l2(y)";
        let tree = parse(code);
        let layers = HashMap::from([
            ("l1".to_string(), linear("3", "5")),
            ("l2".to_string(), linear("5", "7")),
        ]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(
            applications,
            vec![
                app("y", "l1", "x", linear("3", "5")),
                app("z", "l2", "y", linear("5", "7")),
            ]
        );
    }

    #[test]
    fn test_skips_unknown_layer_call() {
        let code = "y = layer(x)\nz = unknown(y)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(applications, vec![app("y", "layer", "x", linear("3", "5"))]);
    }

    #[test]
    fn test_skips_layer_call_without_arguments() {
        let code = "y = layer()";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert!(applications.is_empty());
    }

    #[test]
    fn test_skips_layer_call_with_keyword_only_input() {
        let code = "y = layer(x=x)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert!(applications.is_empty());
    }

    #[test]
    fn test_keeps_expression_input_text() {
        let code = "y = layer(x + residual)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(
            applications,
            vec![app("y", "layer", "x + residual", linear("3", "5"))]
        );
    }

    #[test]
    fn test_skips_attribute_layer_call_for_now() {
        let code = "y = model.layer(x)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert!(applications.is_empty());
    }

    #[test]
    fn test_uses_first_positional_arg_when_extra_args_are_present() {
        let code = "y = layer(x, other, flag=True)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(applications, vec![app("y", "layer", "x", linear("3", "5"))]);
    }

    #[test]
    fn test_preserves_application_order_while_skipping_unknown_calls() {
        let code = "a = l1(x)\nignored = unknown(a)\nb = l2(a)";
        let tree = parse(code);
        let layers = HashMap::from([
            ("l1".to_string(), linear("3", "5")),
            ("l2".to_string(), linear("5", "7")),
        ]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(
            applications,
            vec![
                app("a", "l1", "x", linear("3", "5")),
                app("b", "l2", "a", linear("5", "7")),
            ]
        );
    }

    #[test]
    fn test_layer_name_match_is_exact() {
        let code = "y = layer2(x)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert!(applications.is_empty());
    }

    #[test]
    fn test_extracts_layer_application_inside_function() {
        let code = "def f(x):\n    y = layer(x)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(applications, vec![app("y", "layer", "x", linear("3", "5"))]);
    }

    #[test]
    fn test_keeps_nested_call_input_text() {
        let code = "y = layer(preprocess(x))";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(
            applications,
            vec![app("y", "layer", "preprocess(x)", linear("3", "5"))]
        );
    }

    #[test]
    fn test_keeps_subscript_input_text() {
        let code = "y = layer(x[:, 0])";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(
            applications,
            vec![app("y", "layer", "x[:, 0]", linear("3", "5"))]
        );
    }

    #[test]
    fn test_keeps_parenthesized_input_text() {
        let code = "y = layer((x))";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(
            applications,
            vec![app("y", "layer", "(x)", linear("3", "5"))]
        );
    }

    #[test]
    fn test_duplicate_output_applications_are_kept_in_order() {
        let code = "y = l1(x)\ny = l2(y)";
        let tree = parse(code);
        let layers = HashMap::from([
            ("l1".to_string(), linear("3", "5")),
            ("l2".to_string(), linear("5", "7")),
        ]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(
            applications,
            vec![
                app("y", "l1", "x", linear("3", "5")),
                app("y", "l2", "y", linear("5", "7")),
            ]
        );
    }

    fn range_text<'a>(text: &'a str, range: &Range) -> &'a str {
        &text[range.start_byte..range.end_byte]
    }

    #[test]
    fn test_application_range_covers_arguments() {
        let code = "y = layer(x)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(range_text(code, &applications[0].range), "(x)");
    }

    #[test]
    fn test_application_range_covers_expression_arguments() {
        let code = "y = layer(x + residual)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(range_text(code, &applications[0].range), "(x + residual)");
    }

    #[test]
    fn test_multiple_application_ranges_follow_each_call() {
        let code = "a = l1(x)\nb = l2(a)";
        let tree = parse(code);
        let layers = HashMap::from([
            ("l1".to_string(), linear("3", "5")),
            ("l2".to_string(), linear("5", "7")),
        ]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(range_text(code, &applications[0].range), "(x)");
        assert_eq!(range_text(code, &applications[1].range), "(a)");
    }

    #[test]
    fn test_application_range_covers_all_arguments() {
        let code = "y = layer(x, other, flag=True)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(
            range_text(code, &applications[0].range),
            "(x, other, flag=True)"
        );
    }

    #[test]
    fn test_application_range_covers_nested_call_argument() {
        let code = "y = layer(preprocess(x))";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(range_text(code, &applications[0].range), "(preprocess(x))");
    }

    #[test]
    fn test_application_range_inside_function_includes_only_arguments() {
        let code = "def f(x):\n    y = layer(x)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(range_text(code, &applications[0].range), "(x)");
        assert_eq!(applications[0].range.start_point.row, 1);
    }

    #[test]
    fn test_application_range_covers_multiline_arguments() {
        let code = "y = layer(\n    x\n)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(range_text(code, &applications[0].range), "(\n    x\n)");
        assert_eq!(applications[0].range.start_point.row, 0);
        assert_eq!(applications[0].range.end_point.row, 2);
    }

    #[test]
    fn test_duplicate_output_application_ranges_are_distinct() {
        let code = "y = l1(x)\ny = l2(y)";
        let tree = parse(code);
        let layers = HashMap::from([
            ("l1".to_string(), linear("3", "5")),
            ("l2".to_string(), linear("5", "7")),
        ]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(range_text(code, &applications[0].range), "(x)");
        assert_eq!(range_text(code, &applications[1].range), "(y)");
        assert_ne!(
            applications[0].range.start_byte,
            applications[1].range.start_byte
        );
    }

    #[test]
    fn test_range_after_skipped_unknown_call_belongs_to_known_call() {
        let code = "ignored = unknown(x)\ny = layer(x)";
        let tree = parse(code);
        let layers = HashMap::from([("layer".to_string(), linear("3", "5"))]);

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert_eq!(applications.len(), 1);
        assert_eq!(range_text(code, &applications[0].range), "(x)");
        assert_eq!(applications[0].range.start_point.row, 1);
    }

    #[test]
    fn test_empty_layer_map_returns_empty_applications() {
        let code = "y = layer(x)";
        let tree = parse(code);
        let layers = HashMap::new();

        let applications = extract_layer_applications(tree.root_node(), code, &layers).unwrap();

        assert!(applications.is_empty());
    }
}

#[cfg(test)]
mod extract_layer_assignments_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn write_equinox_linear(tmp: &tempfile::TempDir) {
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(
            tmp.path().join("equinox/nn/__init__.py"),
            "from ._linear import Linear",
        )
        .unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_linear.py"),
            "class Linear:\n    def __init__(self, in_features, out_features, use_bias=True): pass",
        )
        .unwrap();
    }

    #[test]
    fn test_extracts_single_linear_assignment() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Linear {
                in_features: "3".to_string(),
                out_features: "5".to_string()
            })
        );
    }

    #[test]
    fn test_extracts_multiple_linear_assignments() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code =
            "import equinox as eqx\na = eqx.nn.Linear(3, 5)\nb = eqx.nn.Linear(features, hidden)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(layers.len(), 2);
        assert_eq!(
            layers.get("a"),
            Some(&LayerKind::Linear {
                in_features: "3".to_string(),
                out_features: "5".to_string()
            })
        );
        assert_eq!(
            layers.get("b"),
            Some(&LayerKind::Linear {
                in_features: "features".to_string(),
                out_features: "hidden".to_string()
            })
        );
    }

    #[test]
    fn test_supports_from_import_alias_assignment() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code =
            "from equinox.nn import Linear as Lin\nlayer = Lin(in_features=3, out_features=5)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Linear {
                in_features: "3".to_string(),
                out_features: "5".to_string()
            })
        );
    }

    #[test]
    fn test_skips_non_layer_calls() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        fs::write(tmp.path().join("helpers.py"), "def make_layer(): pass").unwrap();
        let code = "import equinox as eqx\nimport helpers\nlayer = eqx.nn.Linear(3, 5)\nother = helpers.make_layer()";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(layers.len(), 1);
        assert!(layers.contains_key("layer"));
        assert!(!layers.contains_key("other"));
    }

    #[test]
    fn test_skips_missing_implementation() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(layers.is_empty());
    }

    #[test]
    fn test_supports_reversed_keyword_order() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(out_features=5, in_features=3)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Linear {
                in_features: "3".to_string(),
                out_features: "5".to_string()
            })
        );
    }

    #[test]
    fn test_supports_extra_constructor_kwargs() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5, use_bias=False, key=key)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Linear {
                in_features: "3".to_string(),
                out_features: "5".to_string()
            })
        );
    }

    #[test]
    fn test_keyword_overrides_positional_constructor_binding() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 4, out_features=5)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Linear {
                in_features: "3".to_string(),
                out_features: "5".to_string()
            })
        );
    }

    #[test]
    fn test_extracts_layer_assignment_inside_function() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef make():\n    layer = eqx.nn.Linear(3, 5)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Linear {
                in_features: "3".to_string(),
                out_features: "5".to_string()
            })
        );
    }

    #[test]
    fn test_skips_layer_assignment_when_required_feature_is_missing() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(in_features=3)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(layers.is_empty());
    }

    #[test]
    fn test_duplicate_layer_assignment_last_wins() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code =
            "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)\nlayer = eqx.nn.Linear(7, 11)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Linear {
                in_features: "7".to_string(),
                out_features: "11".to_string()
            })
        );
    }
}

#[cfg(test)]
mod analyze_layer_shapes_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn write_equinox_linear(tmp: &tempfile::TempDir) {
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(
            tmp.path().join("equinox/nn/__init__.py"),
            "from ._linear import Linear",
        )
        .unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_linear.py"),
            "class Linear:\n    def __init__(self, in_features, out_features): pass",
        )
        .unwrap();
    }

    #[test]
    fn test_analyzes_single_layer_shape_success() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = eqx.nn.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(analysis.shapes.get("x"), Some(&shape(&["batch", "3"])));
        assert_eq!(analysis.shapes.get("y"), Some(&shape(&["batch", "5"])));
        assert!(analysis.layers.contains_key("layer"));
        assert_eq!(analysis.applications.len(), 1);
    }

    #[test]
    fn test_analyzes_chained_layer_shapes() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(3, 5)\n    l2 = eqx.nn.Linear(5, 7)\n    y = l1(x)\n    z = l2(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(analysis.shapes.get("y"), Some(&shape(&["batch", "5"])));
        assert_eq!(analysis.shapes.get("z"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_reports_layer_shape_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 4\"]):\n    layer = eqx.nn.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert!(
            analysis.errors[0]
                .message
                .contains("expected input last dim 3")
        );
        assert_eq!(analysis.shapes.get("x"), Some(&shape(&["batch", "4"])));
        assert!(!analysis.shapes.contains_key("y"));
    }

    #[test]
    fn test_missing_input_shape_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code =
            "import equinox as eqx\ndef f(x):\n    layer = eqx.nn.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(analysis.shapes.is_empty());
        assert_eq!(analysis.layers.len(), 1);
        assert_eq!(analysis.applications.len(), 1);
    }

    #[test]
    fn test_supports_from_import_alias() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "from equinox.nn import Linear as Lin\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = Lin(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(analysis.shapes.get("y"), Some(&shape(&["batch", "5"])));
        assert!(analysis.layers.contains_key("layer"));
    }

    #[test]
    fn test_non_layer_calls_are_ignored_in_analysis() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        fs::write(tmp.path().join("helpers.py"), "def transform(x): pass").unwrap();
        let code = "import equinox as eqx\nimport helpers\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = eqx.nn.Linear(3, 5)\n    a = helpers.transform(x)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(analysis.layers.len(), 1);
        assert_eq!(analysis.applications.len(), 1);
        assert_eq!(analysis.shapes.get("y"), Some(&shape(&["batch", "5"])));
        assert!(!analysis.shapes.contains_key("a"));
    }

    #[test]
    fn test_missing_layer_implementation_keeps_only_annotation_shapes() {
        let tmp = tempfile::tempdir().unwrap();
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = eqx.nn.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(analysis.layers.is_empty());
        assert!(analysis.applications.is_empty());
        assert_eq!(analysis.shapes.get("x"), Some(&shape(&["batch", "3"])));
        assert!(!analysis.shapes.contains_key("y"));
    }

    #[test]
    fn test_analysis_continues_after_one_application_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"], a: Float[Array, \"batch 4\"]):\n    bad_layer = eqx.nn.Linear(3, 5)\n    good_layer = eqx.nn.Linear(4, 6)\n    bad = bad_layer(a)\n    good = good_layer(a)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "bad");
        assert!(analysis.errors[0].message.contains("bad_layer"));
        assert!(!analysis.shapes.contains_key("bad"));
        assert_eq!(analysis.shapes.get("good"), Some(&shape(&["batch", "6"])));
    }

    #[test]
    fn test_analysis_collects_multiple_structured_errors() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(4, 5)\n    l2 = eqx.nn.Linear(6, 7)\n    a = l1(x)\n    b = l2(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 2);
        assert_eq!(analysis.errors[0].variable, "a");
        assert!(analysis.errors[0].message.contains("l1"));
        assert_eq!(analysis.errors[1].variable, "b");
        assert!(analysis.errors[1].message.contains("l2"));
        assert!(!analysis.shapes.contains_key("a"));
        assert!(!analysis.shapes.contains_key("b"));
    }

    #[test]
    fn test_analysis_error_order_with_success_between_errors() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(4, 5)\n    good_layer = eqx.nn.Linear(3, 9)\n    l2 = eqx.nn.Linear(6, 7)\n    a = l1(x)\n    good = good_layer(x)\n    b = l2(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 2);
        assert_eq!(analysis.errors[0].variable, "a");
        assert_eq!(analysis.errors[1].variable, "b");
        assert_eq!(analysis.shapes.get("good"), Some(&shape(&["batch", "9"])));
    }

    #[test]
    fn test_analysis_failed_assignment_does_not_overwrite_existing_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(y: Float[Array, \"old shape\"], x: Float[Array, \"batch 3\"]):\n    bad_layer = eqx.nn.Linear(4, 5)\n    y = bad_layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.errors[0].variable, "y");
        assert_eq!(analysis.shapes.get("y"), Some(&shape(&["old", "shape"])));
    }

    #[test]
    fn test_analysis_error_range_covers_failing_call_arguments() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    bad_layer = eqx.nn.Linear(4, 5)\n    y = bad_layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 1);
        let range = &analysis.errors[0].range;
        assert_eq!(&code[range.start_byte..range.end_byte], "(x)");
    }

    #[test]
    fn test_analysis_multiple_error_ranges_cover_each_call() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(4, 5)\n    l2 = eqx.nn.Linear(6, 7)\n    a = l1(x)\n    b = l2(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert_eq!(analysis.errors.len(), 2);
        assert_eq!(
            &code[analysis.errors[0].range.start_byte..analysis.errors[0].range.end_byte],
            "(x)"
        );
        assert_eq!(
            &code[analysis.errors[1].range.start_byte..analysis.errors[1].range.end_byte],
            "(x)"
        );
        assert_ne!(
            analysis.errors[0].range.start_byte,
            analysis.errors[1].range.start_byte
        );
    }

    #[test]
    fn test_scalar_annotated_input_reports_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"\"]):\n    layer = eqx.nn.Linear(3, 5)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(!analysis.shapes.contains_key("x"));
        assert!(!analysis.shapes.contains_key("y"));
    }

    #[test]
    fn test_keyword_constructor_analysis() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch features\"]):\n    layer = eqx.nn.Linear(out_features=hidden, in_features=features)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(analysis.shapes.get("y"), Some(&shape(&["batch", "hidden"])));
    }

    #[test]
    fn test_duplicate_layer_assignment_affects_later_application() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 7\"]):\n    layer = eqx.nn.Linear(3, 5)\n    layer = eqx.nn.Linear(7, 11)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(analysis.shapes.get("y"), Some(&shape(&["batch", "11"])));
    }

    #[test]
    fn test_symbolic_dims_propagate_through_analysis() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch features\"]):\n    layer = eqx.nn.Linear(features, hidden)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(analysis.shapes.get("y"), Some(&shape(&["batch", "hidden"])));
    }

    #[test]
    fn test_application_source_order_is_respected() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(3, 5)\n    l2 = eqx.nn.Linear(5, 7)\n    z = l2(y)\n    y = l1(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(analysis.shapes.get("y"), Some(&shape(&["batch", "5"])));
        assert!(!analysis.shapes.contains_key("z"));
    }

    #[test]
    fn test_missing_input_does_not_block_later_valid_application() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    layer = eqx.nn.Linear(3, 5)\n    missing_out = layer(missing)\n    y = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(!analysis.shapes.contains_key("missing_out"));
        assert_eq!(analysis.shapes.get("y"), Some(&shape(&["batch", "5"])));
    }

    #[test]
    fn test_output_reassignment_shape_last_wins() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\ndef f(x: Float[Array, \"batch 3\"]):\n    l1 = eqx.nn.Linear(3, 5)\n    l2 = eqx.nn.Linear(5, 7)\n    y = l1(x)\n    y = l2(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert_eq!(analysis.shapes.get("y"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_file_with_only_annotations_returns_initial_shapes() {
        let tmp = tempfile::tempdir().unwrap();
        let code =
            "def f(x: Float[Array, \"batch features\"], y: Float[Array, \"batch\"]):\n    pass";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let analysis = analyze_layer_shapes(tree.root_node(), code, &roots, read, 5).unwrap();

        assert!(analysis.errors.is_empty());
        assert!(analysis.layers.is_empty());
        assert!(analysis.applications.is_empty());
        assert_eq!(
            analysis.shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
        assert_eq!(analysis.shapes.get("y"), Some(&shape(&["batch"])));
    }
}

#[cfg(test)]
mod end_to_end_layer_shape_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn write_equinox_linear(tmp: &tempfile::TempDir) {
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(
            tmp.path().join("equinox/nn/__init__.py"),
            "from ._linear import Linear",
        )
        .unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_linear.py"),
            "class Linear:\n    def __init__(self, in_features, out_features): pass",
        )
        .unwrap();
    }

    #[test]
    fn test_extracts_layers_and_propagates_single_application() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)\ny = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();
        let apps = extract_layer_applications(tree.root_node(), code, &layers).unwrap();
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert!(errors.is_empty());
        assert_eq!(shapes.get("y"), Some(&shape(&["batch", "5"])));
    }

    #[test]
    fn test_extracts_layers_and_propagates_chain() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nl1 = eqx.nn.Linear(3, 5)\nl2 = eqx.nn.Linear(5, 7)\ny = l1(x)\nz = l2(y)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();
        let apps = extract_layer_applications(tree.root_node(), code, &layers).unwrap();
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert!(errors.is_empty());
        assert_eq!(shapes.get("y"), Some(&shape(&["batch", "5"])));
        assert_eq!(shapes.get("z"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_reports_mismatch_in_end_to_end_pipeline() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)\ny = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();
        let apps = extract_layer_applications(tree.root_node(), code, &layers).unwrap();
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "4"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "y");
        assert!(errors[0].message.contains("expected input last dim 3"));
        assert!(!shapes.contains_key("y"));
    }

    #[test]
    fn test_unknown_input_in_end_to_end_pipeline_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)\ny = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();
        let apps = extract_layer_applications(tree.root_node(), code, &layers).unwrap();
        let mut shapes = HashMap::new();

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert!(errors.is_empty());
        assert!(!shapes.contains_key("y"));
    }

    #[test]
    fn test_in_place_layer_application_uses_old_input_shape_then_overwrites() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(&tmp);
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)\nx = layer(x)";
        let tree = parse(code);
        let roots = vec![tmp.path().to_path_buf()];

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, read, 5).unwrap();
        let apps = extract_layer_applications(tree.root_node(), code, &layers).unwrap();
        let mut shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let errors = apply_layer_applications(&apps, &mut shapes);

        assert!(errors.is_empty());
        assert_eq!(shapes.get("x"), Some(&shape(&["batch", "5"])));
    }
}

#[cfg(test)]
mod classify_layer_call_tests {
    use super::*;

    fn call(
        module_parts: &[&str],
        owner: Option<&str>,
        name: &str,
        bindings: &[(&str, &str)],
    ) -> ResolvedCallSignature {
        ResolvedCallSignature {
            implementation: ResolvedImplementation {
                target: ResolvedModuleTarget {
                    dots: 0,
                    module_parts: module_parts.iter().map(|p| p.to_string()).collect(),
                    file_path: PathBuf::from("unused.py"),
                    symbol_parts: vec![owner.unwrap_or(name).to_string()],
                },
                symbol: owner.map(|owner| PythonSymbol::Class {
                    name: owner.to_string(),
                }),
            },
            signature: PythonCallableSignature {
                owner: owner.map(|owner| owner.to_string()),
                name: name.to_string(),
                params: Vec::new(),
            },
            arguments: Vec::new(),
            bindings: bindings
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
        }
    }

    #[test]
    fn test_classifies_equinox_linear() {
        let call = call(
            &["equinox", "nn", "_linear"],
            Some("Linear"),
            "__init__",
            &[("in_features", "3"), ("out_features", "5")],
        );

        assert_eq!(
            classify_layer_call(&call),
            Some(LayerKind::Linear {
                in_features: "3".to_string(),
                out_features: "5".to_string()
            })
        );
    }

    #[test]
    fn test_classifies_symbolic_equinox_linear() {
        let call = call(
            &["equinox", "nn", "_linear"],
            Some("Linear"),
            "__init__",
            &[("in_features", "features"), ("out_features", "hidden")],
        );

        assert_eq!(
            classify_layer_call(&call),
            Some(LayerKind::Linear {
                in_features: "features".to_string(),
                out_features: "hidden".to_string()
            })
        );
    }

    #[test]
    fn test_missing_in_features_returns_none() {
        let call = call(
            &["equinox", "nn", "_linear"],
            Some("Linear"),
            "__init__",
            &[("out_features", "5")],
        );

        assert_eq!(classify_layer_call(&call), None);
    }

    #[test]
    fn test_missing_out_features_returns_none() {
        let call = call(
            &["equinox", "nn", "_linear"],
            Some("Linear"),
            "__init__",
            &[("in_features", "3")],
        );

        assert_eq!(classify_layer_call(&call), None);
    }

    #[test]
    fn test_non_equinox_linear_returns_none() {
        let call = call(
            &["my_project", "layers"],
            Some("Linear"),
            "__init__",
            &[("in_features", "3"), ("out_features", "5")],
        );

        assert_eq!(classify_layer_call(&call), None);
    }

    #[test]
    fn test_non_constructor_returns_none() {
        let call = call(
            &["equinox", "nn", "_linear"],
            Some("Linear"),
            "forward",
            &[("in_features", "3"), ("out_features", "5")],
        );

        assert_eq!(classify_layer_call(&call), None);
    }

    #[test]
    fn test_nested_equinox_nn_module_is_accepted() {
        let call = call(
            &["equinox", "nn", "layers", "linear"],
            Some("Linear"),
            "__init__",
            &[("in_features", "3"), ("out_features", "5")],
        );

        assert_eq!(
            classify_layer_call(&call),
            Some(LayerKind::Linear {
                in_features: "3".to_string(),
                out_features: "5".to_string()
            })
        );
    }

    #[test]
    fn test_equinox_not_nn_returns_none() {
        let call = call(
            &["equinox", "experimental"],
            Some("Linear"),
            "__init__",
            &[("in_features", "3"), ("out_features", "5")],
        );

        assert_eq!(classify_layer_call(&call), None);
    }

    #[test]
    fn test_wrong_owner_returns_none() {
        let call = call(
            &["equinox", "nn"],
            Some("Dense"),
            "__init__",
            &[("in_features", "3"), ("out_features", "5")],
        );

        assert_eq!(classify_layer_call(&call), None);
    }

    #[test]
    fn test_function_call_returns_none() {
        let call = call(&["jax", "numpy"], None, "concatenate", &[("arrays", "xs")]);

        assert_eq!(classify_layer_call(&call), None);
    }
}

#[cfg(test)]
mod resolve_call_signature_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    fn write_equinox_linear(tmp: &tempfile::TempDir, init_source: &str) {
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(
            tmp.path().join("equinox/nn/__init__.py"),
            "from ._linear import Linear",
        )
        .unwrap();
        fs::write(tmp.path().join("equinox/nn/_linear.py"), init_source).unwrap();
    }

    #[test]
    fn test_resolves_call_signature_through_reexport() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(
            &tmp,
            "class Linear:\n    def __init__(self, in_features, out_features, use_bias=True): pass",
        );

        let source = "import equinox as eqx\nlayer = eqx.nn.Linear(3, out_features=5)";
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5)
            .unwrap()
            .unwrap();

        assert_eq!(found.signature.owner, Some("Linear".to_string()));
        assert_eq!(found.signature.name, "__init__");
        assert_eq!(found.bindings.get("in_features"), Some(&"3".to_string()));
        assert_eq!(found.bindings.get("out_features"), Some(&"5".to_string()));
        assert_eq!(found.bindings.get("self"), None);
    }

    #[test]
    fn test_classifies_real_resolved_equinox_linear_call() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(
            &tmp,
            "class Linear:\n    def __init__(self, in_features, out_features, use_bias=True): pass",
        );

        let source = "import equinox as eqx\nlayer = eqx.nn.Linear(features, hidden)";
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5)
            .unwrap()
            .unwrap();

        assert_eq!(
            classify_layer_call(&found),
            Some(LayerKind::Linear {
                in_features: "features".to_string(),
                out_features: "hidden".to_string()
            })
        );
    }

    #[test]
    fn test_resolves_from_imported_linear_alias() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(
            &tmp,
            "class Linear:\n    def __init__(self, in_features, out_features): pass",
        );

        let source = "from equinox.nn import Linear as Lin\nlayer = Lin(3, 5)";
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5)
            .unwrap()
            .unwrap();

        assert_eq!(found.bindings.get("in_features"), Some(&"3".to_string()));
        assert_eq!(found.bindings.get("out_features"), Some(&"5".to_string()));
        assert_eq!(
            classify_layer_call(&found),
            Some(LayerKind::Linear {
                in_features: "3".to_string(),
                out_features: "5".to_string()
            })
        );
    }

    #[test]
    fn test_classmethod_cls_param_is_skipped_for_bindings() {
        let tmp = tempfile::tempdir().unwrap();
        write_equinox_linear(
            &tmp,
            "class Linear:\n    def __init__(cls, in_features, out_features): pass",
        );

        let source = "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)";
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5)
            .unwrap()
            .unwrap();

        assert_eq!(found.bindings.get("in_features"), Some(&"3".to_string()));
        assert_eq!(found.bindings.get("out_features"), Some(&"5".to_string()));
        assert_eq!(found.bindings.get("cls"), None);
    }

    #[test]
    fn test_resolves_top_level_function_call_signature() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("jax/numpy")).unwrap();
        fs::write(
            tmp.path().join("jax/numpy/__init__.py"),
            "def concatenate(arrays, axis=0): pass",
        )
        .unwrap();

        let source = "import jax.numpy as jnp\nout = jnp.concatenate(xs, axis=1)";
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let found = resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5)
            .unwrap()
            .unwrap();

        assert_eq!(found.signature.owner, None);
        assert_eq!(found.signature.name, "concatenate");
        assert_eq!(found.bindings.get("arrays"), Some(&"xs".to_string()));
        assert_eq!(found.bindings.get("axis"), Some(&"1".to_string()));
    }

    #[test]
    fn test_returns_none_when_implementation_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let source = "import missing\nx = missing.Foo()";
        let tree = parse(source);
        let import_map = build_import_map(tree.root_node(), source).unwrap();
        let calls = extract_calls(tree.root_node(), source).unwrap();
        let roots = vec![tmp.path().to_path_buf()];

        let found =
            resolve_call_signature(&calls[0], source, &import_map, &roots, read, 5).unwrap();

        assert_eq!(found, None);
    }
}

#[cfg(test)]
mod call_argument_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse_args(code: &str) -> (tree_sitter::Tree, String) {
        let wrapped = format!("x = f({code})");
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        let tree = parser.parse(&wrapped, None).unwrap();
        (tree, wrapped)
    }

    fn args_node<'tree>(tree: &'tree tree_sitter::Tree, text: &str) -> Node<'tree> {
        let root = tree.root_node();
        let call = root
            .descendant_for_byte_range(text.find("f(").unwrap(), text.len())
            .unwrap();
        call.child_by_field_name("arguments").unwrap()
    }

    fn sig(owner: Option<&str>, params: &[&str]) -> PythonCallableSignature {
        PythonCallableSignature {
            owner: owner.map(|s| s.to_string()),
            name: "f".to_string(),
            params: params.iter().map(|p| p.to_string()).collect(),
        }
    }

    #[test]
    fn test_extracts_positional_arguments() {
        let (tree, text) = parse_args("x, y + 1");
        let args = extract_call_arguments(args_node(&tree, &text), &text).unwrap();

        assert_eq!(
            args,
            vec![
                CallArgument::Positional {
                    value: "x".to_string()
                },
                CallArgument::Positional {
                    value: "y + 1".to_string()
                }
            ]
        );
    }

    #[test]
    fn test_extracts_keyword_arguments() {
        let (tree, text) = parse_args("in_features=3, out_features=features");
        let args = extract_call_arguments(args_node(&tree, &text), &text).unwrap();

        assert_eq!(
            args,
            vec![
                CallArgument::Keyword {
                    name: "in_features".to_string(),
                    value: "3".to_string()
                },
                CallArgument::Keyword {
                    name: "out_features".to_string(),
                    value: "features".to_string()
                }
            ]
        );
    }

    #[test]
    fn test_binds_function_positional_and_keyword_arguments() {
        let signature = sig(None, &["x", "axis", "keepdims"]);
        let args = vec![
            CallArgument::Positional {
                value: "arr".to_string(),
            },
            CallArgument::Keyword {
                name: "keepdims".to_string(),
                value: "True".to_string(),
            },
        ];

        let bindings = bind_call_arguments(&signature, &args);

        assert_eq!(bindings.get("x"), Some(&"arr".to_string()));
        assert_eq!(bindings.get("keepdims"), Some(&"True".to_string()));
        assert_eq!(bindings.get("axis"), None);
    }

    #[test]
    fn test_binds_class_constructor_skipping_self() {
        let signature = sig(Some("Linear"), &["self", "in_features", "out_features"]);
        let args = vec![
            CallArgument::Positional {
                value: "3".to_string(),
            },
            CallArgument::Keyword {
                name: "out_features".to_string(),
                value: "5".to_string(),
            },
        ];

        let bindings = bind_call_arguments(&signature, &args);

        assert_eq!(bindings.get("in_features"), Some(&"3".to_string()));
        assert_eq!(bindings.get("out_features"), Some(&"5".to_string()));
        assert_eq!(bindings.get("self"), None);
    }

    #[test]
    fn test_keyword_overrides_positional_binding() {
        let signature = sig(None, &["x", "axis"]);
        let args = vec![
            CallArgument::Positional {
                value: "arr".to_string(),
            },
            CallArgument::Positional {
                value: "0".to_string(),
            },
            CallArgument::Keyword {
                name: "axis".to_string(),
                value: "1".to_string(),
            },
        ];

        let bindings = bind_call_arguments(&signature, &args);

        assert_eq!(bindings.get("axis"), Some(&"1".to_string()));
    }

    #[test]
    fn test_extra_positional_arguments_are_ignored() {
        let signature = sig(None, &["x"]);
        let args = vec![
            CallArgument::Positional {
                value: "arr".to_string(),
            },
            CallArgument::Positional {
                value: "extra".to_string(),
            },
        ];

        let bindings = bind_call_arguments(&signature, &args);

        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings.get("x"), Some(&"arr".to_string()));
    }
}

#[cfg(test)]
mod extract_callable_signature_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn signature(owner: Option<&str>, name: &str, params: &[&str]) -> PythonCallableSignature {
        PythonCallableSignature {
            owner: owner.map(|s| s.to_string()),
            name: name.to_string(),
            params: params.iter().map(|p| p.to_string()).collect(),
        }
    }

    #[test]
    fn test_extracts_top_level_function_params() {
        let code = "def concatenate(arrays, axis=0): pass";
        let tree = parse(code);

        let found = extract_callable_signature(
            tree.root_node(),
            code,
            &PythonSymbol::Function {
                name: "concatenate".to_string(),
            },
        )
        .unwrap();

        assert_eq!(
            found,
            Some(signature(None, "concatenate", &["arrays", "axis"]))
        );
    }

    #[test]
    fn test_extracts_annotated_function_params() {
        let code = "def f(x: Array, y: int = 1) -> Array: pass";
        let tree = parse(code);

        let found = extract_callable_signature(
            tree.root_node(),
            code,
            &PythonSymbol::Function {
                name: "f".to_string(),
            },
        )
        .unwrap();

        assert_eq!(found, Some(signature(None, "f", &["x", "y"])));
    }

    #[test]
    fn test_extracts_varargs_and_kwargs() {
        let code = "def f(x, *args, **kwargs): pass";
        let tree = parse(code);

        let found = extract_callable_signature(
            tree.root_node(),
            code,
            &PythonSymbol::Function {
                name: "f".to_string(),
            },
        )
        .unwrap();

        assert_eq!(found, Some(signature(None, "f", &["x", "args", "kwargs"])));
    }

    #[test]
    fn test_extracts_class_init_params() {
        let code =
            "class Linear:\n    def __init__(self, in_features, out_features, use_bias=True): pass";
        let tree = parse(code);

        let found = extract_callable_signature(
            tree.root_node(),
            code,
            &PythonSymbol::Class {
                name: "Linear".to_string(),
            },
        )
        .unwrap();

        assert_eq!(
            found,
            Some(signature(
                Some("Linear"),
                "__init__",
                &["self", "in_features", "out_features", "use_bias"]
            ))
        );
    }

    #[test]
    fn test_ignores_nested_init_inside_method() {
        let code = "class Linear:\n    def outer(self):\n        def __init__(x): pass";
        let tree = parse(code);

        let found = extract_callable_signature(
            tree.root_node(),
            code,
            &PythonSymbol::Class {
                name: "Linear".to_string(),
            },
        )
        .unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_extracts_keyword_only_params() {
        let code = "def f(x, *, axis=0, keepdims=False): pass";
        let tree = parse(code);

        let found = extract_callable_signature(
            tree.root_node(),
            code,
            &PythonSymbol::Function {
                name: "f".to_string(),
            },
        )
        .unwrap();

        assert_eq!(
            found,
            Some(signature(None, "f", &["x", "axis", "keepdims"]))
        );
    }

    #[test]
    fn test_import_symbol_returns_none() {
        let code = "from ._linear import Linear";
        let tree = parse(code);

        let found = extract_callable_signature(
            tree.root_node(),
            code,
            &PythonSymbol::Import {
                name: "Linear".to_string(),
                path: ImportPath {
                    dots: 1,
                    module: vec!["_linear".to_string()],
                    name: "Linear".to_string(),
                },
            },
        )
        .unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_function_returns_none() {
        let code = "def other(): pass";
        let tree = parse(code);

        let found = extract_callable_signature(
            tree.root_node(),
            code,
            &PythonSymbol::Function {
                name: "f".to_string(),
            },
        )
        .unwrap();

        assert_eq!(found, None);
    }
}

#[cfg(test)]
mod resolve_implementation_tests {
    use super::*;
    use std::fs;

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn target(parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots: 0,
            parts: self::parts(parts),
        }
    }

    fn implementation(
        module_parts: &[&str],
        file_path: PathBuf,
        symbol_parts: &[&str],
        symbol: Option<PythonSymbol>,
    ) -> ResolvedImplementation {
        ResolvedImplementation {
            target: ResolvedModuleTarget {
                dots: 0,
                module_parts: parts(module_parts),
                file_path,
                symbol_parts: parts(symbol_parts),
            },
            symbol,
        }
    }

    fn read(path: &PathBuf) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    #[test]
    fn test_resolves_direct_class() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("pkg")).unwrap();
        fs::write(tmp.path().join("pkg/linear.py"), "class Linear: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found =
            resolve_implementation(target(&["pkg", "linear", "Linear"]), &roots, read, 5).unwrap();

        assert_eq!(
            found,
            Some(implementation(
                &["pkg", "linear"],
                tmp.path().join("pkg/linear.py"),
                &["Linear"],
                Some(PythonSymbol::Class {
                    name: "Linear".to_string()
                })
            ))
        );
    }

    #[test]
    fn test_resolves_direct_function() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("jax/numpy")).unwrap();
        fs::write(
            tmp.path().join("jax/numpy/api.py"),
            "def concatenate(): pass",
        )
        .unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(
            target(&["jax", "numpy", "api", "concatenate"]),
            &roots,
            read,
            5,
        )
        .unwrap();

        assert_eq!(
            found,
            Some(implementation(
                &["jax", "numpy", "api"],
                tmp.path().join("jax/numpy/api.py"),
                &["concatenate"],
                Some(PythonSymbol::Function {
                    name: "concatenate".to_string()
                })
            ))
        );
    }

    #[test]
    fn test_resolves_module_itself_when_no_symbol_parts() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("pkg")).unwrap();
        fs::write(tmp.path().join("pkg/mod.py"), "VALUE = 1").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["pkg", "mod"]), &roots, read, 5).unwrap();

        assert_eq!(
            found,
            Some(implementation(
                &["pkg", "mod"],
                tmp.path().join("pkg/mod.py"),
                &[],
                None
            ))
        );
    }

    #[test]
    fn test_follows_reexport_then_resolves_class() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(
            tmp.path().join("equinox/nn/__init__.py"),
            "from ._linear import Linear",
        )
        .unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_linear.py"),
            "class Linear: pass",
        )
        .unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found =
            resolve_implementation(target(&["equinox", "nn", "Linear"]), &roots, read, 5).unwrap();

        assert_eq!(
            found,
            Some(implementation(
                &["equinox", "nn", "_linear"],
                tmp.path().join("equinox/nn/_linear.py"),
                &["Linear"],
                Some(PythonSymbol::Class {
                    name: "Linear".to_string()
                })
            ))
        );
    }

    #[test]
    fn test_missing_module_returns_none() {
        let tmp = tempfile::tempdir().unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found =
            resolve_implementation(target(&["missing", "Linear"]), &roots, read, 5).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_read_failure_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("foo.py"), "class Foo: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["foo", "Foo"]), &roots, |_| None, 5).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_symbol_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("foo.py"), "class Bar: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["foo", "Foo"]), &roots, read, 5).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_max_depth_zero_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("foo.py"), "class Foo: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["foo", "Foo"]), &roots, read, 0).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_reexport_preserves_remaining_symbol_parts_across_loop() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("realpkg")).unwrap();
        fs::write(tmp.path().join("aliaspkg.py"), "from realpkg import layers").unwrap();
        fs::write(tmp.path().join("realpkg/layers.py"), "class Linear: pass").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found =
            resolve_implementation(target(&["aliaspkg", "layers", "Linear"]), &roots, read, 10)
                .unwrap();

        assert_eq!(
            found,
            Some(implementation(
                &["realpkg", "layers"],
                tmp.path().join("realpkg/layers.py"),
                &["Linear"],
                Some(PythonSymbol::Class {
                    name: "Linear".to_string()
                })
            ))
        );
    }

    #[test]
    fn test_cycle_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("pkg")).unwrap();
        fs::write(tmp.path().join("pkg/__init__.py"), "from .a import X").unwrap();
        fs::write(tmp.path().join("pkg/a.py"), "from . import X").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_implementation(target(&["pkg", "X"]), &roots, read, 10).unwrap();

        assert_eq!(found, None);
    }
}

#[cfg(test)]
mod resolve_reexport_once_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn resolved(module_parts: &[&str], symbol_parts: &[&str]) -> ResolvedModuleTarget {
        ResolvedModuleTarget {
            dots: 0,
            module_parts: parts(module_parts),
            file_path: PathBuf::from("unused.py"),
            symbol_parts: parts(symbol_parts),
        }
    }

    fn target(parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots: 0,
            parts: self::parts(parts),
        }
    }

    #[test]
    fn test_follows_relative_reexport() {
        let code = "from ._linear import Linear";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &["Linear"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, Some(target(&["equinox", "nn", "_linear", "Linear"])));
    }

    #[test]
    fn test_follows_relative_package_reexport() {
        let code = "from . import layers";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &["layers", "Linear"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, Some(target(&["equinox", "nn", "layers", "Linear"])));
    }

    #[test]
    fn test_preserves_remaining_symbol_parts_after_reexport() {
        let code = "from . import layers";
        let tree = parse(code);
        let current = resolved(&["pkg"], &["layers", "nn", "Linear"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, Some(target(&["pkg", "layers", "nn", "Linear"])));
    }

    #[test]
    fn test_follows_absolute_reexport() {
        let code = "from jax.numpy import concatenate";
        let tree = parse(code);
        let current = resolved(&["my", "pkg"], &["concatenate"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, Some(target(&["jax", "numpy", "concatenate"])));
    }

    #[test]
    fn test_class_returns_none() {
        let code = "class Linear: pass";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &["Linear"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_function_returns_none() {
        let code = "def linear(): pass";
        let tree = parse(code);
        let current = resolved(&["pkg"], &["linear"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_symbol_returns_none() {
        let code = "from ._linear import Other";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &["Linear"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_empty_symbol_parts_returns_none() {
        let code = "from ._linear import Linear";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &[]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_too_many_relative_dots_returns_none() {
        let code = "from ..x import Y";
        let tree = parse(code);
        let current = resolved(&["pkg"], &["Y"]);

        let found = resolve_reexport_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }
}

#[cfg(test)]
mod resolve_terminal_symbol_once_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn resolved(module_parts: &[&str], symbol_parts: &[&str]) -> ResolvedModuleTarget {
        ResolvedModuleTarget {
            dots: 0,
            module_parts: parts(module_parts),
            file_path: PathBuf::from("unused.py"),
            symbol_parts: parts(symbol_parts),
        }
    }

    #[test]
    fn test_returns_class_symbol() {
        let code = "class Linear: pass";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn", "_linear"], &["Linear"]);

        let found = resolve_terminal_symbol_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(
            found,
            Some(PythonSymbol::Class {
                name: "Linear".to_string()
            })
        );
    }

    #[test]
    fn test_returns_function_symbol() {
        let code = "def concatenate(): pass";
        let tree = parse(code);
        let current = resolved(&["jax", "numpy"], &["concatenate"]);

        let found = resolve_terminal_symbol_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(
            found,
            Some(PythonSymbol::Function {
                name: "concatenate".to_string()
            })
        );
    }

    #[test]
    fn test_import_symbol_returns_none() {
        let code = "from ._linear import Linear";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &["Linear"]);

        let found = resolve_terminal_symbol_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_symbol_returns_none() {
        let code = "class Other: pass";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &["Linear"]);

        let found = resolve_terminal_symbol_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_empty_symbol_parts_returns_none() {
        let code = "class Linear: pass";
        let tree = parse(code);
        let current = resolved(&["equinox", "nn"], &[]);

        let found = resolve_terminal_symbol_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_uses_first_symbol_part_only() {
        let code = "class nn: pass";
        let tree = parse(code);
        let current = resolved(&["torch"], &["nn", "Linear"]);

        let found = resolve_terminal_symbol_once(&current, tree.root_node(), code).unwrap();

        assert_eq!(
            found,
            Some(PythonSymbol::Class {
                name: "nn".to_string()
            })
        );
    }
}

#[cfg(test)]
mod follow_import_symbol_once_tests {
    use super::*;

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: parts(module),
            name: name.to_string(),
        }
    }

    fn target(parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots: 0,
            parts: self::parts(parts),
        }
    }

    fn import(path: ImportPath) -> PythonSymbol {
        PythonSymbol::Import {
            name: path.name.clone(),
            path,
        }
    }

    #[test]
    fn test_follows_relative_import_with_module() {
        let found = follow_import_symbol_once(
            &parts(&["equinox", "nn"]),
            &import(ip(1, &["_linear"], "Linear")),
        );

        assert_eq!(found, Some(target(&["equinox", "nn", "_linear", "Linear"])));
    }

    #[test]
    fn test_follows_relative_import_without_module() {
        let found =
            follow_import_symbol_once(&parts(&["equinox", "nn"]), &import(ip(1, &[], "layers")));

        assert_eq!(found, Some(target(&["equinox", "nn", "layers"])));
    }

    #[test]
    fn test_follows_relative_parent_import() {
        let found = follow_import_symbol_once(
            &parts(&["pkg", "sub"]),
            &import(ip(2, &["layers"], "Linear")),
        );

        assert_eq!(found, Some(target(&["pkg", "layers", "Linear"])));
    }

    #[test]
    fn test_follows_absolute_import() {
        let found = follow_import_symbol_once(
            &parts(&["equinox", "nn"]),
            &import(ip(0, &["jax", "numpy"], "concatenate")),
        );

        assert_eq!(found, Some(target(&["jax", "numpy", "concatenate"])));
    }

    #[test]
    fn test_too_many_relative_dots_returns_none() {
        let found = follow_import_symbol_once(&parts(&["pkg"]), &import(ip(2, &["x"], "Y")));

        assert_eq!(found, None);
    }

    #[test]
    fn test_class_returns_none() {
        let found = follow_import_symbol_once(
            &parts(&["equinox", "nn"]),
            &PythonSymbol::Class {
                name: "Linear".to_string(),
            },
        );

        assert_eq!(found, None);
    }

    #[test]
    fn test_function_returns_none() {
        let found = follow_import_symbol_once(
            &parts(&["equinox", "nn"]),
            &PythonSymbol::Function {
                name: "linear".to_string(),
            },
        );

        assert_eq!(found, None);
    }
}

#[cfg(test)]
mod find_top_level_symbol_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: module.iter().map(|part| part.to_string()).collect(),
            name: name.to_string(),
        }
    }

    fn class(name: &str) -> PythonSymbol {
        PythonSymbol::Class {
            name: name.to_string(),
        }
    }

    fn function(name: &str) -> PythonSymbol {
        PythonSymbol::Function {
            name: name.to_string(),
        }
    }

    fn import(name: &str, path: ImportPath) -> PythonSymbol {
        PythonSymbol::Import {
            name: name.to_string(),
            path,
        }
    }

    #[test]
    fn test_finds_top_level_class() {
        let code = "class Linear: pass";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, Some(class("Linear")));
    }

    #[test]
    fn test_finds_top_level_function() {
        let code = "def linear(): pass";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "linear").unwrap();

        assert_eq!(found, Some(function("linear")));
    }

    #[test]
    fn test_finds_plain_import() {
        let code = "import foo";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "foo").unwrap();

        assert_eq!(found, Some(import("foo", ip(0, &[], "foo"))));
    }

    #[test]
    fn test_finds_plain_import_alias() {
        let code = "import foo as bar";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "bar").unwrap();

        assert_eq!(found, Some(import("bar", ip(0, &[], "foo"))));
    }

    #[test]
    fn test_finds_deep_import_alias() {
        let code = "import foo.bar as baz";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "baz").unwrap();

        assert_eq!(found, Some(import("baz", ip(0, &["foo"], "bar"))));
    }

    #[test]
    fn test_finds_from_import() {
        let code = "from jax import random";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "random").unwrap();

        assert_eq!(found, Some(import("random", ip(0, &["jax"], "random"))));
    }

    #[test]
    fn test_finds_from_import_alias() {
        let code = "from equinox.nn import Linear as Lin";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Lin").unwrap();

        assert_eq!(
            found,
            Some(import("Lin", ip(0, &["equinox", "nn"], "Linear")))
        );
    }

    #[test]
    fn test_finds_relative_from_import() {
        let code = "from ._linear import Linear";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, Some(import("Linear", ip(1, &["_linear"], "Linear"))));
    }

    #[test]
    fn test_finds_relative_package_import() {
        let code = "from . import layers";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "layers").unwrap();

        assert_eq!(found, Some(import("layers", ip(1, &[], "layers"))));
    }

    #[test]
    fn test_skips_nested_class() {
        let code = "def outer():\n    class Linear: pass";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_skips_nested_function() {
        let code = "def outer():\n    def inner(): pass";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "inner").unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_star_import_does_not_define_named_symbol() {
        let code = "from foo import *";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_symbol_returns_none() {
        let code = "class Other: pass";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn test_last_top_level_definition_wins() {
        let code = "from foo import Linear\nclass Linear: pass";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, Some(class("Linear")));
    }

    #[test]
    fn test_last_top_level_import_wins() {
        let code = "class Linear: pass\nfrom foo import Linear";
        let tree = parse(code);

        let found = find_top_level_symbol(tree.root_node(), code, "Linear").unwrap();

        assert_eq!(found, Some(import("Linear", ip(0, &["foo"], "Linear"))));
    }
}

#[cfg(test)]
mod resolve_call_target_tests {
    use super::*;

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: module.iter().map(|part| part.to_string()).collect(),
            name: name.to_string(),
        }
    }

    fn rt(dots: usize, parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots,
            parts: parts.iter().map(|part| part.to_string()).collect(),
        }
    }

    #[test]
    fn test_unimported_single_segment_target_returns_itself() {
        let import_map = HashMap::new();

        let resolved = resolve_call_target("foo", &import_map);

        assert_eq!(resolved, rt(0, &["foo"]));
    }

    #[test]
    fn test_unimported_dotted_target_returns_itself() {
        let import_map = HashMap::new();

        let resolved = resolve_call_target("foo.bar.baz", &import_map);

        assert_eq!(resolved, rt(0, &["foo", "bar", "baz"]));
    }

    #[test]
    fn test_plain_import_alias_resolves_first_segment() {
        let import_map = HashMap::from([("jnp".to_string(), ip(0, &["jax"], "numpy"))]);

        let resolved = resolve_call_target("jnp.concatenate", &import_map);

        assert_eq!(resolved, rt(0, &["jax", "numpy", "concatenate"]));
    }

    #[test]
    fn test_plain_import_without_alias_resolves_first_segment() {
        let import_map = HashMap::from([("jax".to_string(), ip(0, &[], "jax"))]);

        let resolved = resolve_call_target("jax.numpy.concatenate", &import_map);

        assert_eq!(resolved, rt(0, &["jax", "numpy", "concatenate"]));
    }

    #[test]
    fn test_from_import_resolves_imported_name() {
        let import_map = HashMap::from([("random".to_string(), ip(0, &["jax"], "random"))]);

        let resolved = resolve_call_target("random.PRNGKey", &import_map);

        assert_eq!(resolved, rt(0, &["jax", "random", "PRNGKey"]));
    }

    #[test]
    fn test_from_import_alias_resolves_alias() {
        let import_map = HashMap::from([("lin".to_string(), ip(0, &["equinox", "nn"], "Linear"))]);

        let resolved = resolve_call_target("lin", &import_map);

        assert_eq!(resolved, rt(0, &["equinox", "nn", "Linear"]));
    }

    #[test]
    fn test_deep_import_alias_resolves_first_segment_only() {
        let import_map = HashMap::from([("nn".to_string(), ip(0, &["equinox"], "nn"))]);

        let resolved = resolve_call_target("nn.Linear", &import_map);

        assert_eq!(resolved, rt(0, &["equinox", "nn", "Linear"]));
    }

    #[test]
    fn test_only_exact_first_segment_is_resolved() {
        let import_map = HashMap::from([("foo".to_string(), ip(0, &["real"], "foo"))]);

        let resolved = resolve_call_target("foobar.baz", &import_map);

        assert_eq!(resolved, rt(0, &["foobar", "baz"]));
    }

    #[test]
    fn test_empty_target_returns_empty_parts() {
        let import_map = HashMap::new();

        let resolved = resolve_call_target("", &import_map);

        assert_eq!(resolved, rt(0, &[]));
    }

    #[test]
    fn test_extra_dots_are_ignored() {
        let import_map = HashMap::new();

        let resolved = resolve_call_target("foo..bar.", &import_map);

        assert_eq!(resolved, rt(0, &["foo", "bar"]));
    }

    #[test]
    fn test_relative_import_preserves_one_dot() {
        let import_map = HashMap::from([("Linear".to_string(), ip(1, &["layers"], "Linear"))]);

        let resolved = resolve_call_target("Linear", &import_map);

        assert_eq!(resolved, rt(1, &["layers", "Linear"]));
    }

    #[test]
    fn test_relative_import_preserves_multiple_dots() {
        let import_map = HashMap::from([("Linear".to_string(), ip(2, &["layers"], "Linear"))]);

        let resolved = resolve_call_target("Linear", &import_map);

        assert_eq!(resolved, rt(2, &["layers", "Linear"]));
    }

    #[test]
    fn test_relative_import_preserves_dots_with_remaining_target_parts() {
        let import_map = HashMap::from([("layers".to_string(), ip(1, &[], "layers"))]);

        let resolved = resolve_call_target("layers.Linear", &import_map);

        assert_eq!(resolved, rt(1, &["layers", "Linear"]));
    }
}

#[cfg(test)]
mod resolve_target_on_disk_tests {
    use super::*;
    use std::fs;

    fn target(dots: usize, parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots,
            parts: parts.iter().map(|part| part.to_string()).collect(),
        }
    }

    fn expected(
        dots: usize,
        module_parts: &[&str],
        file_path: PathBuf,
        symbol_parts: &[&str],
    ) -> ResolvedModuleTarget {
        ResolvedModuleTarget {
            dots,
            module_parts: module_parts.iter().map(|part| part.to_string()).collect(),
            file_path,
            symbol_parts: symbol_parts.iter().map(|part| part.to_string()).collect(),
        }
    }

    #[test]
    fn test_resolves_exact_module_file_with_no_symbol_parts() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar/baz.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar", "baz"],
                tmp.path().join("foo/bar/baz.py"),
                &[]
            ))
        );
    }

    #[test]
    fn test_resolves_exact_package_with_no_symbol_parts() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar/baz")).unwrap();
        fs::write(tmp.path().join("foo/bar/baz/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar", "baz"],
                tmp.path().join("foo/bar/baz/__init__.py"),
                &[]
            ))
        );
    }

    #[test]
    fn test_resolves_longest_module_prefix_and_keeps_symbol() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "Baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar"],
                tmp.path().join("foo/bar.py"),
                &["Baz"]
            ))
        );
    }

    #[test]
    fn test_falls_back_to_shorter_module_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo")).unwrap();
        fs::write(tmp.path().join("foo.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "Baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo"],
                tmp.path().join("foo.py"),
                &["bar", "Baz"]
            ))
        );
    }

    #[test]
    fn test_prefers_longer_module_over_shorter_module() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo")).unwrap();
        fs::write(tmp.path().join("foo.py"), "").unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "Baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar"],
                tmp.path().join("foo/bar.py"),
                &["Baz"]
            ))
        );
    }

    #[test]
    fn test_module_file_wins_over_package_init_for_same_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();
        fs::write(tmp.path().join("foo/bar/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar", "Baz"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo", "bar"],
                tmp.path().join("foo/bar.py"),
                &["Baz"]
            ))
        );
    }

    #[test]
    fn test_searches_roots_in_order_for_same_prefix() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(first.path().join("foo.py"), "").unwrap();
        fs::write(second.path().join("foo.py"), "").unwrap();

        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "Bar"]), &roots);

        assert_eq!(
            found,
            Some(expected(0, &["foo"], first.path().join("foo.py"), &["Bar"]))
        );
    }

    #[test]
    fn test_searches_later_roots_if_missing_in_first_root() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(second.path().join("foo.py"), "").unwrap();

        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "Bar"]), &roots);

        assert_eq!(
            found,
            Some(expected(
                0,
                &["foo"],
                second.path().join("foo.py"),
                &["Bar"]
            ))
        );
    }

    #[test]
    fn test_empty_target_parts_returns_none() {
        let tmp = tempfile::tempdir().unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &[]), &roots);

        assert_eq!(found, None);
    }

    #[test]
    fn test_empty_search_roots_returns_none() {
        let found = resolve_target_on_disk(&target(0, &["foo"]), &[]);

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_module_returns_none() {
        let tmp = tempfile::tempdir().unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(0, &["foo", "bar"]), &roots);

        assert_eq!(found, None);
    }

    #[test]
    fn test_relative_target_returns_none_until_base_package_is_known() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("layers")).unwrap();
        fs::write(tmp.path().join("layers.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_target_on_disk(&target(1, &["layers", "Linear"]), &roots);

        assert_eq!(found, None);
    }
}

#[cfg(test)]
mod resolve_python_module_on_disk_tests {
    use super::*;
    use std::fs;

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    #[test]
    fn test_resolves_module_file() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo")).unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/bar.py")));
    }

    #[test]
    fn test_resolves_package_init() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/bar/__init__.py")));
    }

    #[test]
    fn test_module_file_wins_over_package_init() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar")).unwrap();
        fs::write(tmp.path().join("foo/bar.py"), "").unwrap();
        fs::write(tmp.path().join("foo/bar/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/bar.py")));
    }

    #[test]
    fn test_resolves_deeply_nested_module_file() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar/baz")).unwrap();
        fs::write(tmp.path().join("foo/bar/baz/qux.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar", "baz", "qux"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/bar/baz/qux.py")));
    }

    #[test]
    fn test_resolves_deeply_nested_package_init() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo/bar/baz/qux")).unwrap();
        fs::write(tmp.path().join("foo/bar/baz/qux/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar", "baz", "qux"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/bar/baz/qux/__init__.py")));
    }

    #[test]
    fn test_resolves_single_segment_module_file() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("foo.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo.py")));
    }

    #[test]
    fn test_resolves_single_segment_package_init() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("foo")).unwrap();
        fs::write(tmp.path().join("foo/__init__.py"), "").unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo"]), &roots);

        assert_eq!(found, Some(tmp.path().join("foo/__init__.py")));
    }

    #[test]
    fn test_searches_roots_in_order() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(first.path().join("foo.py"), "").unwrap();
        fs::write(second.path().join("foo.py"), "").unwrap();

        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo"]), &roots);

        assert_eq!(found, Some(first.path().join("foo.py")));
    }

    #[test]
    fn test_searches_later_roots_if_missing_in_first_root() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(second.path().join("foo.py"), "").unwrap();

        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo"]), &roots);

        assert_eq!(found, Some(second.path().join("foo.py")));
    }

    #[test]
    fn test_empty_module_returns_none() {
        let tmp = tempfile::tempdir().unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&[], &roots);

        assert_eq!(found, None);
    }

    #[test]
    fn test_empty_search_roots_returns_none() {
        let found = resolve_python_module_on_disk(&parts(&["foo"]), &[]);

        assert_eq!(found, None);
    }

    #[test]
    fn test_missing_module_returns_none() {
        let tmp = tempfile::tempdir().unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let found = resolve_python_module_on_disk(&parts(&["foo", "bar"]), &roots);

        assert_eq!(found, None);
    }
}

#[cfg(test)]
mod known_function_shape_rule_tests {
    use super::*;

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn pos(value: &str) -> CallArgument {
        CallArgument::Positional {
            value: value.to_string(),
        }
    }

    fn kw(name: &str, value: &str) -> CallArgument {
        CallArgument::Keyword {
            name: name.to_string(),
            value: value.to_string(),
        }
    }

    #[test]
    fn test_concatenate_axis_0_numeric_dims() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["2", "features"])),
            ("b".to_string(), shape(&["3", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["5", "features"])));
    }

    #[test]
    fn test_concatenate_axis_1_numeric_dims() {
        let args = vec![pos("[a, b]"), kw("axis", "1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "2"])),
            ("b".to_string(), shape(&["batch", "3"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "5"])));
    }

    #[test]
    fn test_concatenate_negative_axis() {
        let args = vec![pos("[a, b]"), kw("axis", "-1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "2"])),
            ("b".to_string(), shape(&["batch", "3"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "5"])));
    }

    #[test]
    fn test_concatenate_torch_dim_keyword() {
        let args = vec![pos("[a, b]"), kw("dim", "1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "m"])),
            ("b".to_string(), shape(&["batch", "n"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "m+n"])));
    }

    #[test]
    fn test_concatenate_arrays_keyword() {
        let args = vec![kw("arrays", "[a, b]"), kw("axis", "0")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "features"])),
            ("b".to_string(), shape(&["n", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m+n", "features"])));
    }

    #[test]
    fn test_concatenate_tensors_keyword() {
        let args = vec![kw("tensors", "[a, b]"), kw("dim", "0")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "features"])),
            ("b".to_string(), shape(&["n", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m+n", "features"])));
    }

    #[test]
    fn test_concatenate_tuple_inputs() {
        let args = vec![pos("(a, b)")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["m", "features"])),
            ("b".to_string(), shape(&["n", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["m+n", "features"])));
    }

    #[test]
    fn test_concatenate_single_input_returns_same_shape() {
        let args = vec![pos("[a]")];
        let shapes = HashMap::from([("a".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_concatenate_missing_input_returns_none() {
        let args = vec![pos("[a, missing]")];
        let shapes = HashMap::from([("a".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_concatenate_no_args_returns_none() {
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Concatenate, &[], &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_concatenate_rank_mismatch_errors() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["features"])),
        ]);

        let error = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap_err();

        assert!(error.contains("rank"));
    }

    #[test]
    fn test_concatenate_non_axis_dim_mismatch_errors() {
        let args = vec![pos("[a, b]"), kw("axis", "1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "2"])),
            ("b".to_string(), shape(&["other", "3"])),
        ]);

        let error = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap_err();

        assert!(error.contains("dimension mismatch"));
        assert!(error.contains("expected batch"));
        assert!(error.contains("got other"));
    }

    #[test]
    fn test_concatenate_axis_out_of_bounds_errors() {
        let args = vec![pos("[a, b]"), kw("axis", "2")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "2"])),
            ("b".to_string(), shape(&["batch", "3"])),
        ]);

        let error = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap_err();

        assert!(error.contains("out of bounds"));
    }

    #[test]
    fn test_concatenate_negative_axis_out_of_bounds_errors() {
        let args = vec![pos("[a, b]"), kw("axis", "-3")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "2"])),
            ("b".to_string(), shape(&["batch", "3"])),
        ]);

        let error = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap_err();

        assert!(error.contains("out of bounds"));
    }

    #[test]
    fn test_concatenate_scalar_input_errors() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([("a".to_string(), Vec::new()), ("b".to_string(), Vec::new())]);

        let error = apply_known_function(&KnownFunction::Concatenate, &args, &shapes).unwrap_err();

        assert!(error.contains("rank >= 1"));
    }

    #[test]
    fn test_stack_default_axis() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "batch", "features"])));
    }

    #[test]
    fn test_stack_axis_1() {
        let args = vec![pos("[a, b, c]"), kw("axis", "1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch", "features"])),
            ("c".to_string(), shape(&["batch", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "3", "features"])));
    }

    #[test]
    fn test_stack_negative_axis() {
        let args = vec![pos("[a, b]"), kw("axis", "-1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features", "2"])));
    }

    #[test]
    fn test_stack_torch_dim_keyword() {
        let args = vec![pos("[a, b]"), kw("dim", "1")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch", "features"])),
        ]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "2", "features"])));
    }

    #[test]
    fn test_stack_arys_keyword() {
        let args = vec![kw("arys", "[a, b]"), kw("axis", "0")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch"])),
            ("b".to_string(), shape(&["batch"])),
        ]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "batch"])));
    }

    #[test]
    fn test_stack_tuple_inputs() {
        let args = vec![pos("(a, b)")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch"])),
            ("b".to_string(), shape(&["batch"])),
        ]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "batch"])));
    }

    #[test]
    fn test_stack_missing_input_returns_none() {
        let args = vec![pos("[a, missing]")];
        let shapes = HashMap::from([("a".to_string(), shape(&["batch"]))]);

        let output = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_stack_no_args_returns_none() {
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Stack, &[], &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_stack_rank_mismatch_errors() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch"])),
        ]);

        let error = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap_err();

        assert!(error.contains("rank"));
    }

    #[test]
    fn test_stack_dim_mismatch_errors() {
        let args = vec![pos("[a, b]")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch", "other"])),
        ]);

        let error = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap_err();

        assert!(error.contains("dimension mismatch"));
        assert!(error.contains("expected features"));
        assert!(error.contains("got other"));
    }

    #[test]
    fn test_stack_axis_out_of_bounds_errors() {
        let args = vec![pos("[a, b]"), kw("axis", "3")];
        let shapes = HashMap::from([
            ("a".to_string(), shape(&["batch", "features"])),
            ("b".to_string(), shape(&["batch", "features"])),
        ]);

        let error = apply_known_function(&KnownFunction::Stack, &args, &shapes).unwrap_err();

        assert!(error.contains("out of bounds"));
    }

    #[test]
    fn test_reshape_positional_shape() {
        let args = vec![pos("x"), pos("(2, 6)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["3", "4"]))]);

        let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "6"])));
    }

    #[test]
    fn test_reshape_keyword_shape() {
        let args = vec![pos("x"), kw("shape", "[2, 6]")];
        let shapes = HashMap::from([("x".to_string(), shape(&["3", "4"]))]);

        let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "6"])));
    }

    #[test]
    fn test_reshape_infers_minus_one() {
        let args = vec![pos("x"), pos("(2, -1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["3", "4"]))]);

        let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["2", "6"])));
    }

    #[test]
    fn test_reshape_symbolic_shape_allowed_without_size_check() {
        let args = vec![pos("x"), pos("(batch, features)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "features"])));
    }

    #[test]
    fn test_reshape_size_mismatch_errors() {
        let args = vec![pos("x"), pos("(5, 5)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["3", "4"]))]);

        let error = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap_err();

        assert!(error.contains("changes total size"));
    }

    #[test]
    fn test_reshape_multiple_minus_one_errors() {
        let args = vec![pos("x"), pos("(-1, -1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["3", "4"]))]);

        let error = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap_err();

        assert!(error.contains("only infer one -1"));
    }

    #[test]
    fn test_flatten_numeric_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["2", "3", "4"]))]);

        let output = apply_known_function(&KnownFunction::Flatten, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["24"])));
    }

    #[test]
    fn test_flatten_symbolic_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Flatten, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch*features"])));
    }

    #[test]
    fn test_ravel_uses_flatten_rule() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["2", "3"]))]);

        let output = apply_known_function(&KnownFunction::Ravel, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["6"])));
    }

    #[test]
    fn test_transpose_default_reverses_axes() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::Transpose, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["c", "b", "a"])));
    }

    #[test]
    fn test_transpose_axes_keyword() {
        let args = vec![pos("x"), kw("axes", "(1, 0, 2)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::Transpose, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["b", "a", "c"])));
    }

    #[test]
    fn test_permute_dims_keyword() {
        let args = vec![pos("x"), kw("dims", "(2, 0, 1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::Permute, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["c", "a", "b"])));
    }

    #[test]
    fn test_transpose_negative_axes() {
        let args = vec![pos("x"), kw("axes", "(-1, -2, -3)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::Transpose, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["c", "b", "a"])));
    }

    #[test]
    fn test_transpose_wrong_number_of_axes_errors() {
        let args = vec![pos("x"), kw("axes", "(0, 1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let error = apply_known_function(&KnownFunction::Transpose, &args, &shapes).unwrap_err();

        assert!(error.contains("expected 3 axes"));
    }

    #[test]
    fn test_transpose_duplicate_axes_errors() {
        let args = vec![pos("x"), kw("axes", "(0, 0, 1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let error = apply_known_function(&KnownFunction::Transpose, &args, &shapes).unwrap_err();

        assert!(error.contains("duplicate"));
    }

    #[test]
    fn test_swapaxes_positional() {
        let args = vec![pos("x"), pos("0"), pos("2")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::SwapAxes, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["c", "b", "a"])));
    }

    #[test]
    fn test_swapaxes_torch_dim_keywords() {
        let args = vec![pos("x"), kw("dim0", "0"), kw("dim1", "1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::SwapAxes, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["b", "a", "c"])));
    }

    #[test]
    fn test_swapaxes_missing_axis_returns_none() {
        let args = vec![pos("x"), pos("0")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::SwapAxes, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_moveaxis_single_axis() {
        let args = vec![pos("x"), pos("0"), pos("2")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::MoveAxis, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["b", "c", "a"])));
    }

    #[test]
    fn test_moveaxis_keyword_axes() {
        let args = vec![pos("x"), kw("source", "0"), kw("destination", "1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let output = apply_known_function(&KnownFunction::MoveAxis, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["b", "a", "c"])));
    }

    #[test]
    fn test_moveaxis_length_mismatch_errors() {
        let args = vec![pos("x"), kw("source", "(0, 1)"), kw("destination", "2")];
        let shapes = HashMap::from([("x".to_string(), shape(&["a", "b", "c"]))]);

        let error = apply_known_function(&KnownFunction::MoveAxis, &args, &shapes).unwrap_err();

        assert!(error.contains("lengths differ"));
    }

    #[test]
    fn test_reduction_no_axis_returns_scalar_shape() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap();

        assert_eq!(output, Some(Vec::new()));
    }

    #[test]
    fn test_reduction_no_axis_keepdims() {
        let args = vec![pos("x"), kw("keepdims", "True")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Mean, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["1", "1"])));
    }

    #[test]
    fn test_reduction_axis_keyword_removes_axis() {
        let args = vec![pos("x"), kw("axis", "1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features", "hidden"]))]);

        let output = apply_known_function(&KnownFunction::Max, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "hidden"])));
    }

    #[test]
    fn test_reduction_dim_keyword_removes_axis() {
        let args = vec![pos("x"), kw("dim", "1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features", "hidden"]))]);

        let output = apply_known_function(&KnownFunction::Min, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "hidden"])));
    }

    #[test]
    fn test_reduction_positional_axis() {
        let args = vec![pos("x"), pos("0")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Prod, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["features"])));
    }

    #[test]
    fn test_reduction_negative_axis() {
        let args = vec![pos("x"), kw("axis", "-1")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Std, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch"])));
    }

    #[test]
    fn test_reduction_multiple_axes_tuple() {
        let args = vec![pos("x"), kw("axis", "(0, 2)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "time", "features"]))]);

        let output = apply_known_function(&KnownFunction::Var, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["time"])));
    }

    #[test]
    fn test_reduction_multiple_axes_keepdims() {
        let args = vec![pos("x"), kw("axis", "(0, 2)"), kw("keepdims", "True")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "time", "features"]))]);

        let output = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["1", "time", "1"])));
    }

    #[test]
    fn test_reduction_axis_none_reduces_all_axes() {
        let args = vec![pos("x"), kw("axis", "None")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap();

        assert_eq!(output, Some(Vec::new()));
    }

    #[test]
    fn test_reduction_missing_input_returns_none() {
        let args = vec![pos("x"), kw("axis", "0")];
        let shapes = HashMap::new();

        let output = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }

    #[test]
    fn test_reduction_axis_out_of_bounds_errors() {
        let args = vec![pos("x"), kw("axis", "2")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let error = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap_err();

        assert!(error.contains("out of bounds"));
    }

    #[test]
    fn test_reduction_duplicate_axes_errors() {
        let args = vec![pos("x"), kw("axis", "(1, 1)")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let error = apply_known_function(&KnownFunction::Sum, &args, &shapes).unwrap_err();

        assert!(error.contains("duplicate"));
    }

    #[test]
    fn test_non_implemented_known_function_returns_none() {
        let args = vec![pos("x")];
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "features"]))]);

        let output = apply_known_function(&KnownFunction::Reshape, &args, &shapes).unwrap();

        assert_eq!(output, None);
    }
}

#[cfg(test)]
mod known_function_tests {
    use super::*;
    use tree_sitter::Parser;

    fn target(parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots: 0,
            parts: parts.iter().map(|part| part.to_string()).collect(),
        }
    }

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    macro_rules! known_case {
        ($name:ident, [$($part:expr),*], $kind:expr) => {
            #[test]
            fn $name() {
                assert_eq!(classify_known_function(&target(&[$($part),*])), $kind);
            }
        };
    }

    known_case!(
        jnp_concatenate,
        ["jax", "numpy", "concatenate"],
        Some(KnownFunction::Concatenate)
    );
    known_case!(
        jnp_concat,
        ["jax", "numpy", "concat"],
        Some(KnownFunction::Concatenate)
    );
    known_case!(
        jnp_stack,
        ["jax", "numpy", "stack"],
        Some(KnownFunction::Stack)
    );
    known_case!(
        jnp_reshape,
        ["jax", "numpy", "reshape"],
        Some(KnownFunction::Reshape)
    );
    known_case!(
        jnp_transpose,
        ["jax", "numpy", "transpose"],
        Some(KnownFunction::Transpose)
    );
    known_case!(
        jnp_expand_dims,
        ["jax", "numpy", "expand_dims"],
        Some(KnownFunction::ExpandDims)
    );
    known_case!(
        jnp_squeeze,
        ["jax", "numpy", "squeeze"],
        Some(KnownFunction::Squeeze)
    );
    known_case!(jnp_sum, ["jax", "numpy", "sum"], Some(KnownFunction::Sum));
    known_case!(
        jnp_mean,
        ["jax", "numpy", "mean"],
        Some(KnownFunction::Mean)
    );
    known_case!(jnp_max, ["jax", "numpy", "max"], Some(KnownFunction::Max));
    known_case!(jnp_amax, ["jax", "numpy", "amax"], Some(KnownFunction::Max));
    known_case!(jnp_min, ["jax", "numpy", "min"], Some(KnownFunction::Min));
    known_case!(jnp_amin, ["jax", "numpy", "amin"], Some(KnownFunction::Min));
    known_case!(
        jnp_prod,
        ["jax", "numpy", "prod"],
        Some(KnownFunction::Prod)
    );
    known_case!(jnp_std, ["jax", "numpy", "std"], Some(KnownFunction::Std));
    known_case!(jnp_var, ["jax", "numpy", "var"], Some(KnownFunction::Var));
    known_case!(
        jnp_matmul,
        ["jax", "numpy", "matmul"],
        Some(KnownFunction::Matmul)
    );
    known_case!(jnp_dot, ["jax", "numpy", "dot"], Some(KnownFunction::Dot));
    known_case!(
        jnp_einsum,
        ["jax", "numpy", "einsum"],
        Some(KnownFunction::Einsum)
    );
    known_case!(
        jnp_split,
        ["jax", "numpy", "split"],
        Some(KnownFunction::Split)
    );
    known_case!(
        jnp_tile,
        ["jax", "numpy", "tile"],
        Some(KnownFunction::Tile)
    );
    known_case!(
        jnp_repeat,
        ["jax", "numpy", "repeat"],
        Some(KnownFunction::Repeat)
    );
    known_case!(
        jnp_flatten,
        ["jax", "numpy", "flatten"],
        Some(KnownFunction::Flatten)
    );
    known_case!(
        jnp_ravel,
        ["jax", "numpy", "ravel"],
        Some(KnownFunction::Ravel)
    );
    known_case!(
        jnp_moveaxis,
        ["jax", "numpy", "moveaxis"],
        Some(KnownFunction::MoveAxis)
    );
    known_case!(
        jnp_swapaxes,
        ["jax", "numpy", "swapaxes"],
        Some(KnownFunction::SwapAxes)
    );
    known_case!(
        jnp_where,
        ["jax", "numpy", "where"],
        Some(KnownFunction::Where)
    );
    known_case!(
        jnp_zeros,
        ["jax", "numpy", "zeros"],
        Some(KnownFunction::Zeros)
    );
    known_case!(
        jnp_ones,
        ["jax", "numpy", "ones"],
        Some(KnownFunction::Ones)
    );
    known_case!(
        jnp_full,
        ["jax", "numpy", "full"],
        Some(KnownFunction::Full)
    );
    known_case!(
        jnp_arange,
        ["jax", "numpy", "arange"],
        Some(KnownFunction::Arange)
    );
    known_case!(jnp_eye, ["jax", "numpy", "eye"], Some(KnownFunction::Eye));
    known_case!(
        jnp_broadcast_to,
        ["jax", "numpy", "broadcast_to"],
        Some(KnownFunction::BroadcastTo)
    );
    known_case!(
        jnp_broadcast_arrays,
        ["jax", "numpy", "broadcast_arrays"],
        Some(KnownFunction::BroadcastArrays)
    );
    known_case!(
        jnp_atleast_1d,
        ["jax", "numpy", "atleast_1d"],
        Some(KnownFunction::AtLeast1D)
    );
    known_case!(
        jnp_atleast_2d,
        ["jax", "numpy", "atleast_2d"],
        Some(KnownFunction::AtLeast2D)
    );
    known_case!(
        jnp_atleast_3d,
        ["jax", "numpy", "atleast_3d"],
        Some(KnownFunction::AtLeast3D)
    );
    known_case!(jnp_pad, ["jax", "numpy", "pad"], Some(KnownFunction::Pad));
    known_case!(
        jnp_roll,
        ["jax", "numpy", "roll"],
        Some(KnownFunction::Roll)
    );
    known_case!(
        jnp_flip,
        ["jax", "numpy", "flip"],
        Some(KnownFunction::Flip)
    );
    known_case!(
        jnp_fliplr,
        ["jax", "numpy", "fliplr"],
        Some(KnownFunction::Flip)
    );
    known_case!(
        jnp_flipud,
        ["jax", "numpy", "flipud"],
        Some(KnownFunction::Flip)
    );
    known_case!(
        jnp_rot90,
        ["jax", "numpy", "rot90"],
        Some(KnownFunction::Rot90)
    );
    known_case!(
        jnp_take,
        ["jax", "numpy", "take"],
        Some(KnownFunction::Take)
    );
    known_case!(
        jnp_diag,
        ["jax", "numpy", "diag"],
        Some(KnownFunction::Diag)
    );
    known_case!(
        jnp_diagonal,
        ["jax", "numpy", "diagonal"],
        Some(KnownFunction::Diagonal)
    );
    known_case!(
        jnp_trace,
        ["jax", "numpy", "trace"],
        Some(KnownFunction::Trace)
    );
    known_case!(
        jnp_triu,
        ["jax", "numpy", "triu"],
        Some(KnownFunction::Triu)
    );
    known_case!(
        jnp_tril,
        ["jax", "numpy", "tril"],
        Some(KnownFunction::Tril)
    );
    known_case!(
        jnp_meshgrid,
        ["jax", "numpy", "meshgrid"],
        Some(KnownFunction::Meshgrid)
    );
    known_case!(
        jnp_vstack,
        ["jax", "numpy", "vstack"],
        Some(KnownFunction::Vstack)
    );
    known_case!(
        jnp_row_stack,
        ["jax", "numpy", "row_stack"],
        Some(KnownFunction::Vstack)
    );
    known_case!(
        jnp_hstack,
        ["jax", "numpy", "hstack"],
        Some(KnownFunction::Hstack)
    );
    known_case!(
        jnp_dstack,
        ["jax", "numpy", "dstack"],
        Some(KnownFunction::Dstack)
    );
    known_case!(
        jnp_column_stack,
        ["jax", "numpy", "column_stack"],
        Some(KnownFunction::ColumnStack)
    );
    known_case!(
        jnp_block,
        ["jax", "numpy", "block"],
        Some(KnownFunction::Block)
    );
    known_case!(
        jnp_array,
        ["jax", "numpy", "array"],
        Some(KnownFunction::Array)
    );
    known_case!(
        jnp_asarray,
        ["jax", "numpy", "asarray"],
        Some(KnownFunction::AsArray)
    );

    known_case!(
        np_concatenate,
        ["numpy", "concatenate"],
        Some(KnownFunction::Concatenate)
    );
    known_case!(np_stack, ["numpy", "stack"], Some(KnownFunction::Stack));
    known_case!(
        np_reshape,
        ["numpy", "reshape"],
        Some(KnownFunction::Reshape)
    );
    known_case!(
        np_transpose,
        ["numpy", "transpose"],
        Some(KnownFunction::Transpose)
    );
    known_case!(
        np_expand_dims,
        ["numpy", "expand_dims"],
        Some(KnownFunction::ExpandDims)
    );
    known_case!(
        np_squeeze,
        ["numpy", "squeeze"],
        Some(KnownFunction::Squeeze)
    );
    known_case!(np_sum, ["numpy", "sum"], Some(KnownFunction::Sum));
    known_case!(np_mean, ["numpy", "mean"], Some(KnownFunction::Mean));
    known_case!(np_amax, ["numpy", "amax"], Some(KnownFunction::Max));
    known_case!(np_amin, ["numpy", "amin"], Some(KnownFunction::Min));
    known_case!(np_prod, ["numpy", "prod"], Some(KnownFunction::Prod));
    known_case!(np_std, ["numpy", "std"], Some(KnownFunction::Std));
    known_case!(np_var, ["numpy", "var"], Some(KnownFunction::Var));
    known_case!(np_matmul, ["numpy", "matmul"], Some(KnownFunction::Matmul));
    known_case!(np_dot, ["numpy", "dot"], Some(KnownFunction::Dot));
    known_case!(np_einsum, ["numpy", "einsum"], Some(KnownFunction::Einsum));
    known_case!(np_split, ["numpy", "split"], Some(KnownFunction::Split));
    known_case!(np_tile, ["numpy", "tile"], Some(KnownFunction::Tile));
    known_case!(np_repeat, ["numpy", "repeat"], Some(KnownFunction::Repeat));
    known_case!(np_ravel, ["numpy", "ravel"], Some(KnownFunction::Ravel));
    known_case!(
        np_moveaxis,
        ["numpy", "moveaxis"],
        Some(KnownFunction::MoveAxis)
    );
    known_case!(
        np_swapaxes,
        ["numpy", "swapaxes"],
        Some(KnownFunction::SwapAxes)
    );
    known_case!(np_where, ["numpy", "where"], Some(KnownFunction::Where));
    known_case!(np_zeros, ["numpy", "zeros"], Some(KnownFunction::Zeros));
    known_case!(np_ones, ["numpy", "ones"], Some(KnownFunction::Ones));
    known_case!(np_full, ["numpy", "full"], Some(KnownFunction::Full));
    known_case!(np_arange, ["numpy", "arange"], Some(KnownFunction::Arange));
    known_case!(np_eye, ["numpy", "eye"], Some(KnownFunction::Eye));
    known_case!(
        np_broadcast_to,
        ["numpy", "broadcast_to"],
        Some(KnownFunction::BroadcastTo)
    );
    known_case!(
        np_broadcast_arrays,
        ["numpy", "broadcast_arrays"],
        Some(KnownFunction::BroadcastArrays)
    );
    known_case!(
        np_atleast_1d,
        ["numpy", "atleast_1d"],
        Some(KnownFunction::AtLeast1D)
    );
    known_case!(
        np_atleast_2d,
        ["numpy", "atleast_2d"],
        Some(KnownFunction::AtLeast2D)
    );
    known_case!(
        np_atleast_3d,
        ["numpy", "atleast_3d"],
        Some(KnownFunction::AtLeast3D)
    );
    known_case!(np_pad, ["numpy", "pad"], Some(KnownFunction::Pad));
    known_case!(np_roll, ["numpy", "roll"], Some(KnownFunction::Roll));
    known_case!(np_flip, ["numpy", "flip"], Some(KnownFunction::Flip));
    known_case!(np_fliplr, ["numpy", "fliplr"], Some(KnownFunction::Flip));
    known_case!(np_flipud, ["numpy", "flipud"], Some(KnownFunction::Flip));
    known_case!(np_rot90, ["numpy", "rot90"], Some(KnownFunction::Rot90));
    known_case!(np_take, ["numpy", "take"], Some(KnownFunction::Take));
    known_case!(np_diag, ["numpy", "diag"], Some(KnownFunction::Diag));
    known_case!(
        np_diagonal,
        ["numpy", "diagonal"],
        Some(KnownFunction::Diagonal)
    );
    known_case!(np_trace, ["numpy", "trace"], Some(KnownFunction::Trace));
    known_case!(np_triu, ["numpy", "triu"], Some(KnownFunction::Triu));
    known_case!(np_tril, ["numpy", "tril"], Some(KnownFunction::Tril));
    known_case!(
        np_meshgrid,
        ["numpy", "meshgrid"],
        Some(KnownFunction::Meshgrid)
    );
    known_case!(np_vstack, ["numpy", "vstack"], Some(KnownFunction::Vstack));
    known_case!(
        np_row_stack,
        ["numpy", "row_stack"],
        Some(KnownFunction::Vstack)
    );
    known_case!(np_hstack, ["numpy", "hstack"], Some(KnownFunction::Hstack));
    known_case!(np_dstack, ["numpy", "dstack"], Some(KnownFunction::Dstack));
    known_case!(
        np_column_stack,
        ["numpy", "column_stack"],
        Some(KnownFunction::ColumnStack)
    );
    known_case!(np_block, ["numpy", "block"], Some(KnownFunction::Block));
    known_case!(np_array, ["numpy", "array"], Some(KnownFunction::Array));
    known_case!(
        np_asarray,
        ["numpy", "asarray"],
        Some(KnownFunction::AsArray)
    );

    known_case!(
        torch_cat,
        ["torch", "cat"],
        Some(KnownFunction::Concatenate)
    );
    known_case!(
        torch_concat,
        ["torch", "concat"],
        Some(KnownFunction::Concatenate)
    );
    known_case!(
        torch_concatenate,
        ["torch", "concatenate"],
        Some(KnownFunction::Concatenate)
    );
    known_case!(torch_stack, ["torch", "stack"], Some(KnownFunction::Stack));
    known_case!(
        torch_reshape,
        ["torch", "reshape"],
        Some(KnownFunction::Reshape)
    );
    known_case!(
        torch_transpose,
        ["torch", "transpose"],
        Some(KnownFunction::Transpose)
    );
    known_case!(
        torch_unsqueeze,
        ["torch", "unsqueeze"],
        Some(KnownFunction::ExpandDims)
    );
    known_case!(
        torch_squeeze,
        ["torch", "squeeze"],
        Some(KnownFunction::Squeeze)
    );
    known_case!(torch_sum, ["torch", "sum"], Some(KnownFunction::Sum));
    known_case!(torch_mean, ["torch", "mean"], Some(KnownFunction::Mean));
    known_case!(torch_max, ["torch", "max"], Some(KnownFunction::Max));
    known_case!(torch_min, ["torch", "min"], Some(KnownFunction::Min));
    known_case!(torch_prod, ["torch", "prod"], Some(KnownFunction::Prod));
    known_case!(torch_std, ["torch", "std"], Some(KnownFunction::Std));
    known_case!(torch_var, ["torch", "var"], Some(KnownFunction::Var));
    known_case!(
        torch_matmul,
        ["torch", "matmul"],
        Some(KnownFunction::Matmul)
    );
    known_case!(torch_dot, ["torch", "dot"], Some(KnownFunction::Dot));
    known_case!(
        torch_einsum,
        ["torch", "einsum"],
        Some(KnownFunction::Einsum)
    );
    known_case!(torch_split, ["torch", "split"], Some(KnownFunction::Split));
    known_case!(torch_tile, ["torch", "tile"], Some(KnownFunction::Tile));
    known_case!(
        torch_repeat,
        ["torch", "repeat"],
        Some(KnownFunction::Repeat)
    );
    known_case!(
        torch_flatten,
        ["torch", "flatten"],
        Some(KnownFunction::Flatten)
    );
    known_case!(torch_ravel, ["torch", "ravel"], Some(KnownFunction::Ravel));
    known_case!(torch_where, ["torch", "where"], Some(KnownFunction::Where));
    known_case!(torch_zeros, ["torch", "zeros"], Some(KnownFunction::Zeros));
    known_case!(torch_ones, ["torch", "ones"], Some(KnownFunction::Ones));
    known_case!(torch_full, ["torch", "full"], Some(KnownFunction::Full));
    known_case!(
        torch_arange,
        ["torch", "arange"],
        Some(KnownFunction::Arange)
    );
    known_case!(torch_eye, ["torch", "eye"], Some(KnownFunction::Eye));
    known_case!(
        torch_broadcast_to,
        ["torch", "broadcast_to"],
        Some(KnownFunction::BroadcastTo)
    );
    known_case!(
        torch_broadcast_tensors,
        ["torch", "broadcast_tensors"],
        Some(KnownFunction::BroadcastArrays)
    );
    known_case!(
        torch_atleast_1d,
        ["torch", "atleast_1d"],
        Some(KnownFunction::AtLeast1D)
    );
    known_case!(
        torch_atleast_2d,
        ["torch", "atleast_2d"],
        Some(KnownFunction::AtLeast2D)
    );
    known_case!(
        torch_atleast_3d,
        ["torch", "atleast_3d"],
        Some(KnownFunction::AtLeast3D)
    );
    known_case!(torch_roll, ["torch", "roll"], Some(KnownFunction::Roll));
    known_case!(torch_flip, ["torch", "flip"], Some(KnownFunction::Flip));
    known_case!(torch_fliplr, ["torch", "fliplr"], Some(KnownFunction::Flip));
    known_case!(torch_flipud, ["torch", "flipud"], Some(KnownFunction::Flip));
    known_case!(torch_rot90, ["torch", "rot90"], Some(KnownFunction::Rot90));
    known_case!(torch_take, ["torch", "take"], Some(KnownFunction::Take));
    known_case!(torch_diag, ["torch", "diag"], Some(KnownFunction::Diag));
    known_case!(
        torch_diagonal,
        ["torch", "diagonal"],
        Some(KnownFunction::Diagonal)
    );
    known_case!(torch_trace, ["torch", "trace"], Some(KnownFunction::Trace));
    known_case!(torch_triu, ["torch", "triu"], Some(KnownFunction::Triu));
    known_case!(torch_tril, ["torch", "tril"], Some(KnownFunction::Tril));
    known_case!(
        torch_meshgrid,
        ["torch", "meshgrid"],
        Some(KnownFunction::Meshgrid)
    );
    known_case!(
        torch_vstack,
        ["torch", "vstack"],
        Some(KnownFunction::Vstack)
    );
    known_case!(
        torch_row_stack,
        ["torch", "row_stack"],
        Some(KnownFunction::Vstack)
    );
    known_case!(
        torch_hstack,
        ["torch", "hstack"],
        Some(KnownFunction::Hstack)
    );
    known_case!(
        torch_dstack,
        ["torch", "dstack"],
        Some(KnownFunction::Dstack)
    );
    known_case!(
        torch_column_stack,
        ["torch", "column_stack"],
        Some(KnownFunction::ColumnStack)
    );
    known_case!(
        torch_permute,
        ["torch", "permute"],
        Some(KnownFunction::Permute)
    );
    known_case!(
        torch_tensor,
        ["torch", "tensor"],
        Some(KnownFunction::Array)
    );
    known_case!(
        torch_as_tensor,
        ["torch", "as_tensor"],
        Some(KnownFunction::AsArray)
    );
    known_case!(torch_pad_rejected_for_now, ["torch", "pad"], None);
    known_case!(torch_block_rejected_for_now, ["torch", "block"], None);

    known_case!(jax_vmap, ["jax", "vmap"], Some(KnownFunction::Vmap));
    known_case!(jax_lax_dot, ["jax", "lax", "dot"], Some(KnownFunction::Dot));
    known_case!(
        jax_lax_dot_general,
        ["jax", "lax", "dot_general"],
        Some(KnownFunction::Matmul)
    );

    known_case!(unknown_module_rejected, ["foo", "concatenate"], None);
    known_case!(
        deep_unknown_module_rejected,
        ["jax", "random", "split"],
        None
    );
    known_case!(too_short_target_rejected, ["concatenate"], None);
    known_case!(empty_target_rejected, [], None);
    known_case!(
        case_sensitive_function_rejected,
        ["jax", "numpy", "Concatenate"],
        None
    );
    known_case!(
        case_sensitive_module_rejected,
        ["JAX", "numpy", "concatenate"],
        None
    );
    known_case!(method_like_known_function_rejected, ["x", "reshape"], None);
    known_case!(jax_numpy_vmap_rejected, ["jax", "numpy", "vmap"], None);
    known_case!(numpy_vmap_rejected, ["numpy", "vmap"], None);
    known_case!(torch_vmap_rejected_for_now, ["torch", "vmap"], None);
    known_case!(torch_nn_function_rejected, ["torch", "nn", "Linear"], None);
    known_case!(
        numpy_linalg_dot_rejected_for_now,
        ["numpy", "linalg", "dot"],
        None
    );
    known_case!(
        jax_numpy_linalg_norm_rejected_for_now,
        ["jax", "numpy", "linalg", "norm"],
        None
    );

    macro_rules! alias_case {
        ($name:ident, $code:expr, $target:expr, $kind:expr) => {
            #[test]
            fn $name() {
                let tree = parse($code);
                let import_map = build_import_map(tree.root_node(), $code).unwrap();
                let resolved = resolve_call_target($target, &import_map);
                assert_eq!(classify_known_function(&resolved), $kind);
            }
        };
    }

    alias_case!(
        alias_jnp_concatenate,
        "import jax.numpy as jnp",
        "jnp.concatenate",
        Some(KnownFunction::Concatenate)
    );
    alias_case!(
        alias_np_stack,
        "import numpy as np",
        "np.stack",
        Some(KnownFunction::Stack)
    );
    alias_case!(
        alias_torch_cat,
        "import torch as th",
        "th.cat",
        Some(KnownFunction::Concatenate)
    );
    alias_case!(
        alias_jax_vmap,
        "import jax",
        "jax.vmap",
        Some(KnownFunction::Vmap)
    );
    alias_case!(
        alias_jax_as_jx_vmap,
        "import jax as jx",
        "jx.vmap",
        Some(KnownFunction::Vmap)
    );
    alias_case!(
        alias_jax_lax_dot,
        "import jax.lax as lax",
        "lax.dot",
        Some(KnownFunction::Dot)
    );
    alias_case!(
        from_import_jax_vmap,
        "from jax import vmap",
        "vmap",
        Some(KnownFunction::Vmap)
    );
    alias_case!(
        from_import_jax_vmap_alias,
        "from jax import vmap as vm",
        "vm",
        Some(KnownFunction::Vmap)
    );
    alias_case!(
        from_import_jnp_reshape,
        "from jax.numpy import reshape",
        "reshape",
        Some(KnownFunction::Reshape)
    );
    alias_case!(
        from_import_np_transpose_alias,
        "from numpy import transpose as T",
        "T",
        Some(KnownFunction::Transpose)
    );
    alias_case!(
        from_import_torch_stack_alias,
        "from torch import stack as stk",
        "stk",
        Some(KnownFunction::Stack)
    );
    alias_case!(
        alias_jnp_broadcast_to,
        "import jax.numpy as jnp",
        "jnp.broadcast_to",
        Some(KnownFunction::BroadcastTo)
    );
    alias_case!(
        alias_np_vstack,
        "import numpy as np",
        "np.vstack",
        Some(KnownFunction::Vstack)
    );
    alias_case!(
        alias_torch_permute,
        "import torch as th",
        "th.permute",
        Some(KnownFunction::Permute)
    );
    alias_case!(
        from_import_np_meshgrid,
        "from numpy import meshgrid",
        "meshgrid",
        Some(KnownFunction::Meshgrid)
    );
    alias_case!(
        from_import_jnp_asarray_alias,
        "from jax.numpy import asarray as arr",
        "arr",
        Some(KnownFunction::AsArray)
    );
    alias_case!(
        from_import_torch_as_tensor,
        "from torch import as_tensor",
        "as_tensor",
        Some(KnownFunction::AsArray)
    );
    alias_case!(
        from_import_unknown_rejected,
        "from foo import concatenate",
        "concatenate",
        None
    );
    alias_case!(unimported_known_name_rejected, "", "concatenate", None);
    alias_case!(
        alias_exact_matching_rejected,
        "import numpy as np",
        "npx.concatenate",
        None
    );
}

#[cfg(test)]
mod additional_edge_case_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: parts(module),
            name: name.to_string(),
        }
    }

    fn rt(dots: usize, parts: &[&str]) -> ResolvedTarget {
        ResolvedTarget {
            dots,
            parts: self::parts(parts),
        }
    }

    fn sig(owner: Option<&str>, params: &[&str]) -> PythonCallableSignature {
        PythonCallableSignature {
            owner: owner.map(|owner| owner.to_string()),
            name: "f".to_string(),
            params: parts(params),
        }
    }

    fn pos(value: &str) -> CallArgument {
        CallArgument::Positional {
            value: value.to_string(),
        }
    }

    fn kw(name: &str, value: &str) -> CallArgument {
        CallArgument::Keyword {
            name: name.to_string(),
            value: value.to_string(),
        }
    }

    fn linear(in_features: &str, out_features: &str) -> LayerKind {
        LayerKind::Linear {
            in_features: in_features.to_string(),
            out_features: out_features.to_string(),
        }
    }

    fn dummy_range() -> Range {
        Range {
            start_byte: 0,
            end_byte: 0,
            start_point: tree_sitter::Point::new(0, 0),
            end_point: tree_sitter::Point::new(0, 0),
        }
    }

    fn app(input: &str, kind: LayerKind) -> LayerApplication {
        LayerApplication {
            variable: "out".to_string(),
            layer: "layer".to_string(),
            input: input.to_string(),
            kind,
            range: dummy_range(),
        }
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        parts(dims)
    }

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    macro_rules! resolve_call_target_case {
        ($name:ident, [$(($alias:expr, $path:expr)),*], $target:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let import_map = HashMap::from([
                    $(($alias.to_string(), $path),)*
                ]);
                assert_eq!(resolve_call_target($target, &import_map), $expected);
            }
        };
    }

    resolve_call_target_case!(
        call_target_alias_with_two_suffixes,
        [("np", ip(0, &["numpy"], "random"))],
        "np.default_rng.seed",
        rt(0, &["numpy", "random", "default_rng", "seed"])
    );
    resolve_call_target_case!(
        call_target_alias_exact_only,
        [("np", ip(0, &["numpy"], "random"))],
        "npx.foo",
        rt(0, &["npx", "foo"])
    );
    resolve_call_target_case!(
        call_target_from_import_with_suffix_chain,
        [("Linear", ip(0, &["equinox", "nn"], "Linear"))],
        "Linear.init.extra",
        rt(0, &["equinox", "nn", "Linear", "init", "extra"])
    );
    resolve_call_target_case!(
        call_target_relative_import_with_suffix_chain,
        [("layers", ip(1, &[], "layers"))],
        "layers.Linear.forward",
        rt(1, &["layers", "Linear", "forward"])
    );
    resolve_call_target_case!(
        call_target_empty_dots_with_import_map_still_empty,
        [("x", ip(0, &["pkg"], "x"))],
        "...",
        rt(0, &[])
    );
    resolve_call_target_case!(
        call_target_leading_dot_ignored_for_unimported,
        [],
        ".foo.bar",
        rt(0, &["foo", "bar"])
    );
    resolve_call_target_case!(
        call_target_trailing_dot_ignored_for_imported,
        [("foo", ip(0, &["pkg"], "foo"))],
        "foo.",
        rt(0, &["pkg", "foo"])
    );
    resolve_call_target_case!(
        call_target_many_empty_segments_ignored,
        [],
        "foo...bar..baz",
        rt(0, &["foo", "bar", "baz"])
    );

    macro_rules! import_path_case {
        ($name:ident, $current:expr, $path:expr, $expected:expr) => {
            #[test]
            fn $name() {
                assert_eq!(
                    resolve_import_path_from_package(&parts($current), &$path),
                    $expected
                );
            }
        };
    }

    import_path_case!(
        import_abs_deep_module,
        &["pkg"],
        ip(0, &["a", "b", "c"], "D"),
        Some(rt(0, &["a", "b", "c", "D"]))
    );
    import_path_case!(
        import_abs_single_name,
        &["pkg", "sub"],
        ip(0, &[], "D"),
        Some(rt(0, &["D"]))
    );
    import_path_case!(
        import_rel_same_deep_module,
        &["pkg", "sub"],
        ip(1, &["a", "b"], "C"),
        Some(rt(0, &["pkg", "sub", "a", "b", "C"]))
    );
    import_path_case!(
        import_rel_parent_no_module,
        &["pkg", "sub"],
        ip(2, &[], "C"),
        Some(rt(0, &["pkg", "C"]))
    );
    import_path_case!(
        import_rel_exact_top_too_far,
        &["pkg"],
        ip(2, &[], "C"),
        None
    );
    import_path_case!(import_rel_empty_current_too_far, &[], ip(1, &[], "C"), None);
    import_path_case!(
        import_rel_three_dots_from_four_parts,
        &["a", "b", "c", "d"],
        ip(3, &["x"], "Y"),
        Some(rt(0, &["a", "b", "x", "Y"]))
    );
    import_path_case!(
        import_rel_four_dots_from_four_parts,
        &["a", "b", "c", "d"],
        ip(4, &["x"], "Y"),
        Some(rt(0, &["a", "x", "Y"]))
    );

    macro_rules! bind_case {
        ($name:ident, $sig:expr, [$($arg:expr),*], [$(($param:expr, $value:expr)),*]) => {
            #[test]
            fn $name() {
                let bindings = bind_call_arguments(&$sig, &[$($arg),*]);
                let expected = HashMap::from([$(($param.to_string(), $value.to_string()),)*]);
                assert_eq!(bindings, expected);
            }
        };
    }

    bind_case!(bind_no_args_empty, sig(None, &["x"]), [], []);
    bind_case!(
        bind_too_many_positionals_ignored,
        sig(None, &["x"]),
        [pos("a"), pos("b")],
        [("x", "a")]
    );
    bind_case!(
        bind_unknown_keyword_preserved,
        sig(None, &["x"]),
        [kw("extra", "1")],
        [("extra", "1")]
    );
    bind_case!(
        bind_keyword_before_positional_does_not_shift_positionals,
        sig(None, &["x", "y"]),
        [kw("y", "2"), pos("1")],
        [("x", "1"), ("y", "2")]
    );
    bind_case!(
        bind_class_skips_cls,
        sig(Some("C"), &["cls", "x", "y"]),
        [pos("1"), pos("2")],
        [("x", "1"), ("y", "2")]
    );
    bind_case!(
        bind_class_does_not_skip_this,
        sig(Some("C"), &["this", "x"]),
        [pos("a"), pos("b")],
        [("this", "a"), ("x", "b")]
    );
    bind_case!(
        bind_function_does_not_skip_self,
        sig(None, &["self", "x"]),
        [pos("a"), pos("b")],
        [("self", "a"), ("x", "b")]
    );
    bind_case!(
        bind_keyword_overwrites_unknown_then_known_order,
        sig(None, &["x"]),
        [kw("x", "1"), kw("x", "2")],
        [("x", "2")]
    );
    bind_case!(
        bind_positional_then_keyword_override,
        sig(None, &["x"]),
        [pos("1"), kw("x", "2")],
        [("x", "2")]
    );
    bind_case!(
        bind_keyword_then_positional_override_by_position,
        sig(None, &["x"]),
        [kw("x", "1"), pos("2")],
        [("x", "2")]
    );

    macro_rules! apply_ok_case {
        ($name:ident, $input:expr, $layer:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let app = app("x", $layer);
                let shapes = HashMap::from([("x".to_string(), shape($input))]);
                assert_eq!(
                    apply_layer_application(&app, &shapes).unwrap(),
                    Some(shape($expected))
                );
            }
        };
    }

    apply_ok_case!(
        apply_rank_three_numeric,
        &["a", "b", "3"],
        linear("3", "5"),
        &["a", "b", "5"]
    );
    apply_ok_case!(
        apply_rank_four_symbolic,
        &["t", "b", "h", "features"],
        linear("features", "out"),
        &["t", "b", "h", "out"]
    );
    apply_ok_case!(
        apply_single_dim_symbolic,
        &["features"],
        linear("features", "out"),
        &["out"]
    );
    apply_ok_case!(
        apply_numeric_no_leading_dims,
        &["10"],
        linear("10", "20"),
        &["20"]
    );
    apply_ok_case!(
        apply_preserves_duplicate_leading_dims,
        &["batch", "batch", "features"],
        linear("features", "out"),
        &["batch", "batch", "out"]
    );

    macro_rules! apply_err_case {
        ($name:ident, $input:expr, $layer:expr, $needle:expr) => {
            #[test]
            fn $name() {
                let app = app("x", $layer);
                let shapes = HashMap::from([("x".to_string(), shape($input))]);
                let error = apply_layer_application(&app, &shapes).unwrap_err();
                assert!(error.contains($needle));
            }
        };
    }

    apply_err_case!(
        apply_mismatch_rank_three,
        &["a", "b", "4"],
        linear("3", "5"),
        "got 4"
    );
    apply_err_case!(
        apply_case_sensitive_symbols,
        &["Batch", "Features"],
        linear("features", "out"),
        "got Features"
    );
    apply_err_case!(
        apply_whitespace_not_normalized_in_dims,
        &["features "],
        linear("features", "out"),
        "got features "
    );
    apply_err_case!(
        apply_expression_dim_exact_mismatch,
        &["hidden * 2"],
        linear("hidden*2", "out"),
        "got hidden * 2"
    );
    apply_err_case!(
        apply_empty_last_dim_mismatch,
        &[""],
        linear("features", "out"),
        "got "
    );

    macro_rules! annotation_case {
        ($name:ident, $code:expr, [$(($param:expr, [$($dim:expr),*])),*]) => {
            #[test]
            fn $name() {
                let tree = parse($code);
                let shapes = extract_jaxtyping_shapes(tree.root_node(), $code).unwrap();
                let expected = HashMap::from([
                    $(($param.to_string(), vec![$($dim.to_string()),*]),)*
                ]);
                assert_eq!(shapes, expected);
            }
        };
    }

    annotation_case!(
        ann_bool_array_shape,
        "def f(x: Bool[Array, \"b d\"]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_complex_array_shape,
        "def f(x: Complex[Array, \"b d\"]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_lowercase_array_rejected,
        "def f(x: Float[array, \"b d\"]): pass",
        []
    );
    annotation_case!(
        ann_numpy_ndarray_rejected_for_now,
        "def f(x: Float[np.ndarray, \"b d\"]): pass",
        []
    );
    annotation_case!(
        ann_list_wrapped_array_shape,
        "def f(x: list[Float[Array, \"b d\"]]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_union_pipe_array_shape,
        "def f(x: Float[Array, \"b d\"] | None): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_two_string_literals_uses_first,
        "def f(x: Annotated[Float[Array, \"b d\"], \"meta\"]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_shape_with_comma_kept_as_token,
        "def f(x: Float[Array, \"b, d\"]): pass",
        [("x", ["b,", "d"])]
    );
    annotation_case!(
        ann_newline_inside_shape_splits_whitespace,
        "def f(x: Float[Array, \"b\\nd\"]): pass",
        [("x", ["b\\nd"])]
    );
    annotation_case!(
        ann_tab_inside_shape_splits_whitespace,
        "def f(x: Float[Array, \"b\td\"]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_uppercase_raw_prefix,
        "def f(x: Float[Array, R\"b d\"]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_unicode_raw_prefix,
        "def f(x: Float[Array, ur\"b d\"]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_capital_f_string_rejected,
        "def f(x: Float[Array, F\"b {d}\"]): pass",
        []
    );
    annotation_case!(
        ann_capital_b_string_rejected,
        "def f(x: Float[Array, B\"b d\"]): pass",
        []
    );
    annotation_case!(
        ann_self_annotated_is_extracted_if_annotated,
        "class C:\n    def f(self: Float[Array, \"b d\"]): pass",
        [("self", ["b", "d"])]
    );
    annotation_case!(
        ann_cls_annotated_is_extracted_if_annotated,
        "class C:\n    def f(cls: Float[Array, \"b d\"]): pass",
        [("cls", ["b", "d"])]
    );
    annotation_case!(
        ann_tuple_wrapped_array_shape,
        "def f(x: tuple[Float[Array, \"b d\"], int]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_dict_wrapped_array_shape,
        "def f(x: dict[str, Float[Array, \"b d\"]]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_typing_optional_wrapped_array_shape,
        "def f(x: typing.Optional[Float[Array, \"b d\"]]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_union_uses_first_shape_string,
        "def f(x: Union[Float[Array, \"b d\"], Float[Array, \"b e\"]]): pass",
        [("x", ["b", "d"])]
    );
    annotation_case!(
        ann_underscore_dimension_name,
        "def f(x: Float[Array, \"batch _features\"]): pass",
        [("x", ["batch", "_features"])]
    );
    annotation_case!(
        ann_question_mark_dimension_name_preserved,
        "def f(x: Float[Array, \"batch features?\"]): pass",
        [("x", ["batch", "features?"])]
    );
    annotation_case!(
        ann_ellipsis_dimension_name_preserved,
        "def f(x: Float[Array, \"batch ...\"]): pass",
        [("x", ["batch", "..."])]
    );
    annotation_case!(
        ann_colon_dimension_name_preserved,
        "def f(x: Float[Array, \"batch time:2\"]): pass",
        [("x", ["batch", "time:2"])]
    );

    apply_ok_case!(
        apply_rank_five_symbolic,
        &["a", "b", "c", "d", "features"],
        linear("features", "out"),
        &["a", "b", "c", "d", "out"]
    );
    apply_ok_case!(
        apply_comma_dimension_exact_match,
        &["batch", "features,"],
        linear("features,", "out"),
        &["batch", "out"]
    );
    apply_ok_case!(
        apply_question_mark_dimension_exact_match,
        &["batch", "features?"],
        linear("features?", "out"),
        &["batch", "out"]
    );
    apply_ok_case!(
        apply_ellipsis_dimension_exact_match,
        &["batch", "..."],
        linear("...", "out"),
        &["batch", "out"]
    );
    apply_err_case!(
        apply_comma_dimension_mismatch,
        &["batch", "features"],
        linear("features,", "out"),
        "got features"
    );
    apply_err_case!(
        apply_question_mark_dimension_mismatch,
        &["batch", "features"],
        linear("features?", "out"),
        "got features"
    );
    apply_err_case!(
        apply_ellipsis_dimension_mismatch,
        &["batch", "features"],
        linear("...", "out"),
        "got features"
    );
    apply_err_case!(
        apply_colon_dimension_mismatch,
        &["batch", "time"],
        linear("time:2", "out"),
        "got time"
    );

    macro_rules! classify_case {
        ($name:ident, $module:expr, $owner:expr, $call_name:expr, [$(($key:expr, $value:expr)),*], $expected:expr) => {
            #[test]
            fn $name() {
                let bindings = HashMap::from([
                    $(($key.to_string(), $value.to_string()),)*
                ]);
                let call = ResolvedCallSignature {
                    implementation: ResolvedImplementation {
                        target: ResolvedModuleTarget {
                            dots: 0,
                            module_parts: parts($module),
                            file_path: PathBuf::from("unused.py"),
                            symbol_parts: Vec::new(),
                        },
                        symbol: $owner.map(|owner| PythonSymbol::Class {
                            name: owner.to_string(),
                        }),
                    },
                    signature: PythonCallableSignature {
                        owner: $owner.map(|owner| owner.to_string()),
                        name: $call_name.to_string(),
                        params: Vec::new(),
                    },
                    arguments: Vec::new(),
                    bindings,
                };
                assert_eq!(classify_layer_call(&call), $expected);
            }
        };
    }

    classify_case!(
        classify_exact_equinox_nn_init,
        &["equinox", "nn"],
        Some("Linear"),
        "__init__",
        [("in_features", "1"), ("out_features", "2")],
        Some(linear("1", "2"))
    );
    classify_case!(
        classify_equinox_nn_deep_init,
        &["equinox", "nn", "_linear"],
        Some("Linear"),
        "__init__",
        [("in_features", "1"), ("out_features", "2")],
        Some(linear("1", "2"))
    );
    classify_case!(
        classify_case_sensitive_module_rejected,
        &["Equinox", "nn"],
        Some("Linear"),
        "__init__",
        [("in_features", "1"), ("out_features", "2")],
        None
    );
    classify_case!(
        classify_case_sensitive_owner_rejected,
        &["equinox", "nn"],
        Some("linear"),
        "__init__",
        [("in_features", "1"), ("out_features", "2")],
        None
    );
    classify_case!(
        classify_missing_bindings_rejected,
        &["equinox", "nn"],
        Some("Linear"),
        "__init__",
        [],
        None
    );
    classify_case!(
        classify_wrong_function_name_rejected,
        &["equinox", "nn"],
        Some("Linear"),
        "__call__",
        [("in_features", "1"), ("out_features", "2")],
        None
    );
}

#[cfg(test)]
mod import_map_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn ip(dots: usize, module: &[&str], name: &str) -> ImportPath {
        ImportPath {
            dots,
            module: module.iter().map(|s| s.to_string()).collect(),
            name: name.to_string(),
        }
    }

    #[test]
    fn test_plain_import() {
        let code = "import jax";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jax"), Some(&ip(0, &[], "jax")));
    }

    #[test]
    fn test_plain_dotted_import() {
        let code = "import jax.numpy";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jax"), Some(&ip(0, &["jax"], "numpy")));
    }

    #[test]
    fn test_plain_import_with_alias() {
        let code = "import jax.numpy as jnp";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jnp"), Some(&ip(0, &["jax"], "numpy")));
        assert_eq!(map.get("jax"), None);
    }

    #[test]
    fn test_plain_import_deeper_with_alias() {
        let code = "import equinox.nn as nn";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("nn"), Some(&ip(0, &["equinox"], "nn")));
    }

    #[test]
    fn test_from_import_single() {
        let code = "from jax import random";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("random"), Some(&ip(0, &["jax"], "random")));
    }

    #[test]
    fn test_from_import_multiple() {
        let code = "from jaxtyping import Float, Array";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("Float"), Some(&ip(0, &["jaxtyping"], "Float")));
        assert_eq!(map.get("Array"), Some(&ip(0, &["jaxtyping"], "Array")));
    }

    #[test]
    fn test_from_import_with_alias() {
        let code = "from jaxtyping import Array as Arr";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("Arr"), Some(&ip(0, &["jaxtyping"], "Array")));
        assert_eq!(map.get("Array"), None);
    }

    #[test]
    fn test_from_import_multiple_mixed_aliases() {
        let code = "from mypackage import transform, MyLinear as ML, helper";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(
            map.get("transform"),
            Some(&ip(0, &["mypackage"], "transform"))
        );
        assert_eq!(map.get("ML"), Some(&ip(0, &["mypackage"], "MyLinear")));
        assert_eq!(map.get("helper"), Some(&ip(0, &["mypackage"], "helper")));
        assert_eq!(map.get("MyLinear"), None);
    }

    #[test]
    fn test_from_import_deeply_nested() {
        let code = "from google.cloud.storage.bucket import Bucket as GCSBucket";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(
            map.get("GCSBucket"),
            Some(&ip(0, &["google", "cloud", "storage", "bucket"], "Bucket"))
        );
    }

    #[test]
    fn test_relative_import_dot() {
        let code = "from . import utils";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("utils"), Some(&ip(1, &[], "utils")));
    }

    #[test]
    fn test_relative_import_dot_with_path() {
        let code = "from .layers import MyLinear";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("MyLinear"), Some(&ip(1, &["layers"], "MyLinear")));
    }

    #[test]
    fn test_relative_import_double_dot() {
        let code = "from ..utils import helper as h";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("h"), Some(&ip(2, &["utils"], "helper")));
    }

    #[test]
    fn test_relative_import_triple_dot() {
        let code = "from ...core.base import BaseModel";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(
            map.get("BaseModel"),
            Some(&ip(3, &["core", "base"], "BaseModel"))
        );
    }

    #[test]
    fn test_relative_import_dot_only_multiple() {
        let code = "from . import utils, helpers";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("utils"), Some(&ip(1, &[], "utils")));
        assert_eq!(map.get("helpers"), Some(&ip(1, &[], "helpers")));
    }

    #[test]
    fn test_comma_separated_plain_imports() {
        let code = "import sys, os";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("sys"), Some(&ip(0, &[], "sys")));
        assert_eq!(map.get("os"), Some(&ip(0, &[], "os")));
    }

    #[test]
    fn test_star_import_skipped() {
        let code = "from os.path import *";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_empty_file() {
        let code = "";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_no_imports() {
        let code = "x = 1\ny = 2";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_mixed_imports() {
        let code = r#"
import jax
import equinox as eqx
from jaxtyping import Float, Array
from mypackage.layers import MyLinear as ML
from . import utils
from ..core import Base
"#;
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jax"), Some(&ip(0, &[], "jax")));
        assert_eq!(map.get("eqx"), Some(&ip(0, &[], "equinox")));
        assert_eq!(map.get("Float"), Some(&ip(0, &["jaxtyping"], "Float")));
        assert_eq!(map.get("Array"), Some(&ip(0, &["jaxtyping"], "Array")));
        assert_eq!(
            map.get("ML"),
            Some(&ip(0, &["mypackage", "layers"], "MyLinear"))
        );
        assert_eq!(map.get("utils"), Some(&ip(1, &[], "utils")));
        assert_eq!(map.get("Base"), Some(&ip(2, &["core"], "Base")));
        assert_eq!(map.len(), 7);
    }

    #[test]
    fn test_relative_import_with_alias() {
        let code = "from . import utils as u";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("u"), Some(&ip(1, &[], "utils")));
        assert_eq!(map.get("utils"), None);
    }

    #[test]
    fn test_relative_import_with_path_and_alias() {
        let code = "from .layers import MyLinear as ML";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("ML"), Some(&ip(1, &["layers"], "MyLinear")));
        assert_eq!(map.get("MyLinear"), None);
    }

    #[test]
    fn test_deeply_nested_plain_import() {
        let code = "import a.b.c.d";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("a"), Some(&ip(0, &["a", "b", "c"], "d")));
    }

    #[test]
    fn test_comma_separated_aliased_imports() {
        let code = "import jax.numpy as jnp, equinox as eqx";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jnp"), Some(&ip(0, &["jax"], "numpy")));
        assert_eq!(map.get("eqx"), Some(&ip(0, &[], "equinox")));
    }

    #[test]
    fn test_parenthesized_from_import() {
        let code = "from jax import (\n    random,\n    numpy\n)";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("random"), Some(&ip(0, &["jax"], "random")));
        assert_eq!(map.get("numpy"), Some(&ip(0, &["jax"], "numpy")));
    }

    #[test]
    fn test_duplicate_import_last_wins() {
        let code = "import jax\nimport jax.numpy as jax";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("jax"), Some(&ip(0, &["jax"], "numpy")));
    }

    #[test]
    fn test_from_import_single_segment_module() {
        let code = "from os import path, getcwd, listdir";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("path"), Some(&ip(0, &["os"], "path")));
        assert_eq!(map.get("getcwd"), Some(&ip(0, &["os"], "getcwd")));
        assert_eq!(map.get("listdir"), Some(&ip(0, &["os"], "listdir")));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_backslash_continuation_import() {
        let code = "from jax \\\n    import random";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("random"), Some(&ip(0, &["jax"], "random")));
    }

    #[test]
    fn test_multiple_relative_aliased_imports() {
        let code = "from . import utils as u, helpers as h";
        let tree = parse(code);
        let map = build_import_map(tree.root_node(), code).unwrap();
        assert_eq!(map.get("u"), Some(&ip(1, &[], "utils")));
        assert_eq!(map.get("h"), Some(&ip(1, &[], "helpers")));
        assert_eq!(map.get("utils"), None);
        assert_eq!(map.get("helpers"), None);
    }
}

#[cfg(test)]
mod extract_calls_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn args_text<'a>(text: &'a str, range: &tree_sitter::Range) -> &'a str {
        &text[range.start_byte..range.end_byte]
    }

    #[test]
    fn test_simple_identifier_call() {
        let code = "x = foo()";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "x");
        assert_eq!(calls[0].target, "foo");
        assert_eq!(args_text(code, &calls[0].args_node_range), "()");
    }

    #[test]
    fn test_identifier_call_with_args() {
        let code = "y = transform(x)";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "y");
        assert_eq!(calls[0].target, "transform");
        assert_eq!(args_text(code, &calls[0].args_node_range), "(x)");
    }

    #[test]
    fn test_attribute_call() {
        let code = "x = jnp.zeros((32, 64))";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "x");
        assert_eq!(calls[0].target, "jnp.zeros");
        assert_eq!(args_text(code, &calls[0].args_node_range), "((32, 64))");
    }

    #[test]
    fn test_deep_attribute_call() {
        let code = "layer = eqx.nn.Linear(128, 64)";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "layer");
        assert_eq!(calls[0].target, "eqx.nn.Linear");
        assert_eq!(args_text(code, &calls[0].args_node_range), "(128, 64)");
    }

    #[test]
    fn test_multiple_calls() {
        let code = "x = foo()\ny = bar(1, 2)\nz = jnp.zeros((3,))";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].variable, "x");
        assert_eq!(calls[0].target, "foo");
        assert_eq!(calls[1].variable, "y");
        assert_eq!(calls[1].target, "bar");
        assert_eq!(calls[2].variable, "z");
        assert_eq!(calls[2].target, "jnp.zeros");
    }

    #[test]
    fn test_skips_assignment_without_call() {
        let code = "x = 5\ny = foo()";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "y");
    }

    #[test]
    fn test_skips_bare_call_without_assignment() {
        let code = "foo()\nx = bar()";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "x");
    }

    #[test]
    fn test_skips_binary_op_assignment() {
        let code = "x = a + b\ny = foo()";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "y");
    }

    #[test]
    fn test_nested_call_captures_outer_only() {
        let code = "x = foo(bar())";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "x");
        assert_eq!(calls[0].target, "foo");
    }

    #[test]
    fn test_call_with_kwargs() {
        let code = "layer = eqx.nn.Linear(128, 64, key=key)";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].target, "eqx.nn.Linear");
        assert_eq!(
            args_text(code, &calls[0].args_node_range),
            "(128, 64, key=key)"
        );
    }

    #[test]
    fn test_empty_file() {
        let code = "";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_no_calls() {
        let code = "x = 5\ny = 'hello'\nz = [1, 2, 3]";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_inside_function() {
        let code = r#"
def forward(x):
    y = jnp.matmul(x, w)
    z = transform(y)
"#;
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].target, "jnp.matmul");
        assert_eq!(calls[1].target, "transform");
    }

    #[test]
    fn test_mixed_module_and_function_level() {
        let code = r#"
x = jnp.zeros((32,))
def forward(a):
    b = transform(a)
y = relu(x)
"#;
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].target, "jnp.zeros");
        assert_eq!(calls[1].target, "transform");
        assert_eq!(calls[2].target, "relu");
    }

    #[test]
    fn test_skips_tuple_unpack_assignment() {
        let code = "a, b = foo()\nx = bar()";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "x");
    }

    #[test]
    fn test_skips_attribute_assignment() {
        let code = "self.x = foo()\ny = bar()";
        let tree = parse(code);
        let calls = extract_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "y");
    }
}
