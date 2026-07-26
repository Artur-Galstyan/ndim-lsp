use std::{collections::HashMap, path::PathBuf};

use tree_sitter::Node;

#[cfg(test)]
use tree_sitter::Range;

use crate::python_ast::{
    build_import_map, extract_call_arguments, extract_calls, extract_self_attr_calls,
};
use crate::resolution::{
    ResolutionCache, bind_call_arguments, resolve_call_signature, resolve_call_target,
};
use crate::types::*;

pub fn classify_layer_call(call: &ResolvedCallSignature) -> Option<LayerKind> {
    let is_equinox_module = call.implementation.target.module_parts.len() >= 2
        && call.implementation.target.module_parts[0] == "equinox"
        && call.implementation.target.module_parts[1] == "nn";
    let is_torch_module = call.implementation.target.module_parts.len() >= 2
        && call.implementation.target.module_parts[0] == "torch"
        && call.implementation.target.module_parts[1] == "nn";
    let is_flax_module = call.implementation.target.module_parts.len() >= 2
        && call.implementation.target.module_parts[0] == "flax"
        && call.implementation.target.module_parts[1] == "linen";

    if !is_equinox_module && !is_torch_module && !is_flax_module {
        return None;
    }

    let owner = call.signature.owner.as_deref()?;
    if call.signature.name != "__init__" {
        return None;
    }

    match owner {
        "Linear" => Some(LayerKind::Linear {
            in_features: call.bindings.get("in_features")?.clone(),
            out_features: call.bindings.get("out_features")?.clone(),
        }),
        "Conv1d" | "Conv2d" | "Conv3d" => {
            // Per-axis tuples (e.g. kernel_size=(3,5)) are not yet supported.
            // Detect and refuse to classify so we don't produce garbage symbolic output.
            let kernel_size = call.bindings.get("kernel_size")?;
            if kernel_size.starts_with('(') {
                return None;
            }
            let stride = call
                .bindings
                .get("stride")
                .cloned()
                .unwrap_or_else(|| "1".to_string());
            if stride.starts_with('(') {
                return None;
            }
            let padding = call
                .bindings
                .get("padding")
                .cloned()
                .unwrap_or_else(|| "0".to_string());
            if padding.starts_with('(') {
                return None;
            }
            let in_channels = call.bindings.get("in_channels")?.clone();
            let out_channels = call.bindings.get("out_channels")?.clone();
            let ks = kernel_size.clone();
            match owner {
                "Conv1d" => Some(LayerKind::Conv1d {
                    in_channels,
                    out_channels,
                    kernel_size: ks,
                    stride,
                    padding,
                }),
                "Conv2d" => Some(LayerKind::Conv2d {
                    in_channels,
                    out_channels,
                    kernel_size: ks,
                    stride,
                    padding,
                }),
                "Conv3d" => Some(LayerKind::Conv3d {
                    in_channels,
                    out_channels,
                    kernel_size: ks,
                    stride,
                    padding,
                }),
                _ => None,
            }
        }
        "Dense" => Some(LayerKind::Dense {
            features: call.bindings.get("features")?.clone(),
        }),
        "Conv" if is_flax_module => {
            let features = call.bindings.get("features")?.clone();
            let kernel_size = call.bindings.get("kernel_size")?;
            // Spatial rank from the kernel tuple: (3, 3) → 2, scalar → 1.
            let spatial_rank = if kernel_size.starts_with('(') {
                kernel_size.matches(',').count() + usize::from(!kernel_size.ends_with(",)"))
            } else {
                1
            };
            // v1 models only the default stride-1 / SAME-padding case.
            if let Some(strides) = call.bindings.get("strides")
                && strides.trim_matches(['(', ')', ' ']).split(',').any(|s| {
                    let s = s.trim();
                    !s.is_empty() && s != "1"
                })
            {
                return None;
            }
            Some(LayerKind::FlaxConv {
                features,
                spatial_rank,
            })
        }
        "MaxPool1d" | "MaxPool2d" | "MaxPool3d" | "AvgPool1d" | "AvgPool2d" | "AvgPool3d" => {
            let spatial_rank = pool_spatial_rank(owner)?;
            let kernel_size = call.bindings.get("kernel_size")?.clone();
            if kernel_size.starts_with('(') {
                return None;
            }
            // torch default: stride = kernel_size. (equinox defaults to
            // stride=1 — divergence accepted until it bites.)
            let stride = call
                .bindings
                .get("stride")
                .cloned()
                .unwrap_or_else(|| kernel_size.clone());
            if stride.starts_with('(') {
                return None;
            }
            let padding = call
                .bindings
                .get("padding")
                .cloned()
                .unwrap_or_else(|| "0".to_string());
            if padding.starts_with('(') {
                return None;
            }
            Some(LayerKind::Pool {
                name: owner.to_string(),
                spatial_rank,
                kernel_size,
                stride,
                padding,
            })
        }
        "AdaptiveMaxPool1d" | "AdaptiveMaxPool2d" | "AdaptiveMaxPool3d" | "AdaptiveAvgPool1d"
        | "AdaptiveAvgPool2d" | "AdaptiveAvgPool3d" => {
            let spatial_rank = pool_spatial_rank(owner)?;
            let output_size = call.bindings.get("output_size")?.clone();
            if output_size.starts_with('(') {
                return None;
            }
            Some(LayerKind::AdaptivePool {
                name: owner.to_string(),
                spatial_rank,
                output_size,
            })
        }
        // torch: MultiheadAttention(embed_dim, num_heads). equinox's real
        // signature is MultiheadAttention(num_heads, query_size, ...) — a
        // different positional order *and* a different name for the
        // per-token dimension, so it must bind `query_size`, not `embed_dim`
        // (previously both frameworks read `embed_dim`, which silently
        // failed to classify equinox ctors that only bind positionally).
        "MultiheadAttention" => {
            let feature_dim = if is_equinox_module {
                call.bindings.get("query_size")?
            } else {
                call.bindings.get("embed_dim")?
            };
            Some(LayerKind::MultiheadAttention {
                feature_dim: feature_dim.clone(),
            })
        }
        // equinox names it `embedding_size`, torch `embedding_dim`.
        "Embedding" => Some(LayerKind::Embedding {
            embedding_size: call
                .bindings
                .get("embedding_size")
                .or_else(|| call.bindings.get("embedding_dim"))?
                .clone(),
        }),
        // Shape-preserving layers (Dropout, BatchNorm, LayerNorm, GroupNorm, activations,
        // Identity, InstanceNorm, RMSNorm, AlphaDropout, and the two Transformer*Layer
        // modules — TransformerEncoderLayer/TransformerDecoderLayer preserve d_model on
        // their primary input; TransformerDecoderLayer's second (`memory`) input isn't
        // tracked, same single-input limitation as Bilinear/CosineSimilarity below).
        "Dropout" | "Dropout1d" | "Dropout2d" | "Dropout3d" | "BatchNorm" | "BatchNorm1d"
        | "BatchNorm2d" | "BatchNorm3d" | "LayerNorm" | "GroupNorm" | "ReLU" | "GELU"
        | "Sigmoid" | "Tanh" | "Softmax" | "PReLU" | "Identity" | "InstanceNorm1d"
        | "InstanceNorm2d" | "InstanceNorm3d" | "RMSNorm" | "AlphaDropout"
        | "TransformerEncoderLayer" | "TransformerDecoderLayer" => {
            Some(LayerKind::ShapePreserving {
                name: owner.to_string(),
            })
        }
        "Flatten" => {
            let start_dim = call
                .bindings
                .get("start_dim")
                .cloned()
                .unwrap_or_else(|| "1".to_string());
            let end_dim = call
                .bindings
                .get("end_dim")
                .cloned()
                .unwrap_or_else(|| "-1".to_string());
            Some(LayerKind::Flatten { start_dim, end_dim })
        }
        "Unflatten" => Some(LayerKind::Unflatten {
            dim: call.bindings.get("dim")?.clone(),
            sizes: call.bindings.get("sizes")?.clone(),
        }),
        "Upsample" => {
            let scale_factor = call.bindings.get("scale_factor").cloned();
            let size = call.bindings.get("size").cloned();
            Some(LayerKind::Upsample { scale_factor, size })
        }
        "ConvTranspose1d" | "ConvTranspose2d" | "ConvTranspose3d" => {
            let spatial_rank = pool_spatial_rank(owner)?;
            let kernel_size = call.bindings.get("kernel_size")?.clone();
            if kernel_size.starts_with('(') {
                return None;
            }
            let stride = call
                .bindings
                .get("stride")
                .cloned()
                .unwrap_or_else(|| "1".to_string());
            if stride.starts_with('(') {
                return None;
            }
            let padding = call
                .bindings
                .get("padding")
                .cloned()
                .unwrap_or_else(|| "0".to_string());
            if padding.starts_with('(') {
                return None;
            }
            Some(LayerKind::ConvTranspose {
                spatial_rank,
                in_channels: call.bindings.get("in_channels")?.clone(),
                out_channels: call.bindings.get("out_channels")?.clone(),
                kernel_size,
                stride,
                padding,
            })
        }
        "RNN" | "LSTM" | "GRU" => Some(LayerKind::Rnn {
            name: owner.to_string(),
            input_size: call.bindings.get("input_size")?.clone(),
            hidden_size: call.bindings.get("hidden_size")?.clone(),
        }),
        "RNNCell" | "LSTMCell" | "GRUCell" => Some(LayerKind::RnnCell {
            name: owner.to_string(),
            input_size: call.bindings.get("input_size")?.clone(),
            hidden_size: call.bindings.get("hidden_size")?.clone(),
        }),
        "PixelShuffle" => Some(LayerKind::PixelShuffle {
            upscale_factor: call.bindings.get("upscale_factor")?.clone(),
        }),
        "PixelUnshuffle" => Some(LayerKind::PixelUnshuffle {
            downscale_factor: call.bindings.get("downscale_factor")?.clone(),
        }),
        "ConstantPad1d" | "ConstantPad2d" | "ConstantPad3d" | "ZeroPad1d" | "ZeroPad2d"
        | "ZeroPad3d" | "ReflectionPad1d" | "ReflectionPad2d" | "ReflectionPad3d"
        | "ReplicationPad1d" | "ReplicationPad2d" | "ReplicationPad3d" => {
            let spatial_rank = pool_spatial_rank(owner)?;
            Some(LayerKind::Pad {
                name: owner.to_string(),
                spatial_rank,
                padding: call.bindings.get("padding")?.clone(),
            })
        }
        "Bilinear" => Some(LayerKind::Bilinear {
            in1_features: call.bindings.get("in1_features")?.clone(),
            in2_features: call.bindings.get("in2_features")?.clone(),
            out_features: call.bindings.get("out_features")?.clone(),
        }),
        "CosineSimilarity" => Some(LayerKind::CosineSimilarity {
            dim: call
                .bindings
                .get("dim")
                .cloned()
                .unwrap_or_else(|| "1".to_string()),
        }),
        "MLP" if is_equinox_module => Some(LayerKind::Mlp {
            in_size: call.bindings.get("in_size")?.clone(),
            out_size: call.bindings.get("out_size")?.clone(),
        }),
        _ => None,
    }
}

/// Spatial rank from a pooling class name's `1d`/`2d`/`3d` suffix.
fn pool_spatial_rank(name: &str) -> Option<usize> {
    name.strip_suffix('d')?
        .chars()
        .last()?
        .to_digit(10)
        .map(|d| d as usize)
}

/// Built-in catalog of framework layer constructors.
///
/// Returns a `PythonCallableSignature` for `equinox.nn.<X>` and `torch.nn.<X>`
/// layers whose constructor params are well-known. This short-circuits disk
/// resolution for the common case so the analyzer still classifies layers when
/// the framework's source isn't on `search_roots`.
///
/// `parts` is the import-resolved call path, e.g. `["equinox", "nn", "Linear"]`
/// or `["torch", "nn", "Conv2d"]`. Only the last element is used as the class
/// name; `parts[..parts.len()-1]` becomes `module_parts` when the caller
/// synthesizes a `ResolvedCallSignature`.
///
/// NOTE: The `params` lists below are an **analyzer-internal binding contract**,
/// not a faithful reproduction of the upstream constructor signatures. They
/// exist only so `bind_call_arguments` can map positional/keyword arguments to
/// the names that `classify_layer_call` reads (`in_features`, `out_features`,
/// `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`). Extra
/// trailing params (e.g. `dilation`, `groups`, `use_bias`) are present only to
/// absorb additional positional args without binding them to the wrong name —
/// they intentionally diverge from `torch.nn.Conv2d` (which has `padding_mode`
/// between `groups` and `bias`) and from `equinox.nn.Conv` (which has `key`
/// and other extras). Do not "fix" these to match the upstream signatures
/// unless `classify_layer_call` starts reading those fields.
fn known_layer_signature(parts: &[String]) -> Option<PythonCallableSignature> {
    if parts.len() < 3 {
        return None;
    }
    let framework = parts[0].as_str();
    let module = parts[1].as_str();
    let known_framework = (module == "nn" && (framework == "equinox" || framework == "torch"))
        || (module == "linen" && framework == "flax");
    if !known_framework {
        return None;
    }
    let class_name = parts.last()?.as_str();
    let params: &[&str] = match class_name {
        "Dense" => &["self", "features", "use_bias"],
        "Conv" if framework == "flax" => {
            &["self", "features", "kernel_size", "strides", "padding"]
        }
        "Linear" => &["self", "in_features", "out_features", "use_bias"],
        "Embedding" => &["self", "num_embeddings", "embedding_size"],
        "MaxPool1d" | "MaxPool2d" | "MaxPool3d" | "AvgPool1d" | "AvgPool2d" | "AvgPool3d" => {
            &["self", "kernel_size", "stride", "padding"]
        }
        "AdaptiveMaxPool1d" | "AdaptiveMaxPool2d" | "AdaptiveMaxPool3d" | "AdaptiveAvgPool1d"
        | "AdaptiveAvgPool2d" | "AdaptiveAvgPool3d" => &["self", "output_size"],
        // equinox.nn.MultiheadAttention(num_heads, query_size, key_size=None,
        // value_size=None, output_size=None, ...) — reversed order and
        // different names vs. torch's (embed_dim, num_heads).
        "MultiheadAttention" if framework == "equinox" => &[
            "self",
            "num_heads",
            "query_size",
            "key_size",
            "value_size",
            "output_size",
        ],
        "MultiheadAttention" => &["self", "embed_dim", "num_heads"],
        "Conv1d" | "Conv2d" | "Conv3d" => &[
            "self",
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "use_bias",
        ],
        "Dropout" | "Dropout1d" | "Dropout2d" | "Dropout3d" | "BatchNorm" | "BatchNorm1d"
        | "BatchNorm2d" | "BatchNorm3d" | "LayerNorm" | "GroupNorm" | "ReLU" | "GELU"
        | "Sigmoid" | "Tanh" | "Softmax" | "PReLU" | "Identity" | "InstanceNorm1d"
        | "InstanceNorm2d" | "InstanceNorm3d" | "RMSNorm" | "AlphaDropout"
        | "TransformerEncoderLayer" | "TransformerDecoderLayer" => &["self"],
        "Flatten" => &["self", "start_dim", "end_dim"],
        "Unflatten" => &["self", "dim", "sizes"],
        "Upsample" => &["self", "size", "scale_factor", "mode", "align_corners"],
        // Shares the first 5 positions (in/out channels, kernel_size, stride,
        // padding) with Conv1d/2d/3d; torch and equinox diverge on the order
        // of `groups`/`dilation`/`bias` afterwards, which we don't read, so
        // the trailing placeholder names only need to absorb positions.
        "ConvTranspose1d" | "ConvTranspose2d" | "ConvTranspose3d" => &[
            "self",
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "groups_or_dilation",
            "dilation_or_groups",
            "use_bias",
        ],
        // torch/equinox agree on (input_size, hidden_size, ...) for all six
        // of these; only the first two are ever read.
        "RNN" | "LSTM" | "GRU" | "RNNCell" | "LSTMCell" | "GRUCell" => &[
            "self",
            "input_size",
            "hidden_size",
            "num_layers_or_bias",
        ],
        "PixelShuffle" => &["self", "upscale_factor"],
        "PixelUnshuffle" => &["self", "downscale_factor"],
        "ConstantPad1d" | "ConstantPad2d" | "ConstantPad3d" => &["self", "padding", "value"],
        "ZeroPad1d" | "ZeroPad2d" | "ZeroPad3d" | "ReflectionPad1d" | "ReflectionPad2d"
        | "ReflectionPad3d" | "ReplicationPad1d" | "ReplicationPad2d" | "ReplicationPad3d" => {
            &["self", "padding"]
        }
        "Bilinear" => &["self", "in1_features", "in2_features", "out_features", "use_bias"],
        "CosineSimilarity" => &["self", "dim", "eps"],
        "MLP" if framework == "equinox" => &[
            "self",
            "in_size",
            "out_size",
            "width_size",
            "depth",
        ],
        _ => return None,
    };

    Some(PythonCallableSignature {
        owner: Some(class_name.to_string()),
        name: "__init__".to_string(),
        params: params.iter().map(|s| s.to_string()).collect(),
    })
}

/// Try the built-in layer catalog for a given call. Returns a synthesized
/// `ResolvedCallSignature` whose module_parts/owner/__init__ match what
/// `classify_layer_call` expects, with no filesystem I/O.
fn try_catalog_signature(
    call: &CallInfo,
    node: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
) -> Result<Option<ResolvedCallSignature>, String> {
    let target = resolve_call_target(&call.target, import_map);
    if target.dots > 0 {
        return Ok(None);
    }
    let Some(signature) = known_layer_signature(&target.parts) else {
        return Ok(None);
    };

    let Some(args_node) = node.descendant_for_byte_range(
        call.args_node_range.start_byte,
        call.args_node_range.end_byte,
    ) else {
        return Ok(None);
    };
    let arguments = extract_call_arguments(args_node, text)?;
    let bindings = bind_call_arguments(&signature, &arguments);

    let class_name = signature.owner.clone().unwrap_or_default();
    let module_parts: Vec<String> = target.parts[..target.parts.len() - 1].to_vec();

    Ok(Some(ResolvedCallSignature {
        implementation: ResolvedImplementation {
            target: ResolvedModuleTarget {
                dots: 0,
                module_parts,
                file_path: PathBuf::new(),
                symbol_parts: vec![class_name.clone()],
            },
            symbol: Some(PythonSymbol::Class { name: class_name }),
        },
        signature,
        arguments,
        bindings,
    }))
}

/// Classify a bare constructor call node (e.g. the `nn.Dense(features=64)` in
/// `nn.Dense(features=64)(x)`) against the built-in catalog. No disk I/O —
/// the inline idiom is only supported for catalogued framework layers.
pub fn classify_inline_constructor(
    ctor_call: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
) -> Option<LayerKind> {
    let func = ctor_call.child_by_field_name("function")?;
    let target = func.utf8_text(text.as_bytes()).ok()?;
    let args_node = ctor_call.child_by_field_name("arguments")?;
    let call = CallInfo {
        variable: String::new(),
        target: target.to_string(),
        args_node_range: args_node.range(),
    };
    let resolved = try_catalog_signature(&call, ctor_call, text, import_map).ok()??;
    classify_layer_call(&resolved)
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
    let records = extract_layer_assignments_scoped(
        node,
        text,
        &import_map,
        search_roots,
        read_file,
        max_depth,
        None,
    )?;
    let mut layers = HashMap::new();
    for rec in records {
        layers.insert(rec.name, rec.kind);
    }
    Ok(layers)
}

/// Catalog-first / disk-fallback resolution of a single layer constructor
/// call to its `LayerKind`, or `None` if the call isn't a recognised layer.
#[allow(clippy::too_many_arguments)]
fn resolve_layer_kind_for_call<F>(
    call: &CallInfo,
    node: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
    search_roots: &[PathBuf],
    read_file: &F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<Option<LayerKind>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    // Catalog-first: hardcoded equinox.nn.* / torch.nn.* signatures bypass
    // disk resolution. Falls through to resolve_call_signature for
    // user-defined layers and frameworks not in the catalog.
    let resolved_call = match try_catalog_signature(call, node, text, import_map)? {
        Some(c) => Some(c),
        None => resolve_call_signature(
            call,
            text,
            import_map,
            search_roots,
            read_file,
            max_depth,
            cache,
        )?,
    };
    let Some(resolved_call) = resolved_call else {
        return Ok(None);
    };
    Ok(classify_layer_call(&resolved_call))
}

pub fn extract_layer_assignments_scoped<F>(
    node: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<Vec<LayerAssignment>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let calls = extract_calls(node, text)?;
    let mut records = Vec::new();

    for call in calls {
        let Some(layer) = resolve_layer_kind_for_call(
            &call,
            node,
            text,
            import_map,
            search_roots,
            &read_file,
            max_depth,
            cache,
        )?
        else {
            continue;
        };
        records.push(LayerAssignment {
            name: call.variable,
            kind: layer,
            byte_position: call.args_node_range.start_byte,
        });
    }

    Ok(records)
}

/// Resolve `self.<attr> = <layer constructor>` assignments (typically in an
/// `__init__`) to a map of bare attribute name → `LayerKind`. Mirrors the
/// catalog-first / disk-fallback resolution of `extract_layer_assignments_scoped`.
/// Flat last-wins view of `extract_self_attr_layers_by_class`.
pub fn extract_self_attr_layers<F>(
    node: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<HashMap<String, LayerKind>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let by_class =
        extract_self_attr_layers_by_class(node, text, import_map, search_roots, read_file, max_depth, cache)?;
    Ok(by_class
        .into_iter()
        .filter_map(|(attr, mut entries)| entries.pop().map(|e| (attr, e.kind)))
        .collect())
}

/// Like `extract_self_attr_layers`, but each binding keeps the byte range of
/// its enclosing `class_definition`, so `self.fc` in class A and `self.fc` in
/// class B resolve independently at their respective call sites.
pub fn extract_self_attr_layers_by_class<F>(
    node: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<HashMap<String, Vec<ScopedSelfAttrLayer>>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let calls = extract_self_attr_calls(node, text)?;
    let classes = class_ranges(node);
    let mut layers: HashMap<String, Vec<ScopedSelfAttrLayer>> = HashMap::new();

    for call in calls {
        let Some(layer) = resolve_layer_kind_for_call(
            &call,
            node,
            text,
            import_map,
            search_roots,
            &read_file,
            max_depth,
            cache,
        )?
        else {
            continue;
        };
        let (class_start, class_end) =
            enclosing_class_range(&classes, call.args_node_range.start_byte);
        layers.entry(call.variable).or_default().push(ScopedSelfAttrLayer {
            class_start,
            class_end,
            kind: layer,
        });
    }

    Ok(layers)
}

/// Byte ranges of every `class_definition` in the tree (DFS, includes nested).
fn class_ranges(node: Node) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        if n.kind() == "class_definition" {
            ranges.push((n.start_byte(), n.end_byte()));
        }
        for i in (0..n.child_count()).rev() {
            if let Some(c) = n.child(i as u32) {
                stack.push(c);
            }
        }
    }
    ranges
}

/// Innermost class range containing `byte`; whole file if none (module-level
/// `self` is malformed Python anyway, but stay permissive).
fn enclosing_class_range(classes: &[(usize, usize)], byte: usize) -> (usize, usize) {
    classes
        .iter()
        .filter(|(s, e)| *s <= byte && byte < *e)
        .min_by_key(|(s, e)| e - s)
        .copied()
        .unwrap_or((0, usize::MAX))
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

/// Apply a signed integer offset to a symbolic expression, constant-folding
/// when the expression itself is an integer.
fn apply_offset(expr: &str, offset: isize) -> String {
    if offset == 0 {
        return expr.to_string();
    }
    // If expr is itself an integer, fold the whole thing
    if let Ok(val) = expr.parse::<isize>() {
        return (val + offset).to_string();
    }
    if offset > 0 {
        format!("{}+{}", expr, offset)
    } else {
        format!("{}-{}", expr, -offset)
    }
}

/// Compute the output spatial dimension for a convolution.
/// Formula: floor((L + 2*padding - kernel_size) / stride) + 1
///
/// If all values and the spatial dim are integers, compute concretely.
/// Otherwise build a symbolic string with constant-folding where possible.
fn conv_spatial_dim(spatial_dim: &str, kernel_size: &str, stride: &str, padding: &str) -> String {
    // Try fully concrete computation
    if let (Ok(l), Ok(k), Ok(s), Ok(p)) = (
        spatial_dim.parse::<isize>(),
        kernel_size.parse::<isize>(),
        stride.parse::<isize>(),
        padding.parse::<isize>(),
    ) && s > 0
    {
        let l_out = (l + 2 * p - k) / s + 1;
        return l_out.to_string();
    }

    // Symbolic: (L + 2*p - k) / s + 1
    // Constant-fold 2*padding - kernel_size (and +1 when stride==1) into a
    // single offset whenever both padding and kernel_size are concrete ints.
    let stride_val: Option<isize> = stride.parse::<isize>().ok();
    let is_stride_one = stride_val == Some(1);

    if let (Ok(p), Ok(k)) = (padding.parse::<isize>(), kernel_size.parse::<isize>()) {
        // Fully concrete offset(s)
        if is_stride_one {
            // Total offset: 2*p - k + 1, folded into one operation
            apply_offset(spatial_dim, 2 * p - k + 1)
        } else {
            // Inner offset: 2*p - k, then divide by stride and add 1
            let inner = apply_offset(spatial_dim, 2 * p - k);
            if let Some(s) = stride_val {
                format!("({})/{}+1", inner, s)
            } else {
                format!("({})/{}+1", inner, stride)
            }
        }
    } else {
        // Build incrementally — at least one of padding/kernel_size is symbolic
        let mut inner = spatial_dim.to_string();

        // Add 2*padding
        if let Ok(p) = padding.parse::<isize>() {
            inner = apply_offset(&inner, 2 * p);
        } else {
            inner = format!("{}+2*{}", inner, padding);
        }

        // Subtract kernel_size
        if let Ok(k) = kernel_size.parse::<isize>() {
            inner = apply_offset(&inner, -k);
        } else {
            inner = format!("{}-{}", inner, kernel_size);
        }

        // Divide by stride and add 1
        if is_stride_one {
            apply_offset(&inner, 1)
        } else if let Some(s) = stride_val {
            format!("({})/{}+1", inner, s)
        } else {
            format!("({})/{}+1", inner, stride)
        }
    }
}

/// Shared shape-rule implementation for Conv1d / Conv2d / Conv3d.
///
/// All three share the same logic: check channel dim, compute each spatial
/// output dim via `conv_spatial_dim`. They differ only in `spatial_rank`.
///
/// Layout assumption: **channels-first** (PyTorch / Equinox convention):
///   Conv1d: [..., C, L]       spatial_rank=1
///   Conv2d: [..., C, H, W]    spatial_rank=2
///   Conv3d: [..., C, D, H, W] spatial_rank=3
///
/// This is the *opposite* of Flax/JAX-NN channel-last layout. When
/// `flax.linen.Conv` is added, a separate variant or layout field will be
/// needed.
/// Inverse of `conv_spatial_dim` for transposed/deconvolutions.
/// Formula: `out = (in-1)*stride - 2*padding + kernel` (`output_padding` and
/// `dilation` are not modelled and assumed 0 / 1).
///
/// Fully concrete when every input is a literal int; otherwise built
/// symbolically with constant-folding where possible, mirroring
/// `conv_spatial_dim`'s approach.
fn conv_transpose_spatial_dim(
    spatial_dim: &str,
    kernel_size: &str,
    stride: &str,
    padding: &str,
) -> String {
    if let (Ok(l), Ok(k), Ok(s), Ok(p)) = (
        spatial_dim.parse::<isize>(),
        kernel_size.parse::<isize>(),
        stride.parse::<isize>(),
        padding.parse::<isize>(),
    ) {
        let l_out = (l - 1) * s - 2 * p + k;
        return l_out.to_string();
    }

    // Symbolic: (L-1)*stride - 2*padding + kernel
    let stride_val: Option<isize> = stride.parse::<isize>().ok();
    let scaled = match stride_val {
        Some(1) => apply_offset(spatial_dim, -1),
        Some(s) => format!("({}-1)*{}", spatial_dim, s),
        None => format!("({}-1)*{}", spatial_dim, stride),
    };

    let mut result = scaled;
    if let Ok(p) = padding.parse::<isize>() {
        result = apply_offset(&result, -2 * p);
    } else {
        result = format!("{}-2*{}", result, padding);
    }
    if let Ok(k) = kernel_size.parse::<isize>() {
        apply_offset(&result, k)
    } else {
        format!("{}+{}", result, kernel_size)
    }
}

/// Multiply a dim by a literal factor: exact when the dim is a literal int,
/// otherwise a symbolic `dim*factor` string.
fn mul_dim(dim: &str, factor: u64) -> String {
    if let Ok(v) = dim.parse::<u64>() {
        (v * factor).to_string()
    } else {
        format!("{}*{}", dim, factor)
    }
}

/// Divide a dim by a literal factor. Errs when the dim is a literal int that
/// isn't evenly divisible (a provable contradiction); otherwise a symbolic
/// `dim/factor` string (unprovable, trusted as-is).
fn div_dim(dim: &str, factor: u64) -> Result<String, String> {
    if let Ok(v) = dim.parse::<u64>() {
        if v % factor == 0 {
            Ok((v / factor).to_string())
        } else {
            Err(format!("{} is not evenly divisible by {}", v, factor))
        }
    } else {
        Ok(format!("{}/{}", dim, factor))
    }
}

/// Product of a list of dims: exact integer product when every dim is a
/// literal int, otherwise a symbolic `d0*d1*...` string.
fn product_dims(dims: &[String]) -> String {
    if dims.is_empty() {
        return "1".to_string();
    }
    if let Some(values) = dims
        .iter()
        .map(|d| d.parse::<u64>().ok())
        .collect::<Option<Vec<_>>>()
    {
        values.iter().product::<u64>().to_string()
    } else {
        dims.join("*")
    }
}

/// Resolve a Python-style (possibly negative) axis literal against a known
/// rank. Returns `None` if the literal isn't a concrete integer or is out of
/// bounds after resolution.
fn resolve_axis(axis: &str, rank: usize) -> Option<usize> {
    let i: isize = axis.trim().parse().ok()?;
    let r = rank as isize;
    let resolved = if i < 0 { i + r } else { i };
    if resolved < 0 || resolved >= r {
        None
    } else {
        Some(resolved as usize)
    }
}

/// Split a tuple/list ctor-argument literal's raw text (e.g. `"(2, 3)"` or
/// `"[2, 3]"`) into its comma-separated component strings. Returns `None` if
/// the text doesn't look like a bracketed sequence.
fn parse_dim_sequence(text: &str) -> Option<Vec<String>> {
    let trimmed = text.trim();
    let inner = trimmed
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .or_else(|| trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')))?;
    let inner = inner.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    Some(
        inner
            .trim_end_matches(',')
            .split(',')
            .map(|s| s.trim().to_string())
            .collect(),
    )
}

/// Shared minimum-rank guard used by every layer whose validity depends on
/// input rank (Conv/Pool/ConvTranspose/shape-preserving families). Returns
/// the standard "requires input with at least N dims" error when the input
/// is too low-rank.
fn check_min_rank(
    layer_name: &str,
    app: &LayerApplication,
    input_rank: usize,
    min_rank: usize,
) -> Result<(), String> {
    if input_rank < min_rank {
        return Err(format!(
            "{} layer '{}' requires input with at least {} dims, got {} for '{}'",
            layer_name, app.layer, min_rank, input_rank, app.input
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_conv_layer(
    layer_name: &str,
    spatial_rank: usize,
    app: &LayerApplication,
    input_shape: &[String],
    in_channels: &str,
    out_channels: &str,
    kernel_size: &str,
    stride: &str,
    padding: &str,
) -> Result<Option<Vec<String>>, String> {
    let min_rank = spatial_rank + 1;
    check_min_rank(layer_name, app, input_shape.len(), min_rank)?;

    let channels_idx = input_shape.len() - spatial_rank - 1;
    let channels_dim = &input_shape[channels_idx];
    if dims_provably_mismatch(in_channels, channels_dim) {
        return Err(format!(
            "{} layer '{}' expected {} input channels, got {} for '{}'",
            layer_name, app.layer, in_channels, channels_dim, app.input
        ));
    }

    let mut output_shape = input_shape.to_vec();
    output_shape[channels_idx] = out_channels.to_string();
    for i in 0..spatial_rank {
        let spatial_idx = channels_idx + 1 + i;
        output_shape[spatial_idx] =
            conv_spatial_dim(&input_shape[spatial_idx], kernel_size, stride, padding);
    }
    Ok(Some(output_shape))
}

/// Returns the minimum input rank required by a shape-preserving layer,
/// or `None` if the layer accepts any rank (e.g. `Dropout`, `ReLU`).
///
/// Channels-first convention (no batch dimension required, matching Conv layers):
///   BatchNorm1d → 2  (C, L)
///   BatchNorm2d → 3  (C, H, W)
///   BatchNorm3d → 4  (C, D, H, W)
///   Dropout1d   → 2  (C, L)
///   Dropout2d   → 3  (C, H, W)
///   Dropout3d   → 4  (C, D, H, W)
///   GroupNorm   → 1  (needs a channel dim)
///   LayerNorm   → 1  (needs at least one dim to normalize)
///
/// Equinox `BatchNorm` and activations (`ReLU`, `GELU`, etc.) accept any rank.
fn min_rank_for_shape_preserving(name: &str) -> Option<usize> {
    match name {
        "BatchNorm1d" | "Dropout1d" | "InstanceNorm1d" => Some(2),
        // Same convention as Conv layers: channels-first without requiring
        // a batch dimension. Conv2d min_rank = 3 (C, H, W), Conv3d = 4.
        "BatchNorm2d" | "Dropout2d" | "InstanceNorm2d" => Some(3),
        "BatchNorm3d" | "Dropout3d" | "InstanceNorm3d" => Some(4),
        "LayerNorm" | "GroupNorm" | "RMSNorm" => Some(1),
        // Dropout, BatchNorm (equinox), ReLU, GELU, Sigmoid, Tanh, Softmax,
        // PReLU, Identity, AlphaDropout, TransformerEncoderLayer,
        // TransformerDecoderLayer accept any rank including scalars.
        _ => None,
    }
}

/// Canonicalize a dim expression for comparison: strip whitespace and sort
/// commutative `+` terms / `*` factors so `a + b` matches `b+a`. Expressions
/// with parens or non-commutative operators are only whitespace-stripped.
// `dims_match` used to duplicate a sort-only canonicalization in-place; it
// now delegates to the shared `canonicalize_dim` (which also folds literal
// arithmetic, e.g. `2*3*d` → `6*d`) so every dim-equality boundary in the
// crate agrees on one canonical form.
fn dims_match(a: &str, b: &str) -> bool {
    crate::types::dims_canonically_equal(a, b)
}

/// A layer's expected dim comes from constructor-argument text, a different
/// vocabulary than jaxtyping annotation dims (issue #47). A mismatch is only
/// an error when both sides are concrete integers; otherwise it's unprovable
/// and the layer's output shape is trusted.
fn dims_provably_mismatch(expected: &str, actual: &str) -> bool {
    !dims_match(expected, actual)
        && expected.trim().parse::<u64>().is_ok()
        && actual.trim().parse::<u64>().is_ok()
}

pub fn apply_layer_application(
    app: &LayerApplication,
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(input_shape) = shapes.shape(&app.input) else {
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

            if dims_provably_mismatch(in_features, last_dim) {
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
        // Channels-first layout: Conv1d expects [..., C, L]
        LayerKind::Conv1d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        } => apply_conv_layer(
            "Conv1d",
            1,
            app,
            input_shape,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        ),
        // Channels-first layout: Conv2d expects [..., C, H, W]
        LayerKind::Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        } => apply_conv_layer(
            "Conv2d",
            2,
            app,
            input_shape,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        ),
        // Channels-first layout: Conv3d expects [..., C, D, H, W]
        LayerKind::Conv3d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        } => apply_conv_layer(
            "Conv3d",
            3,
            app,
            input_shape,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        ),
        // Returns an (output, weights) tuple — a single shape can't express
        // it, so direct application skips; tuple unpacking is handled in
        // analysis::tuple_rhs_shapes.
        LayerKind::MultiheadAttention { .. } => Ok(None),
        // flax channels-last: last dim becomes `features`, nothing to check
        // (flax infers the input width at runtime).
        LayerKind::Dense { features } => {
            if input_shape.is_empty() {
                return Err(format!(
                    "Cannot apply Dense layer '{}' to scalar input '{}'",
                    app.layer, app.input
                ));
            }
            let mut output_shape = input_shape.clone();
            let last = output_shape.len() - 1;
            output_shape[last] = features.clone();
            Ok(Some(output_shape))
        }
        LayerKind::FlaxConv {
            features,
            spatial_rank,
        } => {
            check_min_rank("Conv", app, input_shape.len(), spatial_rank + 1)?;
            // Stride-1 / SAME padding: spatial dims unchanged.
            let mut output_shape = input_shape.clone();
            let last = output_shape.len() - 1;
            output_shape[last] = features.clone();
            Ok(Some(output_shape))
        }
        // Channels-first pooling: channels preserved, trailing spatial dims
        // follow the conv output formula.
        LayerKind::Pool {
            name,
            spatial_rank,
            kernel_size,
            stride,
            padding,
        } => {
            check_min_rank(name, app, input_shape.len(), spatial_rank + 1)?;
            let mut output_shape = input_shape.clone();
            let start = input_shape.len() - spatial_rank;
            for dim in &mut output_shape[start..] {
                *dim = conv_spatial_dim(dim, kernel_size, stride, padding);
            }
            Ok(Some(output_shape))
        }
        LayerKind::AdaptivePool {
            name,
            spatial_rank,
            output_size,
        } => {
            check_min_rank(name, app, input_shape.len(), spatial_rank + 1)?;
            let mut output_shape = input_shape.clone();
            let start = input_shape.len() - spatial_rank;
            for dim in &mut output_shape[start..] {
                dim.clone_from(output_size);
            }
            Ok(Some(output_shape))
        }
        // Index lookup: appends the embedding dim to the (integer) input shape.
        LayerKind::Embedding { embedding_size } => {
            let mut output_shape = input_shape.clone();
            output_shape.push(embedding_size.clone());
            Ok(Some(output_shape))
        }
        // ConvTranspose1d/2d/3d: same channels-first layout as Conv, inverse
        // spatial formula.
        LayerKind::ConvTranspose {
            spatial_rank,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        } => {
            let layer_name = format!("ConvTranspose{}d", spatial_rank);
            check_min_rank(&layer_name, app, input_shape.len(), spatial_rank + 1)?;
            let channels_idx = input_shape.len() - spatial_rank - 1;
            let channels_dim = &input_shape[channels_idx];
            if dims_provably_mismatch(in_channels, channels_dim) {
                return Err(format!(
                    "{} layer '{}' expected {} input channels, got {} for '{}'",
                    layer_name, app.layer, in_channels, channels_dim, app.input
                ));
            }
            let mut output_shape = input_shape.clone();
            output_shape[channels_idx] = out_channels.clone();
            for i in 0..*spatial_rank {
                let spatial_idx = channels_idx + 1 + i;
                output_shape[spatial_idx] = conv_transpose_spatial_dim(
                    &input_shape[spatial_idx],
                    kernel_size,
                    stride,
                    padding,
                );
            }
            Ok(Some(output_shape))
        }
        // torch.nn.Flatten(start_dim, end_dim): collapse the resolved range
        // into one dim. Unresolvable (non-literal) indices are unknown.
        LayerKind::Flatten { start_dim, end_dim } => {
            let rank = input_shape.len();
            let (Some(start), Some(end)) =
                (resolve_axis(start_dim, rank), resolve_axis(end_dim, rank))
            else {
                return Ok(None);
            };
            if start > end {
                return Ok(None);
            }
            let mut output_shape = input_shape[..start].to_vec();
            output_shape.push(product_dims(&input_shape[start..=end]));
            output_shape.extend_from_slice(&input_shape[end + 1..]);
            Ok(Some(output_shape))
        }
        // torch.nn.Unflatten(dim, sizes): expand one dim into the parsed
        // components of `sizes`.
        LayerKind::Unflatten { dim, sizes } => {
            let rank = input_shape.len();
            let Some(axis) = resolve_axis(dim, rank) else {
                return Ok(None);
            };
            let Some(components) = parse_dim_sequence(sizes) else {
                return Ok(None);
            };
            let mut output_shape = input_shape[..axis].to_vec();
            output_shape.extend(components);
            output_shape.extend_from_slice(&input_shape[axis + 1..]);
            Ok(Some(output_shape))
        }
        // torch.nn.Upsample: rank-agnostic — spatial dims are whatever
        // trails the leading (batch, channel) pair, determined here from
        // the actual input rank.
        LayerKind::Upsample { scale_factor, size } => {
            check_min_rank("Upsample", app, input_shape.len(), 3)?;
            let spatial_count = input_shape.len() - 2;
            let mut output_shape = input_shape.clone();
            if let Some(size) = size {
                let sizes = parse_dim_sequence(size).unwrap_or_else(|| vec![size.clone()]);
                if sizes.len() != spatial_count {
                    return Ok(None);
                }
                output_shape[2..].clone_from_slice(&sizes);
            } else if let Some(scale_factor) = scale_factor {
                let factors =
                    parse_dim_sequence(scale_factor).unwrap_or_else(|| vec![scale_factor.clone()]);
                if factors.len() == spatial_count {
                    for (dim, factor) in output_shape[2..].iter_mut().zip(&factors) {
                        let Ok(f) = factor.trim().parse::<u64>() else {
                            return Ok(None);
                        };
                        *dim = mul_dim(dim, f);
                    }
                } else if factors.len() == 1 {
                    let Ok(f) = factors[0].trim().parse::<u64>() else {
                        return Ok(None);
                    };
                    for dim in &mut output_shape[2..] {
                        *dim = mul_dim(dim, f);
                    }
                } else {
                    return Ok(None);
                }
            } else {
                return Ok(None);
            }
            Ok(Some(output_shape))
        }
        // torch.nn.RNN/LSTM/GRU: approximated as the primary output tensor
        // only (see LayerKind::Rnn doc comment).
        LayerKind::Rnn {
            name,
            input_size,
            hidden_size,
        } => {
            check_min_rank(name, app, input_shape.len(), 2)?;
            let last_dim = input_shape.last().unwrap();
            if dims_provably_mismatch(input_size, last_dim) {
                return Err(format!(
                    "{} layer '{}' expected input last dim {}, got {} for '{}'",
                    name, app.layer, input_size, last_dim, app.input
                ));
            }
            let mut output_shape = input_shape.clone();
            let last = output_shape.len() - 1;
            output_shape[last] = hidden_size.clone();
            Ok(Some(output_shape))
        }
        // RNNCell/GRUCell/LSTMCell: same last-dim transform, min_rank 1
        // (single step, optionally unbatched).
        LayerKind::RnnCell {
            name,
            input_size,
            hidden_size,
        } => {
            check_min_rank(name, app, input_shape.len(), 1)?;
            let last_dim = input_shape.last().unwrap();
            if dims_provably_mismatch(input_size, last_dim) {
                return Err(format!(
                    "{} layer '{}' expected input last dim {}, got {} for '{}'",
                    name, app.layer, input_size, last_dim, app.input
                ));
            }
            let mut output_shape = input_shape.clone();
            let last = output_shape.len() - 1;
            output_shape[last] = hidden_size.clone();
            Ok(Some(output_shape))
        }
        // torch.nn.PixelShuffle(r): (*, C*r^2, H, W) -> (*, C, H*r, W*r).
        LayerKind::PixelShuffle { upscale_factor } => {
            check_min_rank("PixelShuffle", app, input_shape.len(), 3)?;
            let Ok(r) = upscale_factor.trim().parse::<u64>() else {
                return Ok(None);
            };
            let mut output_shape = input_shape.clone();
            let len = output_shape.len();
            let channels_idx = len - 3;
            output_shape[channels_idx] = div_dim(&input_shape[channels_idx], r * r)
                .map_err(|e| format!("PixelShuffle layer '{}': {}", app.layer, e))?;
            output_shape[len - 2] = mul_dim(&input_shape[len - 2], r);
            output_shape[len - 1] = mul_dim(&input_shape[len - 1], r);
            Ok(Some(output_shape))
        }
        // torch.nn.PixelUnshuffle(r): (*, C, H*r, W*r) -> (*, C*r^2, H, W).
        LayerKind::PixelUnshuffle { downscale_factor } => {
            check_min_rank("PixelUnshuffle", app, input_shape.len(), 3)?;
            let Ok(r) = downscale_factor.trim().parse::<u64>() else {
                return Ok(None);
            };
            let mut output_shape = input_shape.clone();
            let len = output_shape.len();
            let channels_idx = len - 3;
            output_shape[channels_idx] = mul_dim(&input_shape[channels_idx], r * r);
            output_shape[len - 2] = div_dim(&input_shape[len - 2], r)
                .map_err(|e| format!("PixelUnshuffle layer '{}': {}", app.layer, e))?;
            output_shape[len - 1] = div_dim(&input_shape[len - 1], r)
                .map_err(|e| format!("PixelUnshuffle layer '{}': {}", app.layer, e))?;
            Ok(Some(output_shape))
        }
        // Constant/Zero/Reflection/ReplicationPad Nd: only a concrete
        // uniform-int or fully-literal 2*spatial_rank tuple is modelled;
        // anything else is unknown.
        LayerKind::Pad {
            name,
            spatial_rank,
            padding,
        } => {
            check_min_rank(name, app, input_shape.len(), *spatial_rank)?;
            let mut output_shape = input_shape.clone();
            let start = input_shape.len() - spatial_rank;
            if let Ok(p) = padding.trim().parse::<isize>() {
                for dim in &mut output_shape[start..] {
                    *dim = apply_offset(dim, 2 * p);
                }
                Ok(Some(output_shape))
            } else if let Some(pairs) = parse_dim_sequence(padding)
                && pairs.len() == 2 * spatial_rank
                && let Some(values) = pairs
                    .iter()
                    .map(|p| p.trim().parse::<isize>().ok())
                    .collect::<Option<Vec<_>>>()
            {
                // torch's reverse-axis convention: the first pair pads the
                // *last* spatial dim.
                for (i, dim) in output_shape[start..].iter_mut().rev().enumerate() {
                    let (left, right) = (values[2 * i], values[2 * i + 1]);
                    *dim = apply_offset(dim, left + right);
                }
                Ok(Some(output_shape))
            } else {
                Ok(None)
            }
        }
        // torch.nn.Bilinear: only the first tracked input (x1) is checked/
        // transformed; x2/broadcasting are not modelled (see doc comment).
        LayerKind::Bilinear {
            in1_features,
            out_features,
            ..
        } => {
            let Some(last_dim) = input_shape.last() else {
                return Err(format!(
                    "Cannot apply Bilinear layer '{}' to scalar input '{}'",
                    app.layer, app.input
                ));
            };
            if dims_provably_mismatch(in1_features, last_dim) {
                return Err(format!(
                    "Bilinear layer '{}' expected input last dim {}, got {} for '{}'",
                    app.layer, in1_features, last_dim, app.input
                ));
            }
            let mut output_shape = input_shape.clone();
            let last = output_shape.len() - 1;
            output_shape[last] = out_features.clone();
            Ok(Some(output_shape))
        }
        // torch.nn.CosineSimilarity(dim): drops the reduced axis from x1's
        // shape (same single-tracked-input limitation as Bilinear).
        LayerKind::CosineSimilarity { dim } => {
            let rank = input_shape.len();
            let Some(axis) = resolve_axis(dim, rank) else {
                return Ok(None);
            };
            let mut output_shape = input_shape.clone();
            output_shape.remove(axis);
            Ok(Some(output_shape))
        }
        // equinox.nn.MLP: last-dim transform, same rule as Linear.
        LayerKind::Mlp { in_size, out_size } => {
            let Some(last_dim) = input_shape.last() else {
                return Err(format!(
                    "Cannot apply MLP layer '{}' to scalar input '{}'",
                    app.layer, app.input
                ));
            };
            if dims_provably_mismatch(in_size, last_dim) {
                return Err(format!(
                    "MLP layer '{}' expected input last dim {}, got {} for '{}'",
                    app.layer, in_size, last_dim, app.input
                ));
            }
            let mut output_shape = input_shape.clone();
            let last = output_shape.len() - 1;
            output_shape[last] = out_size.clone();
            Ok(Some(output_shape))
        }
        // Shape-preserving layers: output shape equals input shape, but some
        // layers have minimum-rank expectations (e.g. BatchNorm2d needs 3D (C, H, W)).
        LayerKind::ShapePreserving { name } => {
            if let Some(min_rank) = min_rank_for_shape_preserving(name) {
                check_min_rank(name, app, input_shape.len(), min_rank)?;
            }
            Ok(Some(input_shape.clone()))
        }
    }
}

pub fn apply_layer_applications(
    apps: &[LayerApplication],
    scopes: &mut [FunctionShapeScope],
) -> Vec<ShapeError> {
    let mut errors = Vec::new();

    for app in apps {
        let Some(scope_idx) = scope_index_for_byte(scopes, app.range.start_byte) else {
            continue;
        };
        match apply_layer_application(app, &scopes[scope_idx].shapes) {
            Ok(Some(output_shape)) => {
                scopes[scope_idx]
                    .shapes
                    .insert(app.variable.clone(), output_shape);
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
    fn test_symbolic_mismatch_propagates() {
        // Issue #47: "features" comes from ctor-argument text, "other" from a
        // jaxtyping annotation — different vocabularies, unprovable mismatch.
        // The layer output is trusted instead of erroring.
        let app = app("x", linear("features", "hidden"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "other"]))]);

        let output = apply_layer_application(&app, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "hidden"])));
    }

    #[test]
    fn test_numeric_and_symbolic_dims_propagate() {
        // Issue #47: one symbolic side means the mismatch is unprovable.
        let app = app("x", linear("features", "hidden"));
        let shapes = HashMap::from([("x".to_string(), shape(&["batch", "3"]))]);

        let output = apply_layer_application(&app, &shapes).unwrap();

        assert_eq!(output, Some(shape(&["batch", "hidden"])));
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

    fn scopes_from(shapes: HashMap<String, Vec<String>>) -> Vec<FunctionShapeScope> {
        vec![FunctionShapeScope {
            function_name: None,
            start_byte: 0,
            end_byte: usize::MAX,
            shapes,
            return_shape: None,
            param_order: Vec::new(),
        }]
    }

    #[test]
    fn test_applies_single_application_into_shape_map() {
        let apps = vec![app("y", "layer", "x", linear("3", "5"))];
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert!(errors.is_empty());
        assert_eq!(scopes[0].shapes.get("x"), Some(&shape(&["batch", "3"])));
        assert_eq!(scopes[0].shapes.get("y"), Some(&shape(&["batch", "5"])));
    }

    #[test]
    fn test_applies_chained_applications_in_order() {
        let apps = vec![
            app("y", "l1", "x", linear("3", "5")),
            app("z", "l2", "y", linear("5", "7")),
        ];
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert!(errors.is_empty());
        assert_eq!(scopes[0].shapes.get("y"), Some(&shape(&["batch", "5"])));
        assert_eq!(scopes[0].shapes.get("z"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_missing_input_is_skipped_without_error() {
        let apps = vec![app("y", "layer", "missing", linear("3", "5"))];
        let mut scopes = scopes_from(HashMap::new());

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert!(errors.is_empty());
        assert!(!scopes[0].shapes.contains_key("y"));
    }

    #[test]
    fn test_mismatch_records_error_and_continues() {
        let apps = vec![
            app("bad", "bad_layer", "x", linear("4", "5")),
            app("good", "good_layer", "x", linear("3", "7")),
        ];
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "bad");
        assert!(errors[0].message.contains("bad_layer"));
        assert!(!scopes[0].shapes.contains_key("bad"));
        assert_eq!(scopes[0].shapes.get("good"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_order_matters_for_chains() {
        let apps = vec![
            app("z", "l2", "y", linear("5", "7")),
            app("y", "l1", "x", linear("3", "5")),
        ];
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert!(errors.is_empty());
        assert_eq!(scopes[0].shapes.get("y"), Some(&shape(&["batch", "5"])));
        assert!(!scopes[0].shapes.contains_key("z"));
    }

    #[test]
    fn test_later_assignment_overwrites_existing_output_shape() {
        let apps = vec![app("y", "layer", "x", linear("3", "5"))];
        let mut scopes = scopes_from(HashMap::from([
            ("x".to_string(), shape(&["batch", "3"])),
            ("y".to_string(), shape(&["old"])),
        ]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert!(errors.is_empty());
        assert_eq!(scopes[0].shapes.get("y"), Some(&shape(&["batch", "5"])));
    }

    #[test]
    fn test_collects_multiple_errors() {
        let apps = vec![
            app("a", "l1", "x", linear("4", "5")),
            app("b", "l2", "x", linear("6", "7")),
        ];
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].variable, "a");
        assert!(errors[0].message.contains("l1"));
        assert_eq!(errors[1].variable, "b");
        assert!(errors[1].message.contains("l2"));
        assert!(!scopes[0].shapes.contains_key("a"));
        assert!(!scopes[0].shapes.contains_key("b"));
    }

    #[test]
    fn test_scalar_error_does_not_stop_later_valid_application() {
        let apps = vec![
            app("bad", "l1", "scalar", linear("3", "5")),
            app("good", "l2", "x", linear("3", "7")),
        ];
        let mut scopes = scopes_from(HashMap::from([
            ("scalar".to_string(), Vec::new()),
            ("x".to_string(), shape(&["batch", "3"])),
        ]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "bad");
        assert!(errors[0].message.contains("scalar input"));
        assert_eq!(scopes[0].shapes.get("good"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_shape_error_points_to_output_variable_not_input_variable() {
        let apps = vec![app("projected", "projection", "x", linear("4", "5"))];
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "projected");
        assert!(errors[0].message.contains("projection"));
        assert!(errors[0].message.contains("x"));
    }

    #[test]
    fn test_missing_input_does_not_create_shape_error() {
        let apps = vec![app("y", "layer", "unknown", linear("3", "5"))];
        let mut scopes = scopes_from(HashMap::new());

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_empty_applications_preserve_existing_shapes() {
        let apps = Vec::new();
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert!(errors.is_empty());
        assert_eq!(scopes[0].shapes.get("x"), Some(&shape(&["batch", "3"])));
    }

    #[test]
    fn test_failed_application_does_not_overwrite_existing_output_shape() {
        let apps = vec![app("y", "bad_layer", "x", linear("4", "5"))];
        let mut scopes = scopes_from(HashMap::from([
            ("x".to_string(), shape(&["batch", "3"])),
            ("y".to_string(), shape(&["old", "shape"])),
        ]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "y");
        assert_eq!(scopes[0].shapes.get("y"), Some(&shape(&["old", "shape"])));
    }

    #[test]
    fn test_dependent_application_is_skipped_after_failed_producer() {
        let apps = vec![
            app("bad", "bad_layer", "x", linear("4", "5")),
            app("z", "next_layer", "bad", linear("5", "7")),
        ];
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "bad");
        assert!(!scopes[0].shapes.contains_key("bad"));
        assert!(!scopes[0].shapes.contains_key("z"));
    }

    #[test]
    fn test_successful_application_after_unrelated_error_can_use_known_input() {
        let apps = vec![
            app("bad", "bad_layer", "x", linear("4", "5")),
            app("good", "good_layer", "x", linear("3", "7")),
        ];
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].variable, "bad");
        assert_eq!(scopes[0].shapes.get("good"), Some(&shape(&["batch", "7"])));
    }

    #[test]
    fn test_multiple_errors_for_same_output_variable_are_preserved() {
        let apps = vec![
            app("y", "l1", "x", linear("4", "5")),
            app("y", "l2", "x", linear("6", "7")),
        ];
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].variable, "y");
        assert!(errors[0].message.contains("l1"));
        assert_eq!(errors[1].variable, "y");
        assert!(errors[1].message.contains("l2"));
        assert!(!scopes[0].shapes.contains_key("y"));
    }

    #[test]
    fn test_error_order_follows_application_order_with_successes_between() {
        let apps = vec![
            app("a", "l1", "x", linear("4", "5")),
            app("good", "good_layer", "x", linear("3", "9")),
            app("b", "l2", "x", linear("6", "7")),
        ];
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].variable, "a");
        assert!(errors[0].message.contains("l1"));
        assert_eq!(errors[1].variable, "b");
        assert!(errors[1].message.contains("l2"));
        assert_eq!(scopes[0].shapes.get("good"), Some(&shape(&["batch", "9"])));
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
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

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
        let mut scopes = scopes_from(HashMap::from([("x".to_string(), shape(&["batch", "3"]))]));

        let errors = apply_layer_applications(&apps, &mut scopes);

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
        // User-defined module not in the built-in catalog: must still fall
        // through to disk resolution, which fails for missing impl.
        let tmp = tempfile::tempdir().unwrap();
        let code = "import my_layers\nlayer = my_layers.Linear(3, 5)";
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

    // --- MultiheadAttention bug fix: equinox and torch bind different ctor
    // positions to the tracked feature dim. ---

    #[test]
    fn test_classifies_torch_multihead_attention_via_embed_dim() {
        let call = call(
            &["torch", "nn"],
            Some("MultiheadAttention"),
            "__init__",
            &[("embed_dim", "512"), ("num_heads", "8")],
        );

        assert_eq!(
            classify_layer_call(&call),
            Some(LayerKind::MultiheadAttention {
                feature_dim: "512".to_string()
            })
        );
    }

    #[test]
    fn test_classifies_equinox_multihead_attention_via_query_size() {
        // equinox.nn.MultiheadAttention(num_heads, query_size, ...) — reversed
        // order vs. torch. Before the fix, classify_layer_call always read
        // the `embed_dim` binding key, which equinox ctors never populate, so
        // this returned None.
        let call = call(
            &["equinox", "nn"],
            Some("MultiheadAttention"),
            "__init__",
            &[("num_heads", "8"), ("query_size", "512")],
        );

        assert_eq!(
            classify_layer_call(&call),
            Some(LayerKind::MultiheadAttention {
                feature_dim: "512".to_string()
            })
        );
    }

    #[test]
    fn test_equinox_multihead_attention_ignores_embed_dim_binding() {
        // Even if an `embed_dim` binding happened to be present (e.g. stale
        // data), the equinox path must read `query_size`, not `embed_dim`.
        let call = call(
            &["equinox", "nn"],
            Some("MultiheadAttention"),
            "__init__",
            &[
                ("embed_dim", "8"),
                ("num_heads", "8"),
                ("query_size", "512"),
            ],
        );

        assert_eq!(
            classify_layer_call(&call),
            Some(LayerKind::MultiheadAttention {
                feature_dim: "512".to_string()
            })
        );
    }
}

#[cfg(test)]
mod known_layer_signature_tests {
    use super::*;

    fn parts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|p| p.to_string()).collect()
    }

    #[test]
    fn test_equinox_linear_signature() {
        let sig = known_layer_signature(&parts(&["equinox", "nn", "Linear"])).unwrap();
        assert_eq!(sig.owner.as_deref(), Some("Linear"));
        assert_eq!(sig.name, "__init__");
        assert_eq!(&sig.params[..3], &["self", "in_features", "out_features"]);
    }

    #[test]
    fn test_torch_linear_signature() {
        let sig = known_layer_signature(&parts(&["torch", "nn", "Linear"])).unwrap();
        assert_eq!(sig.owner.as_deref(), Some("Linear"));
        assert_eq!(sig.name, "__init__");
        assert_eq!(&sig.params[..3], &["self", "in_features", "out_features"]);
    }

    #[test]
    fn test_equinox_conv_variants_signatures() {
        for class in ["Conv1d", "Conv2d", "Conv3d"] {
            let sig = known_layer_signature(&parts(&["equinox", "nn", class])).unwrap();
            assert_eq!(sig.owner.as_deref(), Some(class));
            assert_eq!(sig.name, "__init__");
            assert_eq!(
                &sig.params[..4],
                &["self", "in_channels", "out_channels", "kernel_size"]
            );
            assert!(sig.params.iter().any(|p| p == "stride"));
            assert!(sig.params.iter().any(|p| p == "padding"));
        }
    }

    #[test]
    fn test_torch_conv_variants_signatures() {
        for class in ["Conv1d", "Conv2d", "Conv3d"] {
            let sig = known_layer_signature(&parts(&["torch", "nn", class])).unwrap();
            assert_eq!(sig.owner.as_deref(), Some(class));
            assert_eq!(
                &sig.params[..4],
                &["self", "in_channels", "out_channels", "kernel_size"]
            );
        }
    }

    #[test]
    fn test_shape_preserving_layer_signatures() {
        let names = [
            "Dropout",
            "Dropout1d",
            "Dropout2d",
            "Dropout3d",
            "BatchNorm",
            "BatchNorm1d",
            "BatchNorm2d",
            "BatchNorm3d",
            "LayerNorm",
            "GroupNorm",
            "ReLU",
            "GELU",
            "Sigmoid",
            "Tanh",
            "Softmax",
            "PReLU",
        ];
        for framework in ["equinox", "torch"] {
            for name in names {
                let sig = known_layer_signature(&parts(&[framework, "nn", name]))
                    .unwrap_or_else(|| panic!("no signature for {}.nn.{}", framework, name));
                assert_eq!(sig.owner.as_deref(), Some(name));
                assert_eq!(sig.name, "__init__");
            }
        }
    }

    #[test]
    fn test_unknown_framework_returns_none() {
        assert!(known_layer_signature(&parts(&["jax", "nn", "Linear"])).is_none());
        assert!(known_layer_signature(&parts(&["flax", "nn", "Dense"])).is_none());
    }

    #[test]
    fn test_flax_linen_dense_in_catalog() {
        let sig = known_layer_signature(&parts(&["flax", "linen", "Dense"])).unwrap();
        assert_eq!(sig.owner.as_deref(), Some("Dense"));
        assert!(sig.params.contains(&"features".to_string()));
    }

    #[test]
    fn test_unknown_class_returns_none() {
        assert!(known_layer_signature(&parts(&["equinox", "nn", "Mystery"])).is_none());
        assert!(known_layer_signature(&parts(&["torch", "nn", "Mystery"])).is_none());
    }

    #[test]
    fn test_short_path_returns_none() {
        assert!(known_layer_signature(&parts(&["Linear"])).is_none());
        assert!(known_layer_signature(&parts(&["torch", "nn"])).is_none());
    }

    #[test]
    fn test_wrong_subpackage_returns_none() {
        // equinox.experimental.Linear should not match
        assert!(known_layer_signature(&parts(&["equinox", "experimental", "Linear"])).is_none());
        assert!(known_layer_signature(&parts(&["torch", "functional", "Linear"])).is_none());
    }

    #[test]
    fn test_torch_multihead_attention_signature_order() {
        let sig = known_layer_signature(&parts(&["torch", "nn", "MultiheadAttention"])).unwrap();
        assert_eq!(&sig.params[..3], &["self", "embed_dim", "num_heads"]);
    }

    #[test]
    fn test_equinox_multihead_attention_signature_order() {
        // Reversed vs. torch: num_heads first, then query_size.
        let sig = known_layer_signature(&parts(&["equinox", "nn", "MultiheadAttention"])).unwrap();
        assert_eq!(&sig.params[..3], &["self", "num_heads", "query_size"]);
    }

    #[test]
    fn test_flatten_unflatten_upsample_signatures() {
        let flatten =
            known_layer_signature(&parts(&["torch", "nn", "Flatten"])).unwrap();
        assert_eq!(flatten.params, vec!["self", "start_dim", "end_dim"]);

        let unflatten =
            known_layer_signature(&parts(&["torch", "nn", "Unflatten"])).unwrap();
        assert_eq!(unflatten.params, vec!["self", "dim", "sizes"]);

        let upsample =
            known_layer_signature(&parts(&["torch", "nn", "Upsample"])).unwrap();
        assert!(upsample.params.contains(&"scale_factor".to_string()));
        assert!(upsample.params.contains(&"size".to_string()));
    }

    #[test]
    fn test_conv_transpose_variants_share_conv_prefix() {
        for framework in ["equinox", "torch"] {
            for class in ["ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"] {
                let sig = known_layer_signature(&parts(&[framework, "nn", class])).unwrap();
                assert_eq!(
                    &sig.params[..5],
                    &["self", "in_channels", "out_channels", "kernel_size", "stride"]
                );
                assert!(sig.params.iter().any(|p| p == "padding"));
            }
        }
    }

    #[test]
    fn test_rnn_family_signatures() {
        for class in ["RNN", "LSTM", "GRU", "RNNCell", "LSTMCell", "GRUCell"] {
            let sig = known_layer_signature(&parts(&["torch", "nn", class]))
                .unwrap_or_else(|| panic!("no signature for torch.nn.{}", class));
            assert_eq!(&sig.params[..3], &["self", "input_size", "hidden_size"]);
        }
        for class in ["LSTMCell", "GRUCell"] {
            let sig = known_layer_signature(&parts(&["equinox", "nn", class]))
                .unwrap_or_else(|| panic!("no signature for equinox.nn.{}", class));
            assert_eq!(&sig.params[..3], &["self", "input_size", "hidden_size"]);
        }
    }

    #[test]
    fn test_pixel_shuffle_signatures() {
        let shuffle = known_layer_signature(&parts(&["torch", "nn", "PixelShuffle"])).unwrap();
        assert_eq!(shuffle.params, vec!["self", "upscale_factor"]);
        let unshuffle =
            known_layer_signature(&parts(&["torch", "nn", "PixelUnshuffle"])).unwrap();
        assert_eq!(unshuffle.params, vec!["self", "downscale_factor"]);
    }

    #[test]
    fn test_pad_family_signatures() {
        let constant = known_layer_signature(&parts(&["torch", "nn", "ConstantPad2d"])).unwrap();
        assert_eq!(constant.params, vec!["self", "padding", "value"]);

        for class in ["ZeroPad2d", "ReflectionPad1d", "ReplicationPad3d"] {
            let sig = known_layer_signature(&parts(&["torch", "nn", class]))
                .unwrap_or_else(|| panic!("no signature for torch.nn.{}", class));
            assert_eq!(sig.params, vec!["self", "padding"]);
        }
    }

    #[test]
    fn test_bilinear_cosine_similarity_signatures() {
        let bilinear = known_layer_signature(&parts(&["torch", "nn", "Bilinear"])).unwrap();
        assert_eq!(
            &bilinear.params[..4],
            &["self", "in1_features", "in2_features", "out_features"]
        );

        let cosine =
            known_layer_signature(&parts(&["torch", "nn", "CosineSimilarity"])).unwrap();
        assert_eq!(cosine.params, vec!["self", "dim", "eps"]);
    }

    #[test]
    fn test_equinox_mlp_signature() {
        let sig = known_layer_signature(&parts(&["equinox", "nn", "MLP"])).unwrap();
        assert_eq!(&sig.params[..3], &["self", "in_size", "out_size"]);
    }

    #[test]
    fn test_torch_mlp_is_not_in_catalog() {
        // MLP is an equinox.nn concept; torch has no torch.nn.MLP.
        assert!(known_layer_signature(&parts(&["torch", "nn", "MLP"])).is_none());
    }

    #[test]
    fn test_new_shape_preserving_layer_signatures() {
        let names = [
            "Identity",
            "InstanceNorm1d",
            "InstanceNorm2d",
            "InstanceNorm3d",
            "RMSNorm",
            "AlphaDropout",
            "TransformerEncoderLayer",
            "TransformerDecoderLayer",
        ];
        for framework in ["equinox", "torch"] {
            for name in names {
                let sig = known_layer_signature(&parts(&[framework, "nn", name]))
                    .unwrap_or_else(|| panic!("no signature for {}.nn.{}", framework, name));
                assert_eq!(sig.owner.as_deref(), Some(name));
            }
        }
    }
}

#[cfg(test)]
mod catalog_first_extract_layer_assignments_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn no_read(_: &PathBuf) -> Option<String> {
        None
    }

    #[test]
    fn test_catalog_resolves_equinox_linear_without_disk() {
        let code = "import equinox as eqx\nlayer = eqx.nn.Linear(64, 128)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Linear {
                in_features: "64".to_string(),
                out_features: "128".to_string()
            })
        );
    }

    #[test]
    fn test_catalog_resolves_torch_linear_without_disk() {
        let code = "import torch\nlayer = torch.nn.Linear(128, 256)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Linear {
                in_features: "128".to_string(),
                out_features: "256".to_string()
            })
        );
    }

    #[test]
    fn test_catalog_resolves_equinox_embedding_without_disk() {
        let code = "import equinox as eqx\nemb = eqx.nn.Embedding(10000, 512)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("emb"),
            Some(&LayerKind::Embedding {
                embedding_size: "512".to_string()
            })
        );
    }

    #[test]
    fn test_catalog_resolves_torch_embedding_kwarg_without_disk() {
        let code = "import torch\nemb = torch.nn.Embedding(10000, embedding_dim=768)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("emb"),
            Some(&LayerKind::Embedding {
                embedding_size: "768".to_string()
            })
        );
    }

    #[test]
    fn test_apply_embedding_appends_dim() {
        let shapes = HashMap::from([
            ("tokens".to_string(), vec!["batch".to_string(), "seq".to_string()]),
            ("idx".to_string(), Vec::new()),
        ]);
        let kind = LayerKind::Embedding {
            embedding_size: "512".to_string(),
        };
        let app = |input: &str| LayerApplication {
            variable: "y".to_string(),
            layer: "emb".to_string(),
            input: input.to_string(),
            kind: kind.clone(),
            range: tree_sitter::Range {
                start_byte: 0,
                end_byte: 0,
                start_point: tree_sitter::Point::new(0, 0),
                end_point: tree_sitter::Point::new(0, 0),
            },
        };

        assert_eq!(
            apply_layer_application(&app("tokens"), &shapes).unwrap(),
            Some(vec![
                "batch".to_string(),
                "seq".to_string(),
                "512".to_string()
            ])
        );
        assert_eq!(
            apply_layer_application(&app("idx"), &shapes).unwrap(),
            Some(vec!["512".to_string()])
        );
    }

    #[test]
    fn test_catalog_resolves_torch_maxpool2d_without_disk() {
        let code = "import torch\npool = torch.nn.MaxPool2d(2)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("pool"),
            Some(&LayerKind::Pool {
                name: "MaxPool2d".to_string(),
                spatial_rank: 2,
                kernel_size: "2".to_string(),
                stride: "2".to_string(),
                padding: "0".to_string()
            })
        );
    }

    #[test]
    fn test_catalog_resolves_adaptive_avgpool_without_disk() {
        let code = "import torch\npool = torch.nn.AdaptiveAvgPool2d(1)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("pool"),
            Some(&LayerKind::AdaptivePool {
                name: "AdaptiveAvgPool2d".to_string(),
                spatial_rank: 2,
                output_size: "1".to_string()
            })
        );
    }

    fn pool_app(input: &str, kind: LayerKind) -> LayerApplication {
        LayerApplication {
            variable: "y".to_string(),
            layer: "pool".to_string(),
            input: input.to_string(),
            kind,
            range: tree_sitter::Range {
                start_byte: 0,
                end_byte: 0,
                start_point: tree_sitter::Point::new(0, 0),
                end_point: tree_sitter::Point::new(0, 0),
            },
        }
    }

    #[test]
    fn test_apply_maxpool2d_halves_spatial_dims() {
        let shapes = HashMap::from([(
            "x".to_string(),
            vec!["16".to_string(), "32".to_string(), "32".to_string()],
        )]);
        let kind = LayerKind::Pool {
            name: "MaxPool2d".to_string(),
            spatial_rank: 2,
            kernel_size: "2".to_string(),
            stride: "2".to_string(),
            padding: "0".to_string(),
        };

        let output = apply_layer_application(&pool_app("x", kind), &shapes).unwrap();

        assert_eq!(
            output,
            Some(vec!["16".to_string(), "16".to_string(), "16".to_string()])
        );
    }

    #[test]
    fn test_apply_maxpool1d_symbolic_spatial_dim() {
        let shapes = HashMap::from([("x".to_string(), vec!["c".to_string(), "L".to_string()])]);
        let kind = LayerKind::Pool {
            name: "MaxPool1d".to_string(),
            spatial_rank: 1,
            kernel_size: "2".to_string(),
            stride: "2".to_string(),
            padding: "0".to_string(),
        };

        let output = apply_layer_application(&pool_app("x", kind), &shapes).unwrap();

        assert_eq!(
            output,
            Some(vec!["c".to_string(), "(L-2)/2+1".to_string()])
        );
    }

    #[test]
    fn test_apply_adaptive_pool_sets_spatial_dims() {
        let shapes = HashMap::from([(
            "x".to_string(),
            vec!["32".to_string(), "16".to_string(), "16".to_string()],
        )]);
        let kind = LayerKind::AdaptivePool {
            name: "AdaptiveAvgPool2d".to_string(),
            spatial_rank: 2,
            output_size: "1".to_string(),
        };

        let output = apply_layer_application(&pool_app("x", kind), &shapes).unwrap();

        assert_eq!(
            output,
            Some(vec!["32".to_string(), "1".to_string(), "1".to_string()])
        );
    }

    #[test]
    fn test_apply_pool_rank_too_low_errors() {
        let shapes = HashMap::from([("x".to_string(), vec!["16".to_string()])]);
        let kind = LayerKind::Pool {
            name: "MaxPool2d".to_string(),
            spatial_rank: 2,
            kernel_size: "2".to_string(),
            stride: "2".to_string(),
            padding: "0".to_string(),
        };

        let err = apply_layer_application(&pool_app("x", kind), &shapes).unwrap_err();

        assert!(err.contains("requires input with at least 3 dims"));
    }

    #[test]
    fn test_catalog_resolves_equinox_conv2d_without_disk() {
        let code = "import equinox as eqx\nlayer = eqx.nn.Conv2d(3, 16, 3)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Conv2d {
                in_channels: "3".to_string(),
                out_channels: "16".to_string(),
                kernel_size: "3".to_string(),
                stride: "1".to_string(),
                padding: "0".to_string(),
            })
        );
    }

    #[test]
    fn test_catalog_resolves_torch_conv2d_with_stride_padding_without_disk() {
        let code = "import torch\nlayer = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Conv2d {
                in_channels: "3".to_string(),
                out_channels: "16".to_string(),
                kernel_size: "3".to_string(),
                stride: "2".to_string(),
                padding: "1".to_string(),
            })
        );
    }

    #[test]
    fn test_catalog_resolves_torch_dropout_shape_preserving() {
        let code = "import torch\nlayer = torch.nn.Dropout()";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::ShapePreserving {
                name: "Dropout".to_string()
            })
        );
    }

    #[test]
    fn test_catalog_supports_from_import_alias_without_disk() {
        let code =
            "from equinox.nn import Linear as Lin\nlayer = Lin(in_features=3, out_features=5)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Linear {
                in_features: "3".to_string(),
                out_features: "5".to_string()
            })
        );
    }

    #[test]
    fn test_non_layer_call_is_not_added_to_layer_map() {
        let code = "import jax.numpy as jnp\ny = jnp.sum(x)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert!(layers.is_empty());
    }

    #[test]
    fn test_user_layer_without_catalog_or_disk_is_skipped() {
        // `my_module.Linear` is neither equinox.nn.Linear nor torch.nn.Linear,
        // and there's no read_file backing — so it should be silently skipped.
        let code = "import my_module\nlayer = my_module.Linear(3, 5)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert!(layers.is_empty());
    }

    // --- MultiheadAttention bug fix, exercised end-to-end through real
    // parsed positional-arg constructor calls (not just direct
    // ResolvedCallSignature construction). ---

    #[test]
    fn test_catalog_resolves_torch_multihead_attention_unchanged() {
        // Matches corpus/torch_attention.py's `nn.MultiheadAttention(512, 8)`.
        let code = "import torch.nn as nn\nattn = nn.MultiheadAttention(512, 8)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("attn"),
            Some(&LayerKind::MultiheadAttention {
                feature_dim: "512".to_string()
            })
        );
    }

    #[test]
    fn test_catalog_resolves_equinox_multihead_attention_via_query_size() {
        // Before the fix this returned None: classify_layer_call always
        // looked up the `embed_dim` binding, which the equinox positional
        // order (num_heads, query_size) never populates.
        let code = "import equinox as eqx\nattn = eqx.nn.MultiheadAttention(8, 512)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("attn"),
            Some(&LayerKind::MultiheadAttention {
                feature_dim: "512".to_string()
            })
        );
    }

    #[test]
    fn test_catalog_resolves_equinox_multihead_attention_with_keywords() {
        let code =
            "import equinox as eqx\nattn = eqx.nn.MultiheadAttention(num_heads=4, query_size=256)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("attn"),
            Some(&LayerKind::MultiheadAttention {
                feature_dim: "256".to_string()
            })
        );
    }

    #[test]
    fn test_catalog_resolves_torch_flatten_without_disk() {
        let code = "import torch\nlayer = torch.nn.Flatten(1, 2)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::Flatten {
                start_dim: "1".to_string(),
                end_dim: "2".to_string()
            })
        );
    }

    #[test]
    fn test_catalog_resolves_torch_conv_transpose2d_without_disk() {
        let code = "import torch\nlayer = torch.nn.ConvTranspose2d(3, 16, 4, stride=2, padding=1)";
        let tree = parse(code);
        let roots: Vec<PathBuf> = Vec::new();

        let layers = extract_layer_assignments(tree.root_node(), code, &roots, no_read, 5).unwrap();

        assert_eq!(
            layers.get("layer"),
            Some(&LayerKind::ConvTranspose {
                spatial_rank: 2,
                in_channels: "3".to_string(),
                out_channels: "16".to_string(),
                kernel_size: "4".to_string(),
                stride: "2".to_string(),
                padding: "1".to_string(),
            })
        );
    }
}

#[cfg(test)]
mod new_layer_kind_apply_tests {
    use super::*;

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|d| d.to_string()).collect()
    }

    fn app(input: &str, kind: LayerKind) -> LayerApplication {
        LayerApplication {
            variable: "y".to_string(),
            layer: "layer".to_string(),
            input: input.to_string(),
            kind,
            range: tree_sitter::Range {
                start_byte: 0,
                end_byte: 0,
                start_point: tree_sitter::Point::new(0, 0),
                end_point: tree_sitter::Point::new(0, 0),
            },
        }
    }

    fn shapes_of(pairs: &[(&str, &[&str])]) -> HashMap<String, Vec<String>> {
        pairs
            .iter()
            .map(|(name, dims)| (name.to_string(), shape(dims)))
            .collect()
    }

    // --- Flatten ---

    #[test]
    fn test_flatten_collapses_middle_dims() {
        let shapes = shapes_of(&[("x", &["2", "3", "4", "5"])]);
        let kind = LayerKind::Flatten {
            start_dim: "1".to_string(),
            end_dim: "2".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["2", "12", "5"])));
    }

    #[test]
    fn test_flatten_symbolic_product_and_negative_end_dim() {
        let shapes = shapes_of(&[("x", &["batch", "c", "d"])]);
        let kind = LayerKind::Flatten {
            start_dim: "1".to_string(),
            end_dim: "-1".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["batch", "c*d"])));
    }

    #[test]
    fn test_flatten_unresolvable_start_dim_is_unknown() {
        let shapes = shapes_of(&[("x", &["2", "3", "4"])]);
        let kind = LayerKind::Flatten {
            start_dim: "axis".to_string(),
            end_dim: "-1".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, None);
    }

    // --- Unflatten ---

    #[test]
    fn test_unflatten_expands_dim_into_components() {
        let shapes = shapes_of(&[("x", &["batch", "6", "feat"])]);
        let kind = LayerKind::Unflatten {
            dim: "1".to_string(),
            sizes: "(2, 3)".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["batch", "2", "3", "feat"])));
    }

    #[test]
    fn test_unflatten_non_literal_dim_is_unknown() {
        let shapes = shapes_of(&[("x", &["batch", "6", "feat"])]);
        let kind = LayerKind::Unflatten {
            dim: "axis".to_string(),
            sizes: "(2, 3)".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, None);
    }

    // --- Upsample ---

    #[test]
    fn test_upsample_scalar_scale_factor_doubles_spatial_dims() {
        let shapes = shapes_of(&[("x", &["1", "3", "8", "8"])]);
        let kind = LayerKind::Upsample {
            scale_factor: Some("2".to_string()),
            size: None,
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["1", "3", "16", "16"])));
    }

    #[test]
    fn test_upsample_explicit_size_tuple() {
        let shapes = shapes_of(&[("x", &["1", "3", "8", "8"])]);
        let kind = LayerKind::Upsample {
            scale_factor: None,
            size: Some("(20, 20)".to_string()),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["1", "3", "20", "20"])));
    }

    #[test]
    fn test_upsample_rank_too_low_errors() {
        let shapes = shapes_of(&[("x", &["3", "8"])]);
        let kind = LayerKind::Upsample {
            scale_factor: Some("2".to_string()),
            size: None,
        };

        let err = apply_layer_application(&app("x", kind), &shapes).unwrap_err();

        assert!(err.contains("requires input with at least 3 dims"));
    }

    // --- ConvTranspose ---

    #[test]
    fn test_conv_transpose2d_inverts_conv_output_formula() {
        // conv_spatial_dim(8, k=4, s=2, p=1) == 4; the transpose must invert
        // that back to 8.
        let shapes = shapes_of(&[("x", &["3", "4", "4"])]);
        let kind = LayerKind::ConvTranspose {
            spatial_rank: 2,
            in_channels: "3".to_string(),
            out_channels: "16".to_string(),
            kernel_size: "4".to_string(),
            stride: "2".to_string(),
            padding: "1".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["16", "8", "8"])));
    }

    #[test]
    fn test_conv_transpose2d_channel_mismatch_errors() {
        let shapes = shapes_of(&[("x", &["5", "4", "4"])]);
        let kind = LayerKind::ConvTranspose {
            spatial_rank: 2,
            in_channels: "3".to_string(),
            out_channels: "16".to_string(),
            kernel_size: "4".to_string(),
            stride: "2".to_string(),
            padding: "1".to_string(),
        };

        let err = apply_layer_application(&app("x", kind), &shapes).unwrap_err();

        assert!(err.contains("expected 3 input channels"));
    }

    // --- Rnn / RnnCell ---

    #[test]
    fn test_rnn_replaces_last_dim_with_hidden_size() {
        let shapes = shapes_of(&[("x", &["seq", "10"])]);
        let kind = LayerKind::Rnn {
            name: "GRU".to_string(),
            input_size: "10".to_string(),
            hidden_size: "20".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["seq", "20"])));
    }

    #[test]
    fn test_rnn_input_size_mismatch_errors() {
        let shapes = shapes_of(&[("x", &["seq", "9"])]);
        let kind = LayerKind::Rnn {
            name: "GRU".to_string(),
            input_size: "10".to_string(),
            hidden_size: "20".to_string(),
        };

        let err = apply_layer_application(&app("x", kind), &shapes).unwrap_err();

        assert!(err.contains("expected input last dim 10"));
    }

    #[test]
    fn test_rnn_cell_applies_to_unbatched_input() {
        let shapes = shapes_of(&[("x", &["10"])]);
        let kind = LayerKind::RnnCell {
            name: "LSTMCell".to_string(),
            input_size: "10".to_string(),
            hidden_size: "20".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["20"])));
    }

    // --- PixelShuffle / PixelUnshuffle ---

    #[test]
    fn test_pixel_shuffle_trades_channels_for_spatial_resolution() {
        let shapes = shapes_of(&[("x", &["8", "4", "4"])]);
        let kind = LayerKind::PixelShuffle {
            upscale_factor: "2".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["2", "8", "8"])));
    }

    #[test]
    fn test_pixel_shuffle_indivisible_channels_errors() {
        let shapes = shapes_of(&[("x", &["9", "4", "4"])]);
        let kind = LayerKind::PixelShuffle {
            upscale_factor: "2".to_string(),
        };

        let err = apply_layer_application(&app("x", kind), &shapes).unwrap_err();

        assert!(err.contains("not evenly divisible"));
    }

    #[test]
    fn test_pixel_unshuffle_is_inverse_of_shuffle() {
        let shapes = shapes_of(&[("x", &["2", "8", "8"])]);
        let kind = LayerKind::PixelUnshuffle {
            downscale_factor: "2".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["8", "4", "4"])));
    }

    // --- Pad family ---

    #[test]
    fn test_pad_uniform_int_pads_every_spatial_dim() {
        let shapes = shapes_of(&[("x", &["3", "8", "8"])]);
        let kind = LayerKind::Pad {
            name: "ConstantPad2d".to_string(),
            spatial_rank: 2,
            padding: "1".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["3", "10", "10"])));
    }

    #[test]
    fn test_pad_tuple_uses_reverse_axis_convention() {
        // padding=(left, right, top, bottom): last spatial dim (W) gets the
        // first pair, second-to-last (H) gets the second pair.
        let shapes = shapes_of(&[("x", &["3", "8", "8"])]);
        let kind = LayerKind::Pad {
            name: "ConstantPad2d".to_string(),
            spatial_rank: 2,
            padding: "(1, 1, 2, 2)".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["3", "12", "10"])));
    }

    #[test]
    fn test_pad_symbolic_padding_is_unknown() {
        let shapes = shapes_of(&[("x", &["3", "8", "8"])]);
        let kind = LayerKind::Pad {
            name: "ConstantPad2d".to_string(),
            spatial_rank: 2,
            padding: "p".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, None);
    }

    // --- Bilinear / CosineSimilarity ---

    #[test]
    fn test_bilinear_transforms_first_input_last_dim() {
        let shapes = shapes_of(&[("x1", &["batch", "10"])]);
        let kind = LayerKind::Bilinear {
            in1_features: "10".to_string(),
            in2_features: "20".to_string(),
            out_features: "30".to_string(),
        };

        let out = apply_layer_application(&app("x1", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["batch", "30"])));
    }

    #[test]
    fn test_bilinear_in1_features_mismatch_errors() {
        let shapes = shapes_of(&[("x1", &["batch", "9"])]);
        let kind = LayerKind::Bilinear {
            in1_features: "10".to_string(),
            in2_features: "20".to_string(),
            out_features: "30".to_string(),
        };

        let err = apply_layer_application(&app("x1", kind), &shapes).unwrap_err();

        assert!(err.contains("expected input last dim 10"));
    }

    #[test]
    fn test_cosine_similarity_removes_reduced_axis() {
        let shapes = shapes_of(&[("x1", &["batch", "10"])]);
        let kind = LayerKind::CosineSimilarity {
            dim: "1".to_string(),
        };

        let out = apply_layer_application(&app("x1", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["batch"])));
    }

    #[test]
    fn test_cosine_similarity_non_literal_dim_is_unknown() {
        let shapes = shapes_of(&[("x1", &["batch", "10"])]);
        let kind = LayerKind::CosineSimilarity {
            dim: "axis".to_string(),
        };

        let out = apply_layer_application(&app("x1", kind), &shapes).unwrap();

        assert_eq!(out, None);
    }

    // --- equinox MLP ---

    #[test]
    fn test_mlp_transforms_last_dim() {
        let shapes = shapes_of(&[("x", &["10"])]);
        let kind = LayerKind::Mlp {
            in_size: "10".to_string(),
            out_size: "5".to_string(),
        };

        let out = apply_layer_application(&app("x", kind), &shapes).unwrap();

        assert_eq!(out, Some(shape(&["5"])));
    }

    #[test]
    fn test_mlp_in_size_mismatch_errors() {
        let shapes = shapes_of(&[("x", &["9"])]);
        let kind = LayerKind::Mlp {
            in_size: "10".to_string(),
            out_size: "5".to_string(),
        };

        let err = apply_layer_application(&app("x", kind), &shapes).unwrap_err();

        assert!(err.contains("expected input last dim 10"));
    }

    // --- New ShapePreserving names: min-rank vs. any-rank ---

    #[test]
    fn test_instance_norm2d_requires_min_rank() {
        let shapes = shapes_of(&[("x", &["8"])]);
        let kind = LayerKind::ShapePreserving {
            name: "InstanceNorm2d".to_string(),
        };

        let err = apply_layer_application(&app("x", kind), &shapes).unwrap_err();

        assert!(err.contains("requires input with at least 3 dims"));
    }

    #[test]
    fn test_identity_alpha_dropout_and_transformer_layers_accept_any_rank() {
        let shapes = shapes_of(&[("scalar", &[])]);
        for name in [
            "Identity",
            "AlphaDropout",
            "TransformerEncoderLayer",
            "TransformerDecoderLayer",
        ] {
            let kind = LayerKind::ShapePreserving {
                name: name.to_string(),
            };
            let out = apply_layer_application(&app("scalar", kind), &shapes).unwrap();
            assert_eq!(out, Some(Vec::new()), "layer {} should accept rank 0", name);
        }
    }

    #[test]
    fn test_rms_norm_requires_at_least_rank_1() {
        let shapes = shapes_of(&[("scalar", &[])]);
        let kind = LayerKind::ShapePreserving {
            name: "RMSNorm".to_string(),
        };

        let err = apply_layer_application(&app("scalar", kind), &shapes).unwrap_err();

        assert!(err.contains("requires input with at least 1 dims"));
    }
}
