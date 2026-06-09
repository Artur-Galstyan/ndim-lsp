use std::{collections::HashMap, path::PathBuf};

use tree_sitter::Node;

#[cfg(test)]
use tree_sitter::Range;

use crate::python_ast::{build_import_map, extract_call_arguments, extract_calls};
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

    if !is_equinox_module && !is_torch_module {
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
        // Shape-preserving layers (Dropout, BatchNorm, LayerNorm, GroupNorm, activations)
        "Dropout" | "Dropout1d" | "Dropout2d" | "Dropout3d" | "BatchNorm" | "BatchNorm1d"
        | "BatchNorm2d" | "BatchNorm3d" | "LayerNorm" | "GroupNorm" | "ReLU" | "GELU"
        | "Sigmoid" | "Tanh" | "Softmax" | "PReLU" => Some(LayerKind::ShapePreserving {
            name: owner.to_string(),
        }),
        _ => None,
    }
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
    if module != "nn" {
        return None;
    }
    if framework != "equinox" && framework != "torch" {
        return None;
    }
    let class_name = parts.last()?.as_str();
    let params: &[&str] = match class_name {
        "Linear" => &["self", "in_features", "out_features", "use_bias"],
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
        | "Sigmoid" | "Tanh" | "Softmax" | "PReLU" => &["self"],
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
    let records =
        extract_layer_assignments_scoped(node, text, search_roots, read_file, max_depth, None)?;
    let mut layers = HashMap::new();
    for rec in records {
        layers.insert(rec.name, rec.kind);
    }
    Ok(layers)
}

pub fn extract_layer_assignments_scoped<F>(
    node: Node,
    text: &str,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<Vec<LayerAssignment>, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let import_map = build_import_map(node, text)?;
    let calls = extract_calls(node, text)?;
    let mut records = Vec::new();

    for call in calls {
        // Catalog-first: hardcoded equinox.nn.* / torch.nn.* signatures
        // bypass disk resolution. Falls through to resolve_call_signature for
        // user-defined layers and frameworks not in the catalog.
        let resolved_call = match try_catalog_signature(&call, node, text, &import_map)? {
            Some(c) => Some(c),
            None => resolve_call_signature(
                &call,
                text,
                &import_map,
                search_roots,
                &read_file,
                max_depth,
                cache,
            )?,
        };
        let Some(resolved_call) = resolved_call else {
            continue;
        };
        let Some(layer) = classify_layer_call(&resolved_call) else {
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
    if input_shape.len() < min_rank {
        return Err(format!(
            "{} layer '{}' requires input with at least {} dims, got {} for '{}'",
            layer_name,
            app.layer,
            min_rank,
            input_shape.len(),
            app.input
        ));
    }

    let channels_idx = input_shape.len() - spatial_rank - 1;
    let channels_dim = &input_shape[channels_idx];
    if channels_dim != in_channels {
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
        "BatchNorm1d" | "Dropout1d" => Some(2),
        // Same convention as Conv layers: channels-first without requiring
        // a batch dimension. Conv2d min_rank = 3 (C, H, W), Conv3d = 4.
        "BatchNorm2d" | "Dropout2d" => Some(3),
        "BatchNorm3d" | "Dropout3d" => Some(4),
        "LayerNorm" | "GroupNorm" => Some(1),
        // Dropout, BatchNorm (equinox), ReLU, GELU, Sigmoid, Tanh, Softmax, PReLU
        // accept any rank including scalars.
        _ => None,
    }
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
        // Shape-preserving layers: output shape equals input shape, but some
        // layers have minimum-rank expectations (e.g. BatchNorm2d needs 3D (C, H, W)).
        LayerKind::ShapePreserving { name } => {
            if let Some(min_rank) = min_rank_for_shape_preserving(name)
                && input_shape.len() < min_rank
            {
                return Err(format!(
                    "{} layer '{}' requires input with at least {} dims, got {} for '{}'",
                    name,
                    app.layer,
                    min_rank,
                    input_shape.len(),
                    app.input
                ));
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
        assert!(known_layer_signature(&parts(&["flax", "linen", "Dense"])).is_none());
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
}
