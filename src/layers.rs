use std::{collections::HashMap, path::PathBuf};

use tree_sitter::Node;

#[cfg(test)]
use tree_sitter::Range;

use crate::imports::ImportScopes;
use crate::known_functions::{apply_known_einops, apply_loss_layer};
use crate::python_ast::{
    class_ranges, enclosing_class_range, extract_call_arguments, extract_calls,
    extract_self_attr_calls,
};
use crate::resolution::{
    ResolutionCache, bind_call_arguments, resolve_call_signature_with_node, resolve_call_target,
};
use crate::types::*;

/// The fifth positional parameter of `torch.nn.RNN`: `nonlinearity` in recent
/// torch, `bias` before. A call that binds it is not classified.
const RNN_POSITION_4: &str = "__rnn_position_4";

/// Stands for a constructor dim whose value the file does not prove.
pub(crate) const UNPROVEN_DIM: &str = "__unproven";

/// Starts a `padding` that is one name. A string is also a valid padding, so
/// `ShapeCtx::layer_at` keeps the name only when it proves an integer. Without
/// that step the padding of the layer stays unknown.
pub(crate) const PADDING_NAME: &str = "__padding_";

/// Layers whose output shape is the input shape. LocalResponseNorm normalizes
/// across a window of neighboring channels. TransformerEncoderLayer and
/// TransformerDecoderLayer, and the TransformerEncoder/TransformerDecoder stacks,
/// preserve d_model on their primary input; the second (`memory`) input of the
/// decoder is not tracked, the same single-input limit as
/// Bilinear/CosineSimilarity.
const SHAPE_PRESERVING_LAYERS: [&str; 49] = [
    "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "BatchNorm", "BatchNorm1d", "BatchNorm2d",
    "BatchNorm3d", "LayerNorm", "GroupNorm", "ReLU", "GELU", "Sigmoid", "Tanh", "Softmax", "PReLU",
    "Identity", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "RMSNorm", "AlphaDropout",
    "LocalResponseNorm", "TransformerEncoderLayer", "TransformerDecoderLayer", "TransformerEncoder",
    "TransformerDecoder", "SiLU", "LeakyReLU", "ELU", "SELU", "CELU", "Softplus", "Mish", "Hardswish",
    "Hardsigmoid", "Hardtanh", "ReLU6", "LogSoftmax", "Softmin", "Softsign", "Tanhshrink", "Threshold",
    "LogSigmoid", "Softshrink", "Hardshrink", "RReLU", "FeatureAlphaDropout", "Softmax2d",
];

#[cfg(test)]
pub fn classify_layer_call(call: &ResolvedCallSignature) -> Option<LayerKind> {
    classify_layer_text(call).map(strip_self_prefixes)
}

/// The layer of a constructor call, with each dim as the text of its argument.
fn classify_layer_text(call: &ResolvedCallSignature) -> Option<LayerKind> {
    let is_equinox_module = call.implementation.target.module_parts.len() >= 2
        && call.implementation.target.module_parts[0] == "equinox"
        && call.implementation.target.module_parts[1] == "nn";
    let is_torch_module = call.implementation.target.module_parts.len() >= 2
        && call.implementation.target.module_parts[0] == "torch"
        && call.implementation.target.module_parts[1] == "nn";
    let is_flax_module = call.implementation.target.module_parts.len() >= 2
        && call.implementation.target.module_parts[0] == "flax"
        && call.implementation.target.module_parts[1] == "linen";
    // einops.layers.torch / einops.layers.flax — layer-object form of the
    // rearrange/reduce pattern algebra (see `LayerKind::EinopsPattern`).
    let is_einops_layers_module = call.implementation.target.module_parts.len() >= 3
        && call.implementation.target.module_parts[0] == "einops"
        && call.implementation.target.module_parts[1] == "layers"
        && (call.implementation.target.module_parts[2] == "torch"
            || call.implementation.target.module_parts[2] == "flax");

    if !is_equinox_module && !is_torch_module && !is_flax_module && !is_einops_layers_module {
        return None;
    }

    let owner = call.signature.owner.as_deref()?;
    if call.signature.name != "__init__" {
        return None;
    }
    // ponytail: equinox `BatchNorm` gives `(output, state)`, so it is no
    // array layer. Model the pair when the lost shapes matter.
    if is_equinox_module && owner == "BatchNorm" {
        return None;
    }

    match owner {
        "Linear" => Some(LayerKind::Linear {
            in_features: call.bindings.get("in_features")?.clone(),
            out_features: call.bindings.get("out_features")?.clone(),
        }),
        "Conv1d" | "Conv2d" | "Conv3d" => {
            let spatial_rank = pool_spatial_rank(owner)?;
            // Per-axis tuples (e.g. kernel_size=(3,5)) are not yet supported.
            // Detect and refuse to classify so we don't produce garbage symbolic output.
            let kernel_size = call.bindings.get("kernel_size")?;
            if is_sequence(kernel_size) {
                return None;
            }
            let stride = call
                .bindings
                .get("stride")
                .cloned()
                .unwrap_or_else(|| "1".to_string());
            if is_sequence(&stride) {
                return None;
            }
            let padding = padding_argument(call);
            if is_sequence(&padding) {
                return None;
            }
            // torch refuses every other string. equinox takes any case.
            if is_torch_module && string_text(&padding).is_some_and(|text| !matches!(text, "same" | "valid")) {
                return None;
            }
            let dilation = call
                .bindings
                .get("dilation")
                .cloned()
                .unwrap_or_else(|| "1".to_string());
            Some(LayerKind::Conv {
                spatial_rank,
                in_channels: call.bindings.get("in_channels")?.clone(),
                out_channels: call.bindings.get("out_channels")?.clone(),
                kernel_size: kernel_size.clone(),
                stride,
                padding,
                dilation,
            })
        }
        "Dense" => Some(LayerKind::Dense {
            features: call.bindings.get("features")?.clone(),
        }),
        "Conv" if is_flax_module => {
            let features = call.bindings.get("features")?.clone();
            let kernel_size = call.bindings.get("kernel_size")?;
            // Spatial rank from the kernel tuple: (3, 3) → 2, scalar → 1.
            let spatial_rank = if is_sequence(kernel_size) {
                parse_dim_sequence(kernel_size)?.len()
            } else {
                1
            };
            // v1 models only the default stride-1 / SAME-padding case.
            if let Some(strides) = call.bindings.get("strides")
                && parse_dim_sequence(strides)
                    .unwrap_or_else(|| vec![strides.clone()])
                    .iter()
                    .any(|s| !s.is_empty() && s != "1")
            {
                return None;
            }
            if call.bindings.contains_key("input_dilation") || call.bindings.contains_key("kernel_dilation") {
                return None;
            }
            Some(LayerKind::FlaxConv {
                features,
                spatial_rank,
                kernel_size: kernel_size.clone(),
                padding: if call.bindings.contains_key("padding") {
                    padding_argument(call)
                } else {
                    "\"SAME\"".to_string()
                },
            })
        }
        "MaxPool1d" | "MaxPool2d" | "MaxPool3d" | "AvgPool1d" | "AvgPool2d" | "AvgPool3d" => {
            let spatial_rank = pool_spatial_rank(owner)?;
            let kernel_size = call.bindings.get("kernel_size")?.clone();
            if is_sequence(&kernel_size) {
                return None;
            }
            // torch: stride=None means kernel_size. equinox: stride=1.
            let stride = match call.bindings.get("stride").map(|stride| stride.trim()) {
                Some(stride) if stride != "None" => stride.to_string(),
                _ if is_equinox_module => "1".to_string(),
                _ => kernel_size.clone(),
            };
            if is_sequence(&stride) {
                return None;
            }
            let padding = padding_argument(call);
            if is_sequence(&padding) {
                return None;
            }
            let flag = |name: &str| call.bindings.get(name).map_or("False", |value| value.trim());
            // `return_indices=True` gives an `(output, indices)` tuple.
            if flag("return_indices") != "False" {
                return None;
            }
            // equinox `use_ceil` rounds up without the torch rule that drops a
            // last window that starts in the padding: not modelled.
            let ceil_mode = if is_equinox_module {
                if flag("use_ceil") == "False" { "False".to_string() } else { UNPROVEN_DIM.to_string() }
            } else {
                flag("ceil_mode").to_string()
            };
            let dilation = call
                .bindings
                .get("dilation")
                .filter(|_| owner.starts_with("MaxPool") && is_torch_module)
                .cloned()
                .unwrap_or_else(|| "1".to_string());
            Some(LayerKind::Pool {
                name: owner.to_string(),
                spatial_rank,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            })
        }
        "AdaptiveMaxPool1d" | "AdaptiveMaxPool2d" | "AdaptiveMaxPool3d" | "AdaptiveAvgPool1d"
        | "AdaptiveAvgPool2d" | "AdaptiveAvgPool3d" => {
            let spatial_rank = pool_spatial_rank(owner)?;
            let output_size = call.bindings.get("output_size")?.clone();
            if is_sequence(&output_size) {
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
            let flag = |name: &str| call.bindings.get(name).map_or("False", |value| value.trim());
            let added_keys = match (flag("add_bias_kv"), flag("add_zero_attn")) {
                (a @ ("True" | "False"), b @ ("True" | "False")) => {
                    ((a == "True") as u8 + (b == "True") as u8).to_string()
                }
                _ => UNPROVEN_DIM.to_string(),
            };
            let output_size = match call.bindings.get("output_size").map(|value| value.trim()) {
                Some(size) if is_equinox_module && size != "None" => size.to_string(),
                _ => feature_dim.clone(),
            };
            Some(LayerKind::MultiheadAttention {
                output_size,
                feature_dim: feature_dim.clone(),
                num_heads: call.bindings.get("num_heads").cloned().unwrap_or_else(|| UNPROVEN_DIM.to_string()),
                batch_first: flag("batch_first").to_string(),
                added_keys,
                returns_weights: !is_equinox_module,
            })
        }
        // flax.linen.Embed(num_embeddings, features).
        "Embed" if is_flax_module => Some(LayerKind::Embedding {
            embedding_size: call.bindings.get("features")?.clone(),
        }),
        // equinox names it `embedding_size`, torch `embedding_dim`.
        "Embedding" => Some(LayerKind::Embedding {
            embedding_size: call
                .bindings
                .get("embedding_size")
                .or_else(|| call.bindings.get("embedding_dim"))?
                .clone(),
        }),
        // flax attention: `out_features` defaults to the features of the query.
        "MultiHeadDotProductAttention" | "MultiHeadAttention" | "SelfAttention"
            if is_flax_module && !call.bindings.contains_key("out_features") =>
        {
            Some(LayerKind::ShapePreserving { name: owner.to_string() })
        }
        name if SHAPE_PRESERVING_LAYERS.contains(&name) => Some(LayerKind::ShapePreserving {
            name: owner.to_string(),
        }),
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
            if is_sequence(&kernel_size) {
                return None;
            }
            let stride = call
                .bindings
                .get("stride")
                .cloned()
                .unwrap_or_else(|| "1".to_string());
            if is_sequence(&stride) {
                return None;
            }
            let padding = padding_argument(call);
            // torch refuses a string padding here. Only equinox has one.
            if is_sequence(&padding) || (is_torch_module && string_text(&padding).is_some()) {
                return None;
            }
            let or_default = |name: &str, default: &str| {
                call.bindings.get(name).cloned().unwrap_or_else(|| default.to_string())
            };
            Some(LayerKind::ConvTranspose {
                spatial_rank,
                in_channels: call.bindings.get("in_channels")?.clone(),
                out_channels: call.bindings.get("out_channels")?.clone(),
                kernel_size,
                stride,
                padding,
                output_padding: or_default("output_padding", "0"),
                dilation: or_default("dilation", "1"),
            })
        }
        "RNN" | "LSTM" | "GRU" if is_torch_module => {
            // torch.nn.RNN takes `nonlinearity` fourth in recent versions only.
            if call.bindings.contains_key(RNN_POSITION_4) {
                return None;
            }
            let or_default = |name: &str, default: &str| {
                call.bindings.get(name).map_or(default.to_string(), |value| value.trim().to_string())
            };
            Some(LayerKind::Rnn {
                name: owner.to_string(),
                input_size: call.bindings.get("input_size")?.clone(),
                hidden_size: call.bindings.get("hidden_size")?.clone(),
                num_layers: or_default("num_layers", "1"),
                bidirectional: or_default("bidirectional", "False"),
                proj_size: if owner == "LSTM" { or_default("proj_size", "0") } else { "0".to_string() },
                batch_first: or_default("batch_first", "False"),
            })
        }
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
        "MSELoss" | "L1Loss" | "SmoothL1Loss" | "HuberLoss" | "BCELoss" | "BCEWithLogitsLoss"
        | "CrossEntropyLoss" | "NLLLoss"
            if is_torch_module =>
        {
            let given = |name: &str| call.bindings.get(name).is_some_and(|v| v.trim() != "None");
            let reduction = if given("size_average") || given("reduce") {
                None
            } else {
                Some(
                    call.bindings
                        .get("reduction")
                        .cloned()
                        .unwrap_or_else(|| "'mean'".to_string()),
                )
            };
            Some(LayerKind::Loss {
                name: owner.to_string(),
                reduction,
            })
        }
        "MLP" if is_equinox_module => Some(LayerKind::Mlp {
            in_size: call.bindings.get("in_size")?.clone(),
            out_size: call.bindings.get("out_size")?.clone(),
        }),
        // torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1):
        // geometry args are stored as raw ctor text (scalar or 2-tuple) and
        // parsed at apply time, mirroring `Upsample`.
        "Unfold" => Some(LayerKind::Unfold {
            kernel_size: call.bindings.get("kernel_size")?.clone(),
            stride: call
                .bindings
                .get("stride")
                .cloned()
                .unwrap_or_else(|| "1".to_string()),
            padding: call
                .bindings
                .get("padding")
                .cloned()
                .unwrap_or_else(|| "0".to_string()),
            dilation: call
                .bindings
                .get("dilation")
                .cloned()
                .unwrap_or_else(|| "1".to_string()),
        }),
        // torch.nn.Fold(output_size, kernel_size, ...): only the two dims
        // that feed the output-shape formula are tracked.
        "Fold" => Some(LayerKind::Fold {
            output_size: call.bindings.get("output_size")?.clone(),
            kernel_size: call.bindings.get("kernel_size")?.clone(),
        }),
        // einops.layers.{torch,flax}.Rearrange/Reduce: the pattern string and
        // any axis-length kwargs pass straight through to
        // `apply_known_einops` at apply time (see `LayerKind::EinopsPattern`).
        "Rearrange" | "Reduce" if is_einops_layers_module => {
            let pattern = call.bindings.get("pattern")?.clone();
            let kwargs: HashMap<String, String> = call
                .bindings
                .iter()
                .filter(|(name, _)| name.as_str() != "pattern" && name.as_str() != "reduction")
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect();
            Some(LayerKind::EinopsPattern {
                name: owner.to_string(),
                pattern,
                kwargs,
            })
        }
        _ => None,
    }
}

/// Strip a `self.` prefix from a ctor-arg dim expression's identifier
/// tokens (`"self.d_inner"` → `"d_inner"`, `"self.d_inner*2"` →
/// `"d_inner*2"`). `"self.d_inner"` would never match a same-named `d_inner`
/// jaxtyping annotation dim (a different vocabulary — see `dims_provably_mismatch`).
/// The analysis keeps the prefix until `ShapeCtx::layer_at` knows the value of
/// the attribute. This only strips the token-level prefix; it does not
/// consult the `self.<attr> = <ident>` alias map (`extract_self_attr_aliases*`
/// in `python_ast.rs`) the way `normalize_dim` in `analysis.rs` does, since
/// that map isn't available at layer-classification time without threading
/// it through every `classify_layer_call`/`resolve_layer_kind_for_call`
/// call site. In the overwhelmingly common case the attribute is assigned
/// from a same-named parameter (`self.d_inner = d_inner`), so the stripped
/// form already matches the annotation dim directly; the rarer
/// differently-named case is left to whatever later alias normalization
/// applies to the *shape* side of the comparison.
fn strip_self_prefix(dim: &str) -> String {
    if !dim.contains("self.") {
        return dim.to_string();
    }
    let mut result = String::with_capacity(dim.len());
    let mut i = 0;
    while i < dim.len() {
        if dim[i..].starts_with("self.") {
            i += 5; // skip "self."
            continue;
        }
        let ch = dim[i..].chars().next().unwrap();
        result.push(ch);
        i += ch.len_utf8();
    }
    result
}

pub(crate) fn strip_self_prefixes(kind: LayerKind) -> LayerKind {
    map_layer_dims(kind, &strip_self_prefix)
}

/// Apply `f` to every `String` dim field of a `LayerKind`,
/// recursing into `Sequential`'s children and `EinopsPattern`'s kwarg
/// values. `name`/`pattern` fields (class names, einops pattern text) are
/// left untouched — they aren't dim expressions.
pub(crate) fn map_layer_dims(kind: LayerKind, f: &dyn Fn(&str) -> String) -> LayerKind {
    match kind {
        LayerKind::Linear {
            in_features,
            out_features,
        } => LayerKind::Linear {
            in_features: f(&in_features),
            out_features: f(&out_features),
        },
        LayerKind::Conv {
            spatial_rank,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
        } => LayerKind::Conv {
            spatial_rank,
            in_channels: f(&in_channels),
            out_channels: f(&out_channels),
            kernel_size: f(&kernel_size),
            stride: f(&stride),
            padding: f(&padding),
            dilation: f(&dilation),
        },
        LayerKind::Dense { features } => LayerKind::Dense {
            features: f(&features),
        },
        LayerKind::FlaxConv {
            features,
            spatial_rank,
            kernel_size,
            padding,
        } => LayerKind::FlaxConv {
            features: f(&features),
            spatial_rank,
            kernel_size: f(&kernel_size),
            padding: f(&padding),
        },
        LayerKind::Pool {
            name,
            spatial_rank,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        } => LayerKind::Pool {
            name,
            spatial_rank,
            kernel_size: f(&kernel_size),
            stride: f(&stride),
            padding: f(&padding),
            dilation: f(&dilation),
            ceil_mode: f(&ceil_mode),
        },
        LayerKind::AdaptivePool {
            name,
            spatial_rank,
            output_size,
        } => LayerKind::AdaptivePool {
            name,
            spatial_rank,
            output_size: f(&output_size),
        },
        LayerKind::MultiheadAttention {
            feature_dim,
            num_heads,
            batch_first,
            added_keys,
            returns_weights,
            output_size,
        } => LayerKind::MultiheadAttention {
            feature_dim: f(&feature_dim),
            num_heads: f(&num_heads),
            batch_first: f(&batch_first),
            added_keys: f(&added_keys),
            returns_weights,
            output_size: f(&output_size),
        },
        LayerKind::Embedding { embedding_size } => LayerKind::Embedding {
            embedding_size: f(&embedding_size),
        },
        LayerKind::ShapePreserving { name } => LayerKind::ShapePreserving { name },
        LayerKind::Flatten { start_dim, end_dim } => LayerKind::Flatten {
            start_dim: f(&start_dim),
            end_dim: f(&end_dim),
        },
        LayerKind::Unflatten { dim, sizes } => LayerKind::Unflatten {
            dim: f(&dim),
            sizes: f(&sizes),
        },
        LayerKind::Upsample { scale_factor, size } => LayerKind::Upsample {
            scale_factor: scale_factor.map(|s| f(&s)),
            size: size.map(|s| f(&s)),
        },
        LayerKind::ConvTranspose {
            spatial_rank,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
        } => LayerKind::ConvTranspose {
            spatial_rank,
            in_channels: f(&in_channels),
            out_channels: f(&out_channels),
            kernel_size: f(&kernel_size),
            stride: f(&stride),
            padding: f(&padding),
            output_padding: f(&output_padding),
            dilation: f(&dilation),
        },
        LayerKind::Rnn {
            name,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            proj_size,
            batch_first,
        } => LayerKind::Rnn {
            name,
            input_size: f(&input_size),
            hidden_size: f(&hidden_size),
            num_layers: f(&num_layers),
            bidirectional: f(&bidirectional),
            proj_size: f(&proj_size),
            batch_first: f(&batch_first),
        },
        LayerKind::RnnCell {
            name,
            input_size,
            hidden_size,
        } => LayerKind::RnnCell {
            name,
            input_size: f(&input_size),
            hidden_size: f(&hidden_size),
        },
        LayerKind::PixelShuffle { upscale_factor } => LayerKind::PixelShuffle {
            upscale_factor: f(&upscale_factor),
        },
        LayerKind::PixelUnshuffle { downscale_factor } => LayerKind::PixelUnshuffle {
            downscale_factor: f(&downscale_factor),
        },
        LayerKind::Pad {
            name,
            spatial_rank,
            padding,
        } => LayerKind::Pad {
            name,
            spatial_rank,
            padding: f(&padding),
        },
        LayerKind::Bilinear {
            in1_features,
            in2_features,
            out_features,
        } => LayerKind::Bilinear {
            in1_features: f(&in1_features),
            in2_features: f(&in2_features),
            out_features: f(&out_features),
        },
        LayerKind::CosineSimilarity { dim } => LayerKind::CosineSimilarity {
            dim: f(&dim),
        },
        LayerKind::Mlp { in_size, out_size } => LayerKind::Mlp {
            in_size: f(&in_size),
            out_size: f(&out_size),
        },
        LayerKind::Unfold {
            kernel_size,
            stride,
            padding,
            dilation,
        } => LayerKind::Unfold {
            kernel_size: f(&kernel_size),
            stride: f(&stride),
            padding: f(&padding),
            dilation: f(&dilation),
        },
        LayerKind::Fold {
            output_size,
            kernel_size,
        } => LayerKind::Fold {
            output_size: f(&output_size),
            kernel_size: f(&kernel_size),
        },
        LayerKind::Sequential { children } => LayerKind::Sequential {
            children: children.into_iter().map(|child| map_layer_dims(child, f)).collect(),
        },
        LayerKind::EinopsPattern {
            name,
            pattern,
            kwargs,
        } => LayerKind::EinopsPattern {
            name,
            pattern,
            kwargs: kwargs
                .into_iter()
                .map(|(k, v)| (k, f(&v)))
                .collect(),
        },
        loss @ LayerKind::Loss { .. } => loss,
    }
}

/// Spatial rank from a layer class name's `1d`/`2d`/`3d` suffix.
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
/// `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`,
/// `dilation`). Extra
/// trailing params (e.g. `groups`, `use_bias`) are present only to
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
        || (module == "linen" && framework == "flax")
        || (framework == "einops"
            && module == "layers"
            && parts.len() >= 4
            && (parts[2] == "torch" || parts[2] == "flax"));
    if !known_framework {
        return None;
    }
    let class_name = parts.last()?.as_str();
    let params: &[&str] = match class_name {
        "Dense" => &["self", "features", "use_bias"],
        "Conv" if framework == "flax" => {
            &["self", "features", "kernel_size", "strides", "padding", "input_dilation", "kernel_dilation"]
        }
        "Linear" => &["self", "in_features", "out_features", "use_bias"],
        "Embedding" => &["self", "num_embeddings", "embedding_size"],
        "Embed" if framework == "flax" => &["self", "num_embeddings", "features"],
        "MaxPool1d" | "MaxPool2d" | "MaxPool3d" | "AvgPool1d" | "AvgPool2d" | "AvgPool3d"
            if framework == "equinox" =>
        {
            &["self", "kernel_size", "stride", "padding", "use_ceil"]
        }
        "MaxPool1d" | "MaxPool2d" | "MaxPool3d" => {
            &["self", "kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"]
        }
        "AvgPool1d" | "AvgPool2d" | "AvgPool3d" => &["self", "kernel_size", "stride", "padding", "ceil_mode"],
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
        "MultiheadAttention" => &[
            "self",
            "embed_dim",
            "num_heads",
            "dropout",
            "bias",
            "add_bias_kv",
            "add_zero_attn",
            "kdim",
            "vdim",
            "batch_first",
        ],
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
        "MultiHeadDotProductAttention" | "MultiHeadAttention" | "SelfAttention" if framework == "flax" => {
            &["self", "num_heads", "dtype", "param_dtype", "qkv_features", "out_features"]
        }
        name if SHAPE_PRESERVING_LAYERS.contains(&name) => &["self"],
        "Flatten" => &["self", "start_dim", "end_dim"],
        "Unflatten" => &["self", "dim", "sizes"],
        "Upsample" => &["self", "size", "scale_factor", "mode", "align_corners"],
        // equinox puts `output_padding` before `padding`, torch after.
        "ConvTranspose1d" | "ConvTranspose2d" | "ConvTranspose3d" if framework == "equinox" => &[
            "self",
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "output_padding",
            "padding",
            "dilation",
            "groups",
            "use_bias",
        ],
        "ConvTranspose1d" | "ConvTranspose2d" | "ConvTranspose3d" => &[
            "self",
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "groups",
            "bias",
            "dilation",
        ],
        "RNN" => &["self", "input_size", "hidden_size", "num_layers", RNN_POSITION_4],
        "LSTM" | "GRU" => &[
            "self",
            "input_size",
            "hidden_size",
            "num_layers",
            "bias",
            "batch_first",
            "dropout",
            "bidirectional",
            "proj_size",
        ],
        // torch/equinox agree on (input_size, hidden_size, ...) for the
        // cells; only the first two are ever read.
        "RNNCell" | "LSTMCell" | "GRUCell" => &[
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
        "MSELoss" | "L1Loss" if framework == "torch" => {
            &["self", "size_average", "reduce", "reduction"]
        }
        "SmoothL1Loss" if framework == "torch" => {
            &["self", "size_average", "reduce", "reduction", "beta"]
        }
        "HuberLoss" if framework == "torch" => &["self", "reduction", "delta"],
        "BCELoss" if framework == "torch" => {
            &["self", "weight", "size_average", "reduce", "reduction"]
        }
        "BCEWithLogitsLoss" if framework == "torch" => {
            &["self", "weight", "size_average", "reduce", "reduction", "pos_weight"]
        }
        "CrossEntropyLoss" if framework == "torch" => &[
            "self",
            "weight",
            "size_average",
            "ignore_index",
            "reduce",
            "reduction",
            "label_smoothing",
        ],
        "NLLLoss" if framework == "torch" => &[
            "self",
            "weight",
            "size_average",
            "ignore_index",
            "reduce",
            "reduction",
        ],
        "Bilinear" => &["self", "in1_features", "in2_features", "out_features", "use_bias"],
        "CosineSimilarity" => &["self", "dim", "eps"],
        "MLP" if framework == "equinox" => &[
            "self",
            "in_size",
            "out_size",
            "width_size",
            "depth",
        ],
        "Rearrange" if framework == "einops" => &["self", "pattern"],
        "Reduce" if framework == "einops" => &["self", "pattern", "reduction"],
        // torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1);
        // torch.nn.Fold(output_size, kernel_size, dilation=1, padding=0, stride=1).
        "Unfold" => &["self", "kernel_size", "dilation", "padding", "stride"],
        "Fold" => &["self", "output_size", "kernel_size", "dilation", "padding", "stride"],
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
    let Some(bindings) = bind_call_arguments(&signature, &arguments) else {
        return Ok(None);
    };

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
    classify_layer_text(&resolved)
}

#[cfg(test)]
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
    let imports = ImportScopes::new(node, text)?;
    let records = extract_layer_assignments_scoped(
        node,
        text,
        &imports,
        search_roots,
        read_file,
        max_depth,
        None,
    )?;
    let mut layers = HashMap::new();
    for rec in records {
        layers.insert(rec.name, strip_self_prefixes(rec.kind));
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
    // Composite containers need the raw argument nodes (to recurse into
    // nested constructor calls), which `ResolvedCallSignature` doesn't carry
    // (its `arguments`/`bindings` are already flattened to strings). Try
    // these node-level forms first; both return `Ok(None)` for calls that
    // aren't their target class, falling through to the normal catalog/disk
    // path unchanged.
    if let Some(kind) = try_classify_sequential(
        call,
        node,
        text,
        import_map,
        search_roots,
        read_file,
        max_depth,
        cache,
    )? {
        return Ok(Some(kind));
    }
    if let Some(kind) = try_classify_inner_layer_wrapper(
        call,
        node,
        text,
        import_map,
        search_roots,
        read_file,
        max_depth,
        cache,
    )? {
        return Ok(Some(kind));
    }

    // Catalog-first: hardcoded equinox.nn.* / torch.nn.* signatures bypass
    // disk resolution. Falls through to resolve_call_signature for
    // user-defined layers and frameworks not in the catalog.
    let resolved_call = match try_catalog_signature(call, node, text, import_map)? {
        Some(c) => Some(c),
        None => resolve_call_signature_with_node(
            call,
            Some(node),
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
    Ok(classify_layer_text(&resolved_call))
}

/// Named children of a `nn.Sequential`/`SpectralNorm`/`WeightNorm` ctor's
/// argument list that are candidates for nested layer-constructor calls.
///
/// torch's `Sequential` takes variadic positional layers directly:
/// `nn.Sequential(nn.Linear(...), nn.ReLU())`. equinox's `Sequential` takes a
/// single `[layer, ...]` list literal instead: `eqx.nn.Sequential([eqx.nn.Linear(...), ...])`.
/// When the args node holds exactly one positional arg and it's a `list`,
/// unwrap it; otherwise use the args node's own named children.
fn sequential_child_nodes(args_node: Node) -> Vec<Node> {
    if args_node.named_child_count() == 1
        && let Some(only) = args_node.named_child(0)
        && only.kind() == "list"
    {
        return (0..only.named_child_count())
            .filter_map(|i| only.named_child(i as u32))
            .collect();
    }
    (0..args_node.named_child_count())
        .filter_map(|i| args_node.named_child(i as u32))
        .collect()
}

/// Resolve an AST node believed to be a nested layer-constructor call (a
/// `Sequential`/`SpectralNorm`/`WeightNorm` ctor argument) to its `LayerKind`,
/// reusing the same catalog-first/disk-fallback resolution as top-level layer
/// assignments. `Ok(None)` for anything that isn't a plain `call` node
/// (arbitrary callable, `*args` unpacking, a bare identifier, …) — an honest
/// unknown rather than a guess.
#[allow(clippy::too_many_arguments)]
fn classify_nested_layer_node<F>(
    child: Node,
    root: Node,
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
    if child.kind() != "call" {
        return Ok(None);
    }
    let Some(func_node) = child.child_by_field_name("function") else {
        return Ok(None);
    };
    let Some(inner_args_node) = child.child_by_field_name("arguments") else {
        return Ok(None);
    };
    let Ok(target_text) = func_node.utf8_text(text.as_bytes()) else {
        return Ok(None);
    };
    let inner_call = CallInfo {
        variable: String::new(),
        target: target_text.to_string(),
        args_node_range: inner_args_node.range(),
    };
    resolve_layer_kind_for_call(
        &inner_call,
        root,
        text,
        import_map,
        search_roots,
        read_file,
        max_depth,
        cache,
    )
}

/// `torch.nn.Sequential(l1, l2, ...)` / `equinox.nn.Sequential([l1, l2, ...])`:
/// classify each child ctor argument via `classify_nested_layer_node`. If any
/// child fails to classify (`Ok(None)`), the whole `Sequential` is honestly
/// unclassified — `Ok(None)`, not a guess. `Ok(None)` (not an error) when
/// `call` isn't a `Sequential` ctor at all, so callers can fall through to
/// their normal resolution path unchanged.
#[allow(clippy::too_many_arguments)]
fn try_classify_sequential<F>(
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
    let target = resolve_call_target(&call.target, import_map);
    if target.dots != 0 || target.parts.len() < 3 {
        return Ok(None);
    }
    let is_sequential = target.parts.last().map(String::as_str) == Some("Sequential")
        && ((target.parts[0] == "torch" && target.parts[1] == "nn")
            || (target.parts[0] == "equinox" && target.parts[1] == "nn"));
    if !is_sequential {
        return Ok(None);
    }

    let Some(args_node) =
        node.descendant_for_byte_range(call.args_node_range.start_byte, call.args_node_range.end_byte)
    else {
        return Ok(None);
    };

    let mut children = Vec::new();
    for child in sequential_child_nodes(args_node) {
        let Some(kind) = classify_nested_layer_node(
            child,
            node,
            text,
            import_map,
            search_roots,
            read_file,
            max_depth,
            cache,
        )?
        else {
            return Ok(None);
        };
        children.push(kind);
    }
    Ok(Some(LayerKind::Sequential { children }))
}

/// `equinox.nn.SpectralNorm(layer, ...)` / `equinox.nn.WeightNorm(layer, ...)`:
/// wrapper modules around one inner layer, tracked only for shape purposes —
/// classify the first ctor argument and delegate transparently (the wrapper
/// itself carries no shape information of its own). `Ok(None)` when `call`
/// isn't one of these two wrapper ctors, or when the first argument isn't a
/// resolvable layer-constructor call.
#[allow(clippy::too_many_arguments)]
fn try_classify_inner_layer_wrapper<F>(
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
    let target = resolve_call_target(&call.target, import_map);
    if target.dots != 0 || target.parts.len() < 3 {
        return Ok(None);
    }
    let is_wrapper = target.parts[0] == "equinox"
        && target.parts[1] == "nn"
        // `SpectralNorm` gives `(output, state)`, so only `WeightNorm` is transparent.
        && target.parts.last().map(String::as_str) == Some("WeightNorm");
    if !is_wrapper {
        return Ok(None);
    }

    let Some(args_node) =
        node.descendant_for_byte_range(call.args_node_range.start_byte, call.args_node_range.end_byte)
    else {
        return Ok(None);
    };
    let Some(first_arg) = args_node.named_child(0) else {
        return Ok(None);
    };
    classify_nested_layer_node(
        first_arg,
        node,
        text,
        import_map,
        search_roots,
        read_file,
        max_depth,
        cache,
    )
}

pub fn extract_layer_assignments_scoped<F>(
    node: Node,
    text: &str,
    imports: &ImportScopes,
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
            imports.at(call.args_node_range.start_byte),
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
/// `__init__`), keeping for each binding the byte range of its enclosing
/// `class_definition`, so `self.fc` in class A and `self.fc` in class B
/// resolve independently at their respective call sites.
pub fn extract_self_attr_layers_by_class<F>(
    node: Node,
    text: &str,
    imports: &ImportScopes,
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
            imports.at(call.args_node_range.start_byte),
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
            byte: call.args_node_range.start_byte,
        });
    }

    Ok(layers)
}

/// Apply a signed integer offset to a symbolic expression, constant-folding
/// when the expression itself is an integer. An overflow gives `UNKNOWN_DIM`.
fn apply_offset(expr: &str, offset: isize) -> String {
    if offset == 0 {
        return expr.to_string();
    }
    // If expr is itself an integer, fold the whole thing
    if let Ok(val) = expr.parse::<isize>() {
        return val.checked_add(offset).map_or(UNKNOWN_DIM.to_string(), |sum| sum.to_string());
    }
    if offset > 0 {
        format!("{}+{}", expr, offset)
    } else {
        format!("{}-{}", expr, offset.unsigned_abs())
    }
}

/// The text of a string literal.
pub(crate) fn string_text(value: &str) -> Option<&str> {
    let value = value.trim();
    ['"', '\'']
        .iter()
        .find_map(|quote| value.strip_prefix(*quote)?.strip_suffix(*quote))
}

/// Whether the text of a size argument is a tuple or a list.
fn is_sequence(value: &str) -> bool {
    value.starts_with(['(', '['])
}

/// The text of the `padding` of a constructor. One name gets `PADDING_NAME`.
fn padding_argument(call: &ResolvedCallSignature) -> String {
    let Some(padding) = call.bindings.get("padding") else {
        return "0".to_string();
    };
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'.';
    let name = padding.bytes().all(is_word)
        && !padding.starts_with(|c: char| c.is_ascii_digit())
        && !matches!(padding.as_str(), "True" | "False" | "None");
    if name { format!("{PADDING_NAME}{padding}") } else { padding.clone() }
}

/// What the `padding` of a convolution does to a spatial dim.
enum ConvPadding<'a> {
    KeepsSize,
    Amount(&'a str),
    Unknown,
}

/// `"same"` keeps the size only at stride 1: torch refuses another stride, and
/// equinox gives a size that is not modelled.
fn conv_padding<'a>(padding: &'a str, stride: &str) -> ConvPadding<'a> {
    if padding.starts_with(PADDING_NAME) {
        return ConvPadding::Unknown;
    }
    let Some(text) = string_text(padding) else {
        return ConvPadding::Amount(padding);
    };
    match text.to_ascii_lowercase().as_str() {
        "valid" => ConvPadding::Amount("0"),
        "same" | "same_lower" if stride == "1" => ConvPadding::KeepsSize,
        _ => ConvPadding::Unknown,
    }
}

/// Compute the output spatial dimension for a convolution.
/// Formula: floor((L + 2*padding - kernel_size) / stride) + 1
///
/// If all values and the spatial dim are integers, compute concretely: a
/// window larger than the input has no size, so the dim is `UNPROVEN_DIM`.
/// Otherwise build a symbolic string with constant-folding where possible.
fn conv_spatial_dim(spatial_dim: &str, kernel_size: &str, stride: &str, padding: &str) -> String {
    let inputs = [spatial_dim, kernel_size, stride, padding];
    let [spatial_dim, kernel_size, stride, padding] = inputs.map(without_literal_mark);
    // Try fully concrete computation
    if let (Ok(l), Ok(k), Ok(s), Ok(p)) = (
        spatial_dim.parse::<i64>(),
        kernel_size.parse::<i64>(),
        stride.parse::<i64>(),
        padding.parse::<i64>(),
    ) && s > 0
    {
        let size = p.checked_mul(2).and_then(|pad| l.checked_add(pad)?.checked_sub(k)).and_then(|span| {
            if span < 0 { Some(-1) } else { (span / s).checked_add(1) }
        });
        return size.map_or(UNKNOWN_DIM.to_string(), |size| literal_mark(&inputs.concat(), size));
    }

    // Symbolic: (L + 2*p - k) / s + 1
    // Constant-fold 2*padding - kernel_size (and +1 when stride==1) into a
    // single offset whenever both padding and kernel_size are concrete ints.
    let stride_val: Option<isize> = stride.parse::<isize>().ok();
    let is_stride_one = stride_val == Some(1);

    if let (Ok(p), Ok(k)) = (padding.parse::<isize>(), kernel_size.parse::<isize>()) {
        // Fully concrete offset(s)
        let Some(offset) = p.checked_mul(2).and_then(|pad| pad.checked_sub(k)) else {
            return UNKNOWN_DIM.to_string();
        };
        if is_stride_one {
            // Total offset: 2*p - k + 1, folded into one operation
            let Some(offset) = offset.checked_add(1) else {
                return UNKNOWN_DIM.to_string();
            };
            apply_offset(spatial_dim, offset)
        } else {
            // Inner offset: 2*p - k, then divide by stride and add 1
            let inner = apply_offset(spatial_dim, offset);
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
            let Some(pad) = p.checked_mul(2) else {
                return UNKNOWN_DIM.to_string();
            };
            inner = apply_offset(&inner, pad);
        } else {
            inner = format!("{}+2*{}", inner, padding);
        }

        // Subtract kernel_size
        if let Ok(k) = kernel_size.parse::<isize>() {
            let Some(kernel) = k.checked_neg() else {
                return UNKNOWN_DIM.to_string();
            };
            inner = apply_offset(&inner, kernel);
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
/// Formula: `out = (in-1)*stride - 2*padding + kernel + output_padding`, with
/// `kernel` the effective (dilated) kernel size.
///
/// Fully concrete when every input is a literal int; otherwise built
/// symbolically with constant-folding where possible, mirroring
/// `conv_spatial_dim`'s approach. An overflow gives `UNKNOWN_DIM`.
fn conv_transpose_spatial_dim(
    spatial_dim: &str,
    kernel_size: &str,
    stride: &str,
    padding: &str,
    output_padding: &str,
) -> String {
    let inputs = [spatial_dim, kernel_size, stride, padding, output_padding];
    let [spatial_dim, kernel_size, stride, padding, output_padding] = inputs.map(without_literal_mark);
    if let (Ok(l), Ok(k), Ok(s), Ok(p), Ok(op)) = (
        spatial_dim.parse::<i64>(),
        kernel_size.parse::<i64>(),
        stride.parse::<i64>(),
        padding.parse::<i64>(),
        output_padding.parse::<i64>(),
    ) {
        let size = l.checked_sub(1).and_then(|l| l.checked_mul(s)).and_then(|scaled| {
            scaled.checked_sub(p.checked_mul(2)?)?.checked_add(k)?.checked_add(op)
        });
        return size.map_or(UNKNOWN_DIM.to_string(), |size| literal_mark(&inputs.concat(), size));
    }

    // Symbolic: (L-1)*stride - 2*padding + kernel + output_padding
    let stride_val: Option<isize> = stride.parse::<isize>().ok();
    let scaled = match stride_val {
        Some(1) => apply_offset(spatial_dim, -1),
        Some(s) => format!("({}-1)*{}", spatial_dim, s),
        None => format!("({}-1)*{}", spatial_dim, stride),
    };

    let mut result = scaled;
    if let Ok(p) = padding.parse::<isize>() {
        let Some(pad) = p.checked_mul(-2) else {
            return UNKNOWN_DIM.to_string();
        };
        result = apply_offset(&result, pad);
    } else {
        result = format!("{}-2*{}", result, padding);
    }
    for added in [kernel_size, output_padding] {
        result = match added.parse::<isize>() {
            Ok(value) => apply_offset(&result, value),
            Err(_) => format!("{}+{}", result, added),
        };
        if result == UNKNOWN_DIM {
            return result;
        }
    }
    result
}

/// torch pooling with `ceil_mode=True`: `ceil((L + 2*padding - kernel) /
/// stride) + 1`, one less when the last window starts in the right padding.
/// Only integers are modelled: any other input gives `UNKNOWN_DIM`.
fn pool_ceil_spatial_dim(spatial_dim: &str, kernel_size: &str, stride: &str, padding: &str) -> String {
    let inputs = [spatial_dim, kernel_size, stride, padding];
    let [l, k, s, p] = inputs.map(|dim| without_literal_mark(dim).parse::<i64>().ok());
    let (Some(l), Some(k), Some(s), Some(p)) = (l, k, s, p) else {
        return UNKNOWN_DIM.to_string();
    };
    if s <= 0 {
        return UNKNOWN_DIM.to_string();
    }
    let size = p.checked_mul(2).and_then(|pad| l.checked_add(pad)?.checked_sub(k)).and_then(|span| {
        if span < 0 {
            return Some(-1);
        }
        let windows = span.checked_add(s - 1)? / s + 1;
        let last_start = (windows - 1).checked_mul(s)?;
        Some(if last_start >= l.checked_add(p)? { windows - 1 } else { windows })
    });
    size.map_or(UNKNOWN_DIM.to_string(), |size| literal_mark(&inputs.concat(), size))
}

/// The shapes that `torch.nn.RNN`/`LSTM`/`GRU` gives for one input shape.
pub(crate) struct RnnOutputs {
    /// The last dim of `output`.
    pub output: String,
    /// `h_n`; `None` when the input rank is not 2 or 3.
    pub hidden: Option<Vec<String>>,
    /// `c_n` of an LSTM, the same as `hidden` without a projection.
    pub cell: Option<Vec<String>>,
}

/// The output last dim and the final states of an `Rnn` layer. A dim that a
/// non-literal option controls is `UNKNOWN_DIM`.
pub(crate) fn rnn_outputs(kind: &LayerKind, input_shape: &[String]) -> RnnOutputs {
    let LayerKind::Rnn { hidden_size, num_layers, bidirectional, proj_size, batch_first, .. } = kind else {
        unreachable!("rnn_outputs takes only Rnn");
    };
    let unknown = || UNKNOWN_DIM.to_string();
    let directions = match bidirectional.as_str() {
        "False" => Some(1),
        "True" => Some(2),
        _ => None,
    };
    let h_out = match without_literal_mark(proj_size).parse::<u64>() {
        Ok(0) => hidden_size.clone(),
        Ok(_) => proj_size.clone(),
        Err(_) => unknown(),
    };
    let output = match directions {
        Some(1) => h_out.clone(),
        Some(d) => mul_dim(&h_out, d),
        None => unknown(),
    };
    let layers = without_literal_mark(num_layers).parse::<u64>().ok().filter(|n| *n > 0);
    let leading = match (directions, layers) {
        (Some(d), Some(n)) => n.checked_mul(d).map_or_else(unknown, |product| product.to_string()),
        _ => unknown(),
    };
    let batch = match (input_shape.len(), batch_first.as_str()) {
        (2, _) => None,
        (3, "False") => Some(input_shape[1].clone()),
        (3, "True") => Some(input_shape[0].clone()),
        (3, _) => Some(unknown()),
        _ => return RnnOutputs { output, hidden: None, cell: None },
    };
    let state = |last: &String| {
        let mut state = vec![leading.clone()];
        state.extend(batch.clone());
        state.push(last.clone());
        state
    };
    RnnOutputs { output, hidden: Some(state(&h_out)), cell: Some(state(hidden_size)) }
}

/// The attention weights of `torch.nn.MultiheadAttention` for a query and a
/// key shape: `(N, L, S)`, or `(N, num_heads, L, S)` when `average` is
/// `"False"`, without `N` for an unbatched input. `None` when the call does
/// not return weights or an option is not a literal.
pub(crate) fn attention_weights(
    kind: &LayerKind,
    query: &[String],
    key: &[String],
    need_weights: &str,
    average: &str,
) -> Option<Vec<String>> {
    let LayerKind::MultiheadAttention { num_heads, batch_first, added_keys, returns_weights: true, .. } = kind else {
        return None;
    };
    if need_weights != "True" || query.len() != key.len() {
        return None;
    }
    let added = without_literal_mark(added_keys).parse::<isize>().ok()?;
    let (batch, l, s) = match (query.len(), batch_first.as_str()) {
        (2, _) => (None, &query[0], &key[0]),
        (3, "False") => (Some(&query[1]), &query[0], &key[0]),
        (3, "True") => (Some(&query[0]), &query[1], &key[1]),
        _ => return None,
    };
    let s = apply_offset(s, added);
    let mut weights: Vec<String> = batch.into_iter().cloned().collect();
    match average {
        "True" => {}
        "False" if !num_heads.contains(UNPROVEN_DIM) => weights.push(num_heads.clone()),
        _ => return None,
    }
    weights.extend([l.clone(), s]);
    Some(weights)
}

/// The integer of a `LITERAL_DIM` dim, and any other dim as it is.
fn without_literal_mark(dim: &str) -> &str {
    literal_dim(dim).unwrap_or(dim)
}

/// Multiply a dim by a literal factor: exact when the dim is a literal int,
/// otherwise a symbolic `dim*factor` string. An overflow gives `UNKNOWN_DIM`.
fn mul_dim(dim: &str, factor: u64) -> String {
    if dim == UNKNOWN_DIM {
        dim.to_string()
    } else if let Ok(v) = dim.parse::<u64>() {
        v.checked_mul(factor).map_or(UNKNOWN_DIM.to_string(), |product| product.to_string())
    } else {
        format!("{}*{}", factor_text(dim), factor)
    }
}

/// Divide a dim by a literal factor. Errs when the dim is a literal int that
/// isn't evenly divisible (a provable contradiction); otherwise a symbolic
/// `dim/factor` string (unprovable, trusted as-is).
fn div_dim(dim: &str, factor: u64) -> Result<String, String> {
    if factor == 0 {
        Ok(UNKNOWN_DIM.to_string())
    } else if let Ok(v) = dim.parse::<u64>() {
        if v % factor == 0 {
            Ok((v / factor).to_string())
        } else {
            Err(format!("{} is not evenly divisible by {}", v, factor))
        }
    } else {
        Ok(format!("{}/{}", factor_text(dim), factor))
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
        let product = values.iter().try_fold(1u64, |product, value| product.checked_mul(*value));
        product.map_or(UNKNOWN_DIM.to_string(), |product| product.to_string())
    } else {
        joined_dim(&dims.iter().map(|dim| factor_text(dim)).collect::<Vec<_>>(), "*")
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
        split_top_level(inner.trim_end_matches(','))
            .into_iter()
            .map(|s| s.trim().to_string())
            .collect(),
    )
}

/// Parse a per-axis geometry arg (`kernel_size`/`stride`/`padding`/
/// `dilation` for `Unfold`/`Fold`) that torch accepts as either a single
/// value (applied to both H and W) or an explicit 2-tuple/list. Returns
/// `None` for anything else (e.g. a `>2`-length tuple), so the caller can
/// treat it as unknown rather than guessing.
fn parse_pair_arg(text: &str) -> Option<(String, String)> {
    let trimmed = text.trim();
    if trimmed.starts_with('(') || trimmed.starts_with('[') {
        let parts = parse_dim_sequence(trimmed)?;
        if parts.len() == 2 {
            Some((parts[0].clone(), parts[1].clone()))
        } else {
            None
        }
    } else if trimmed.is_empty() {
        None
    } else {
        Some((trimmed.to_string(), trimmed.to_string()))
    }
}

/// Dilated ("effective") kernel size `d*(k-1)+1`, the same substitution
/// `known_functions.rs`'s `LaxConvGeneralDilated`/`FunctionalConv*` rules use
/// before reusing the plain conv output-dim formula. Exact when `kernel`/
/// `dilation` are both literal ints; when `dilation` is literal `1` the
/// effective size is `kernel` unchanged regardless of whether `kernel`
/// itself is literal (covering the overwhelmingly common default even for
/// symbolic kernel sizes). Any other non-literal combination isn't
/// foldable and returns `None`. An overflow gives `UNKNOWN_DIM`.
fn effective_kernel_size(kernel: &str, dilation: &str) -> Option<String> {
    if let (Ok(k), Ok(d)) = (kernel.trim().parse::<i64>(), dilation.trim().parse::<i64>()) {
        let size = k.checked_sub(1).and_then(|span| d.checked_mul(span)?.checked_add(1));
        return Some(size.map_or(UNKNOWN_DIM.to_string(), |size| size.to_string()));
    }
    if dilation.trim() == "1" {
        return Some(kernel.to_string());
    }
    None
}

/// Divide a dim by the product of one or more (possibly symbolic) factors.
/// Errs only when the dim and every factor are literal ints and the
/// division doesn't come out even (a provable contradiction); otherwise a
/// symbolic `dim/(f0*f1*...)` string — mirrors `div_dim`'s policy, extended
/// to a multi-factor (possibly symbolic) divisor for `Fold`'s channel-dim
/// recovery.
fn div_dim_by_dims(dim: &str, factors: &[String]) -> Result<String, String> {
    if let Some(dim_val) = dim.parse::<u64>().ok()
        && let Some(factor_vals) = factors
            .iter()
            .map(|f| f.parse::<u64>().ok())
            .collect::<Option<Vec<_>>>()
    {
        let Some(divisor) = factor_vals.iter().try_fold(1u64, |product, value| product.checked_mul(*value)) else {
            return Ok(UNKNOWN_DIM.to_string());
        };
        if divisor != 0 && dim_val % divisor == 0 {
            return Ok((dim_val / divisor).to_string());
        } else if divisor != 0 {
            return Err(format!("{} is not evenly divisible by {}", dim_val, divisor));
        }
    }
    Ok(format!("{}/({})", factor_text(dim), product_dims(factors)))
}

/// Shared minimum-rank guard used by every layer whose validity depends on
/// input rank (Conv/Pool/ConvTranspose/shape-preserving families). Returns
/// the standard "requires input with at least N dims" error when the input
/// is too low-rank.
fn check_min_rank(
    layer_name: &str,
    layer: &str,
    input: &str,
    input_rank: usize,
    min_rank: usize,
) -> Result<(), String> {
    if input_rank < min_rank {
        return Err(format!(
            "{} layer '{}' requires input with at least {} dims, got {} for '{}'",
            layer_name, layer, min_rank, input_rank, input
        ));
    }
    Ok(())
}

/// Replace the last dim with `output`. An `expected` last dim that provably
/// differs is an error.
fn replace_last_dim(
    name: &str,
    layer: &str,
    input: &str,
    input_shape: &[String],
    expected: Option<&String>,
    output: &str,
) -> Result<Option<Vec<String>>, String> {
    let Some(last_dim) = input_shape.last() else {
        return Err(format!("Cannot apply {} layer '{}' to scalar input '{}'", name, layer, input));
    };
    if let Some(expected) = expected
        && dims_provably_mismatch(expected, last_dim)
    {
        return Err(format!(
            "{} layer '{}' expected input last dim {}, got {} for '{}'",
            name, layer, expected, last_dim, input
        ));
    }
    let mut output_shape = input_shape.to_vec();
    *output_shape.last_mut().unwrap() = output.to_string();
    Ok(Some(output_shape))
}

/// Conv and ConvTranspose, channels-first: `[..., C, *spatial]`.
fn apply_conv_layer(
    kind: &LayerKind,
    layer: &str,
    input: &str,
    input_shape: &[String],
) -> Result<Option<Vec<String>>, String> {
    let (name, spatial_rank, in_channels, out_channels, kernel_size, stride, padding, dilation, output_padding) = match kind {
        LayerKind::Conv { spatial_rank, in_channels, out_channels, kernel_size, stride, padding, dilation } => {
            ("Conv", *spatial_rank, in_channels, out_channels, kernel_size, stride, padding, dilation, None)
        }
        LayerKind::ConvTranspose {
            spatial_rank,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
        } => ("ConvTranspose", *spatial_rank, in_channels, out_channels, kernel_size, stride, padding, dilation, Some(output_padding)),
        _ => unreachable!("apply_conv_layer takes only Conv and ConvTranspose"),
    };
    let layer_name = format!("{name}{spatial_rank}d");
    check_min_rank(&layer_name, layer, input, input_shape.len(), spatial_rank + 1)?;

    let channels_idx = input_shape.len() - spatial_rank - 1;
    let channels_dim = &input_shape[channels_idx];
    if dims_provably_mismatch(in_channels, channels_dim) {
        return Err(format!(
            "{} layer '{}' expected {} input channels, got {} for '{}'",
            layer_name, layer, in_channels, channels_dim, input
        ));
    }

    let mut output_shape = input_shape.to_vec();
    output_shape[channels_idx] = out_channels.clone();
    let padding = match conv_padding(padding, stride) {
        ConvPadding::KeepsSize => return Ok(Some(output_shape)),
        ConvPadding::Unknown => return Ok(None),
        ConvPadding::Amount(padding) => padding,
    };
    let spatial = &mut output_shape[channels_idx + 1..];
    let Some(dilations) = conv_dilations(dilation, spatial_rank) else {
        return Ok(None);
    };
    let output_paddings = match output_padding {
        None => None,
        Some(output_padding) => match per_axis(output_padding, spatial_rank) {
            Some(output_paddings) => Some(output_paddings),
            None => return Ok(None),
        },
    };
    for (axis, (dim, dilation)) in spatial.iter_mut().zip(&dilations).enumerate() {
        let Some(kernel) = effective_kernel_size(without_literal_mark(kernel_size), dilation) else {
            return Ok(None);
        };
        let kernel = if dilation == "1" { kernel_size } else { &kernel };
        *dim = match &output_paddings {
            Some(output_paddings) => {
                let output_padding = &output_paddings[axis];
                // torch refuses an output_padding that is not below stride or dilation.
                if let (Ok(op), Ok(s), Ok(d)) = (
                    output_padding.parse::<i64>(),
                    without_literal_mark(stride).parse::<i64>(),
                    dilation.parse::<i64>(),
                ) && (op < 0 || (op >= s && op >= d))
                {
                    return Ok(None);
                }
                conv_transpose_spatial_dim(dim, kernel, stride, padding, output_padding)
            }
            None => conv_spatial_dim(dim, kernel, stride, padding),
        };
    }
    Ok(Some(output_shape))
}

/// The dilation of each of `spatial_rank` axes. `None` when one is not known
/// as a positive integer.
fn conv_dilations(dilation: &str, spatial_rank: usize) -> Option<Vec<String>> {
    let dilations = per_axis(dilation, spatial_rank)?;
    dilations.iter().all(|axis| axis.parse::<i64>().is_ok_and(|d| d > 0)).then_some(dilations)
}

/// The value of each of `spatial_rank` axes: a scalar applies to every axis,
/// a sequence must have one value per axis.
fn per_axis(value: &str, spatial_rank: usize) -> Option<Vec<String>> {
    let values = if is_sequence(value) {
        parse_dim_sequence(value).filter(|axes| axes.len() == spatial_rank)?
    } else {
        vec![value.to_string(); spatial_rank]
    };
    Some(values.iter().map(|axis| without_literal_mark(axis).to_string()).collect())
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
///   LocalResponseNorm, Softmax2d → 3
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
        "LocalResponseNorm" | "Softmax2d" => Some(3),
        // Dropout, BatchNorm (equinox), ReLU, GELU, Sigmoid, Tanh, Softmax,
        // PReLU, Identity, AlphaDropout, TransformerEncoderLayer,
        // TransformerDecoderLayer accept any rank including scalars.
        _ => None,
    }
}

/// A layer's expected dim comes from constructor-argument text, a different
/// vocabulary than jaxtyping annotation dims (issue #47). A mismatch is only
/// an error when both sides are concrete integers; otherwise it's unprovable
/// and the layer's output shape is trusted.
pub(crate) fn dims_provably_mismatch(expected: &str, actual: &str) -> bool {
    !dims_canonically_equal(expected, actual)
        && without_literal_mark(expected).trim().parse::<u64>().is_ok()
        && without_literal_mark(actual).trim().parse::<u64>().is_ok()
}

pub fn apply_layer_application(
    app: &LayerApplication,
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    if let LayerKind::Loss { name, reduction } = &app.kind {
        let Some(target) = app.extra_inputs.first() else {
            return Ok(None);
        };
        return apply_loss_layer(name, reduction.as_deref(), &app.input, target, shapes);
    }
    let Some(input_shape) = shapes.shape(&app.input) else {
        return Ok(None);
    };
    apply_layer_kind(&app.kind, input_shape, &app.layer, &app.input)
}

/// Apply a `LayerKind`'s shape rule to a known input shape, given the names
/// used only for error-message text (`layer`, the layer's variable/label;
/// `input`, the input's variable/label). Factored out of
/// `apply_layer_application` so `LayerKind::Sequential` can thread a shape
/// through each child's rule in turn without needing a `ShapeLookup`/
/// `LayerApplication` for intermediate (unnamed) shapes. An output or an error
/// that depends on an `UNPROVEN_DIM` is unknown.
pub(crate) fn apply_layer_kind(
    kind: &LayerKind,
    input_shape: &[String],
    layer: &str,
    input: &str,
) -> Result<Option<Vec<String>>, String> {
    match apply_layer_rule(kind, input_shape, layer, input) {
        Ok(Some(output)) if output.iter().any(|dim| dim.contains(UNPROVEN_DIM)) => Ok(None),
        Ok(Some(output)) => Ok(bounded_shape(output)),
        Err(message) if message.contains(UNPROVEN_DIM) => Ok(None),
        result => result,
    }
}

fn apply_layer_rule(
    kind: &LayerKind,
    input_shape: &[String],
    layer: &str,
    input: &str,
) -> Result<Option<Vec<String>>, String> {
    match kind {
        LayerKind::Linear {
            in_features,
            out_features,
        } => replace_last_dim("Linear", layer, input, input_shape, Some(in_features), out_features),
        LayerKind::Conv { .. } | LayerKind::ConvTranspose { .. } => apply_conv_layer(kind, layer, input, input_shape),
        // torch returns an (output, weights) tuple: no shape here; tuple
        // unpacking is handled in analysis::tuple_rhs_shapes. equinox returns
        // `(query_seq, output_size)` for a `(query_seq, query_size)` query.
        LayerKind::MultiheadAttention { returns_weights: true, .. } => Ok(None),
        LayerKind::MultiheadAttention { feature_dim, output_size, .. } => {
            if input_shape.len() != 2 {
                return Ok(None);
            }
            replace_last_dim("MultiheadAttention", layer, input, input_shape, Some(feature_dim), output_size)
        }
        // flax channels-last: last dim becomes `features`, nothing to check
        // (flax infers the input width at runtime).
        LayerKind::Dense { features } => replace_last_dim("Dense", layer, input, input_shape, None, features),
        LayerKind::FlaxConv {
            features,
            spatial_rank,
            kernel_size,
            padding,
        } => {
            check_min_rank("Conv", layer, input, input_shape.len(), spatial_rank + 1)?;
            let mut output_shape = input_shape.to_vec();
            let last = output_shape.len() - 1;
            output_shape[last] = features.clone();
            let wraps = string_text(padding).is_some_and(|text| {
                ["CIRCULAR", "REFLECT", "CAUSAL"].contains(&text.to_ascii_uppercase().as_str())
            });
            let padding = if wraps { ConvPadding::KeepsSize } else { conv_padding(padding, "1") };
            match padding {
                ConvPadding::KeepsSize => Ok(Some(output_shape)),
                ConvPadding::Unknown => Ok(None),
                ConvPadding::Amount(padding) => {
                    let kernels = if kernel_size.starts_with(['(', '[']) {
                        parse_dim_sequence(kernel_size)
                    } else {
                        Some(vec![kernel_size.clone(); *spatial_rank])
                    };
                    let Some(kernels) = kernels.filter(|k| k.len() == *spatial_rank) else {
                        return Ok(None);
                    };
                    if without_literal_mark(padding).parse::<i64>().is_err() {
                        return Ok(None);
                    }
                    let spatial = &mut output_shape[last - spatial_rank..last];
                    for (dim, kernel) in spatial.iter_mut().zip(&kernels) {
                        *dim = conv_spatial_dim(dim, kernel, "1", padding);
                        // flax gives an empty axis where torch fails.
                        if *dim == UNPROVEN_DIM {
                            *dim = "0".to_string();
                        }
                    }
                    Ok(Some(output_shape))
                }
            }
        }
        // Channels-first pooling: channels preserved, trailing spatial dims
        // follow the conv output formula.
        LayerKind::Pool {
            name,
            spatial_rank,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        } => {
            check_min_rank(name, layer, input, input_shape.len(), spatial_rank + 1)?;
            if string_text(padding).is_some() || padding.starts_with(PADDING_NAME) {
                return Ok(None);
            }
            let Some(dilations) = conv_dilations(dilation, *spatial_rank) else {
                return Ok(None);
            };
            let mut output_shape = input_shape.to_vec();
            let start = input_shape.len() - spatial_rank;
            for (dim, dilation) in output_shape[start..].iter_mut().zip(&dilations) {
                let Some(kernel) = effective_kernel_size(without_literal_mark(kernel_size), dilation) else {
                    return Ok(None);
                };
                let kernel = if dilation == "1" { kernel_size } else { &kernel };
                *dim = match ceil_mode.as_str() {
                    "False" => conv_spatial_dim(dim, kernel, stride, padding),
                    "True" => pool_ceil_spatial_dim(dim, kernel, stride, padding),
                    _ => UNKNOWN_DIM.to_string(),
                };
            }
            Ok(Some(output_shape))
        }
        LayerKind::AdaptivePool {
            name,
            spatial_rank,
            output_size,
        } => {
            check_min_rank(name, layer, input, input_shape.len(), spatial_rank + 1)?;
            let mut output_shape = input_shape.to_vec();
            let start = input_shape.len() - spatial_rank;
            for dim in &mut output_shape[start..] {
                dim.clone_from(output_size);
            }
            Ok(Some(output_shape))
        }
        // Index lookup: appends the embedding dim to the (integer) input shape.
        LayerKind::Embedding { embedding_size } => {
            let mut output_shape = input_shape.to_vec();
            output_shape.push(embedding_size.clone());
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
            check_min_rank("Upsample", layer, input, input_shape.len(), 3)?;
            let spatial_count = input_shape.len() - 2;
            let mut output_shape = input_shape.to_vec();
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
        // torch.nn.RNN/LSTM/GRU and LSTMCell return a tuple: the input is
        // checked, but one name gets no shape. Tuple unpacking gives the parts.
        LayerKind::Rnn { name, input_size, .. } => {
            check_min_rank(name, layer, input, input_shape.len(), 2)?;
            if input_shape.len() > 3 {
                return Err(format!(
                    "{} layer '{}' expects input with 2 or 3 dims, got {} for '{}'",
                    name,
                    layer,
                    input_shape.len(),
                    input
                ));
            }
            let output = rnn_outputs(kind, input_shape).output;
            replace_last_dim(name, layer, input, input_shape, Some(input_size), &output).map(|_| None)
        }
        LayerKind::RnnCell {
            name,
            input_size,
            hidden_size,
        } => {
            check_min_rank(name, layer, input, input_shape.len(), 1)?;
            let output = replace_last_dim(name, layer, input, input_shape, Some(input_size), hidden_size)?;
            Ok(output.filter(|_| name != "LSTMCell"))
        }
        // torch.nn.PixelShuffle(r): (*, C*r^2, H, W) -> (*, C, H*r, W*r).
        LayerKind::PixelShuffle { upscale_factor } => {
            check_min_rank("PixelShuffle", layer, input, input_shape.len(), 3)?;
            let Ok(r) = upscale_factor.trim().parse::<u64>() else {
                return Ok(None);
            };
            let mut output_shape = input_shape.to_vec();
            let len = output_shape.len();
            let channels_idx = len - 3;
            output_shape[channels_idx] = div_dim(&input_shape[channels_idx], r.checked_mul(r).unwrap_or(0))
                .map_err(|e| format!("PixelShuffle layer '{}': {}", layer, e))?;
            output_shape[len - 2] = mul_dim(&input_shape[len - 2], r);
            output_shape[len - 1] = mul_dim(&input_shape[len - 1], r);
            Ok(Some(output_shape))
        }
        // torch.nn.PixelUnshuffle(r): (*, C, H*r, W*r) -> (*, C*r^2, H, W).
        LayerKind::PixelUnshuffle { downscale_factor } => {
            check_min_rank("PixelUnshuffle", layer, input, input_shape.len(), 3)?;
            let Ok(r) = downscale_factor.trim().parse::<u64>() else {
                return Ok(None);
            };
            let mut output_shape = input_shape.to_vec();
            let len = output_shape.len();
            let channels_idx = len - 3;
            output_shape[channels_idx] = r.checked_mul(r).map_or(UNKNOWN_DIM.to_string(), |square| {
                mul_dim(&input_shape[channels_idx], square)
            });
            output_shape[len - 2] = div_dim(&input_shape[len - 2], r)
                .map_err(|e| format!("PixelUnshuffle layer '{}': {}", layer, e))?;
            output_shape[len - 1] = div_dim(&input_shape[len - 1], r)
                .map_err(|e| format!("PixelUnshuffle layer '{}': {}", layer, e))?;
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
            check_min_rank(name, layer, input, input_shape.len(), *spatial_rank)?;
            let mut output_shape = input_shape.to_vec();
            let start = input_shape.len() - spatial_rank;
            if let Ok(p) = padding.trim().parse::<isize>() {
                for dim in &mut output_shape[start..] {
                    *dim = p.checked_mul(2).map_or(UNKNOWN_DIM.to_string(), |pad| apply_offset(dim, pad));
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
                    *dim = left.checked_add(right).map_or(UNKNOWN_DIM.to_string(), |pad| apply_offset(dim, pad));
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
        } => replace_last_dim("Bilinear", layer, input, input_shape, Some(in1_features), out_features),
        // torch.nn.CosineSimilarity(dim): drops the reduced axis from x1's
        // shape (same single-tracked-input limitation as Bilinear).
        LayerKind::CosineSimilarity { dim } => {
            let rank = input_shape.len();
            let Some(axis) = resolve_axis(dim, rank) else {
                return Ok(None);
            };
            let mut output_shape = input_shape.to_vec();
            output_shape.remove(axis);
            Ok(Some(output_shape))
        }
        // equinox.nn.MLP: last-dim transform, same rule as Linear.
        LayerKind::Mlp { in_size, out_size } => {
            replace_last_dim("MLP", layer, input, input_shape, Some(in_size), out_size)
        }
        // torch.nn.Unfold: channels-first (C, H, W) / (N, C, H, W) ->
        // (C*kh*kw, L) / (N, C*kh*kw, L), L = num_blocks_h * num_blocks_w.
        LayerKind::Unfold {
            kernel_size,
            stride,
            padding,
            dilation,
        } => {
            check_min_rank("Unfold", layer, input, input_shape.len(), 3)?;
            let rank = input_shape.len();
            let channels_idx = rank - 3;
            let (Some((kh, kw)), Some((sh, sw)), Some((ph, pw)), Some((dh, dw))) = (
                parse_pair_arg(kernel_size),
                parse_pair_arg(stride),
                parse_pair_arg(padding),
                parse_pair_arg(dilation),
            ) else {
                return Ok(None);
            };
            let (Some(eff_kh), Some(eff_kw)) =
                (effective_kernel_size(&kh, &dh), effective_kernel_size(&kw, &dw))
            else {
                return Ok(None);
            };

            let blocks_h = conv_spatial_dim(&input_shape[channels_idx + 1], &eff_kh, &sh, &ph);
            let blocks_w = conv_spatial_dim(&input_shape[channels_idx + 2], &eff_kw, &sw, &pw);
            let l = product_dims(&[blocks_h, blocks_w]);
            let out_channels = product_dims(&[input_shape[channels_idx].clone(), kh, kw]);

            let mut output_shape = input_shape[..channels_idx].to_vec();
            output_shape.push(out_channels);
            output_shape.push(l);
            Ok(Some(output_shape))
        }
        // torch.nn.Fold: inverse of Unfold — (C*kh*kw, L) / (N, C*kh*kw, L)
        // -> (C, oh, ow) / (N, C, oh, ow). `L` isn't cross-checked.
        LayerKind::Fold {
            output_size,
            kernel_size,
        } => {
            check_min_rank("Fold", layer, input, input_shape.len(), 2)?;
            let rank = input_shape.len();
            let channels_idx = rank - 2;
            let (Some((oh, ow)), Some((kh, kw))) =
                (parse_pair_arg(output_size), parse_pair_arg(kernel_size))
            else {
                return Ok(None);
            };

            let channels_out = div_dim_by_dims(&input_shape[channels_idx], &[kh, kw])
                .map_err(|e| format!("Fold layer '{}': {}", layer, e))?;

            let mut output_shape = input_shape[..channels_idx].to_vec();
            output_shape.push(channels_out);
            output_shape.push(oh);
            output_shape.push(ow);
            Ok(Some(output_shape))
        }
        // Shape-preserving layers: output shape equals input shape, but some
        // layers have minimum-rank expectations (e.g. BatchNorm2d needs 3D (C, H, W)).
        LayerKind::ShapePreserving { name } => {
            if let Some(min_rank) = min_rank_for_shape_preserving(name) {
                check_min_rank(name, layer, input, input_shape.len(), min_rank)?;
            }
            Ok(Some(input_shape.to_vec()))
        }
        // torch.nn.Sequential / equinox.nn.Sequential: thread the shape
        // through each child in order via this same function. An unknown
        // (`Ok(None)`) child makes the whole chain unknown; an `Err` from a
        // child propagates as-is, so the reported message is that child's
        // own (`layer[index]`-qualified for context).
        LayerKind::Sequential { children } => {
            let mut current = input_shape.to_vec();
            for (i, child) in children.iter().enumerate() {
                let child_layer = format!("{layer}[{i}]");
                match apply_layer_rule(child, &current, &child_layer, input)?.and_then(bounded_shape) {
                    Some(shape) => current = shape,
                    None => return Ok(None),
                }
            }
            Ok(Some(current))
        }
        // einops.layers.{torch,flax}.Rearrange/Reduce: delegate straight to
        // the free-function pattern engine (`apply_known_einops`) via a
        // single-entry `ShapeLookup` that answers only for `input`'s shape.
        LayerKind::EinopsPattern { pattern, kwargs, .. } => {
            let mut args = vec![
                CallArgument::Positional {
                    value: input.to_string(),
                },
                CallArgument::Positional {
                    value: pattern.clone(),
                },
            ];
            args.extend(kwargs.iter().map(|(name, value)| CallArgument::Keyword {
                name: name.clone(),
                value: value.clone(),
            }));
            let lookup = SingleShapeLookup {
                name: input.to_string(),
                shape: input_shape.to_vec(),
            };
            apply_known_einops(&args, &lookup)
        }
        // A loss needs its target too: `apply_layer_application` runs it.
        LayerKind::Loss { .. } => Ok(None),
    }
}

/// `ShapeLookup` over exactly one (name, shape) pair — lets
/// `known_functions::apply_known_einops` (built for free-function calls
/// where the input tensor is itself a named `CallArgument`) be reused for a
/// `LayerKind::EinopsPattern` layer application, where the input shape is
/// already known but has no scope-wide variable name.
struct SingleShapeLookup {
    name: String,
    shape: Vec<String>,
}

impl ShapeLookup for SingleShapeLookup {
    fn shape(&self, name: &str) -> Option<&Vec<String>> {
        (name == self.name).then_some(&self.shape)
    }
}

#[cfg(test)]
mod table_tests {
    use super::*;
    use std::fs;
    use tree_sitter::Parser;

    fn s(text: &str) -> String {
        text.to_string()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn app(input: &str, kind: LayerKind) -> LayerApplication {
        LayerApplication {
            variable: s("y"),
            layer: s("layer"),
            input: s(input),
            kind,
            range: Range {
                start_byte: 0,
                end_byte: 0,
                start_point: tree_sitter::Point::new(0, 0),
                end_point: tree_sitter::Point::new(0, 0),
            },
            extra_inputs: Vec::new(),
        }
    }

    fn linear(in_features: &str, out_features: &str) -> LayerKind {
        LayerKind::Linear { in_features: s(in_features), out_features: s(out_features) }
    }

    fn kept(name: &str) -> LayerKind {
        LayerKind::ShapePreserving { name: s(name) }
    }

    fn seq(children: Vec<LayerKind>) -> LayerKind {
        LayerKind::Sequential { children }
    }

    fn conv(spatial_rank: usize, in_channels: &str, out_channels: &str, kernel_size: &str, stride: &str, padding: &str) -> LayerKind {
        LayerKind::Conv {
            spatial_rank,
            in_channels: s(in_channels),
            out_channels: s(out_channels),
            kernel_size: s(kernel_size),
            stride: s(stride),
            padding: s(padding),
            dilation: s("1"),
        }
    }

    fn conv_transpose(spatial_rank: usize, in_channels: &str, out_channels: &str, kernel_size: &str, stride: &str, padding: &str) -> LayerKind {
        conv_transpose_with(spatial_rank, in_channels, out_channels, kernel_size, stride, padding, "0", "1")
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_transpose_with(
        spatial_rank: usize,
        in_channels: &str,
        out_channels: &str,
        kernel_size: &str,
        stride: &str,
        padding: &str,
        output_padding: &str,
        dilation: &str,
    ) -> LayerKind {
        LayerKind::ConvTranspose {
            spatial_rank,
            in_channels: s(in_channels),
            out_channels: s(out_channels),
            kernel_size: s(kernel_size),
            stride: s(stride),
            padding: s(padding),
            output_padding: s(output_padding),
            dilation: s(dilation),
        }
    }

    fn pool(name: &str, spatial_rank: usize, kernel_size: &str, stride: &str, padding: &str) -> LayerKind {
        pool_with(name, spatial_rank, kernel_size, stride, padding, "1", "False")
    }

    fn pool_with(name: &str, spatial_rank: usize, kernel_size: &str, stride: &str, padding: &str, dilation: &str, ceil_mode: &str) -> LayerKind {
        LayerKind::Pool {
            name: s(name),
            spatial_rank,
            kernel_size: s(kernel_size),
            stride: s(stride),
            padding: s(padding),
            dilation: s(dilation),
            ceil_mode: s(ceil_mode),
        }
    }

    fn einops(name: &str, pattern: &str, kwargs: &[(&str, &str)]) -> LayerKind {
        let kwargs = kwargs.iter().map(|(k, v)| (s(k), s(v))).collect();
        LayerKind::EinopsPattern { name: s(name), pattern: s(pattern), kwargs }
    }

    fn unfold(kernel_size: &str, stride: &str, padding: &str, dilation: &str) -> LayerKind {
        LayerKind::Unfold { kernel_size: s(kernel_size), stride: s(stride), padding: s(padding), dilation: s(dilation) }
    }

    fn fold(output_size: &str, kernel_size: &str) -> LayerKind {
        LayerKind::Fold { output_size: s(output_size), kernel_size: s(kernel_size) }
    }

    fn mha(feature_dim: &str) -> LayerKind {
        mha_with(feature_dim, "8", "False", "0")
    }

    fn mha_with(feature_dim: &str, num_heads: &str, batch_first: &str, added_keys: &str) -> LayerKind {
        LayerKind::MultiheadAttention {
            feature_dim: s(feature_dim),
            num_heads: s(num_heads),
            batch_first: s(batch_first),
            added_keys: s(added_keys),
            returns_weights: true,
            output_size: s(feature_dim),
        }
    }

    fn eqx_mha(feature_dim: &str, num_heads: &str) -> LayerKind {
        eqx_mha_with(feature_dim, num_heads, feature_dim)
    }

    fn eqx_mha_with(feature_dim: &str, num_heads: &str, output_size: &str) -> LayerKind {
        match mha_with(feature_dim, num_heads, "False", "0") {
            LayerKind::MultiheadAttention { feature_dim, num_heads, batch_first, added_keys, .. } => {
                LayerKind::MultiheadAttention {
                    feature_dim,
                    num_heads,
                    batch_first,
                    added_keys,
                    returns_weights: false,
                    output_size: s(output_size),
                }
            }
            _ => unreachable!(),
        }
    }

    fn embedding(embedding_size: &str) -> LayerKind {
        LayerKind::Embedding { embedding_size: s(embedding_size) }
    }

    fn rnn(name: &str, input_size: &str, hidden_size: &str) -> LayerKind {
        rnn_with(name, input_size, hidden_size, ["1", "False", "0", "False"])
    }

    /// `options`: num_layers, bidirectional, proj_size, batch_first.
    fn rnn_with(name: &str, input_size: &str, hidden_size: &str, options: [&str; 4]) -> LayerKind {
        let [num_layers, bidirectional, proj_size, batch_first] = options.map(s);
        LayerKind::Rnn {
            name: s(name),
            input_size: s(input_size),
            hidden_size: s(hidden_size),
            num_layers,
            bidirectional,
            proj_size,
            batch_first,
        }
    }

    fn pad(padding: &str) -> LayerKind {
        LayerKind::Pad { name: s("ConstantPad2d"), spatial_rank: 2, padding: s(padding) }
    }

    fn bilinear(in1_features: &str) -> LayerKind {
        LayerKind::Bilinear { in1_features: s(in1_features), in2_features: s("20"), out_features: s("30") }
    }

    fn mlp(in_size: &str, out_size: &str) -> LayerKind {
        LayerKind::Mlp { in_size: s(in_size), out_size: s(out_size) }
    }

    enum Applied {
        Gives(&'static [&'static str]),
        Unknown,
        Fails(&'static [&'static str]),
    }
    use Applied::*;

    #[test]
    fn apply_cases() {
        let flatten = |start_dim: &str, end_dim: &str| LayerKind::Flatten { start_dim: s(start_dim), end_dim: s(end_dim) };
        let unflatten = |dim: &str| LayerKind::Unflatten { dim: s(dim), sizes: s("(2, 3)") };
        let upsample = |scale_factor: Option<&str>, size: Option<&str>| LayerKind::Upsample {
            scale_factor: scale_factor.map(s),
            size: size.map(s),
        };
        let cosine = |dim: &str| LayerKind::CosineSimilarity { dim: s(dim) };
        let rows: &[(&str, LayerKind, &[&str], Applied)] = &[
            ("linear rank 2", linear("3", "5"), &["batch", "3"], Gives(&["batch", "5"])),
            ("linear rank 1", linear("3", "5"), &["3"], Gives(&["5"])),
            ("linear symbolic", linear("features", "hidden"), &["batch", "features"], Gives(&["batch", "hidden"])),
            ("linear keeps leading dims", linear("features", "hidden"), &["time", "batch", "features"], Gives(&["time", "batch", "hidden"])),
            ("linear scalar input", linear("3", "5"), &[], Fails(&["Cannot apply Linear layer", "scalar input"])),
            ("linear last dim mismatch", linear("3", "5"), &["batch", "4"], Fails(&["expected input last dim 3", "got 4"])),
            // Issue #47: ctor-argument text and annotation dims are different
            // vocabularies, so a symbolic mismatch is unprovable and trusted.
            ("linear symbolic mismatch", linear("features", "hidden"), &["batch", "other"], Gives(&["batch", "hidden"])),
            ("linear numeric vs symbolic", linear("features", "hidden"), &["batch", "3"], Gives(&["batch", "hidden"])),
            ("linear same in and out", linear("features", "features"), &["batch", "features"], Gives(&["batch", "features"])),
            ("embedding appends dim", embedding("512"), &["batch", "seq"], Gives(&["batch", "seq", "512"])),
            ("embedding scalar index", embedding("512"), &[], Gives(&["512"])),
            ("maxpool2d halves spatial dims", pool("MaxPool2d", 2, "2", "2", "0"), &["16", "32", "32"], Gives(&["16", "16", "16"])),
            ("maxpool1d symbolic", pool("MaxPool1d", 1, "2", "2", "0"), &["c", "L"], Gives(&["c", "(L-2)/2+1"])),
            (
                "adaptive pool sets spatial dims",
                LayerKind::AdaptivePool { name: s("AdaptiveAvgPool2d"), spatial_rank: 2, output_size: s("1") },
                &["32", "16", "16"],
                Gives(&["32", "1", "1"]),
            ),
            ("maxpool ceil mode", pool_with("MaxPool2d", 2, "3", "3", "0", "1", "True"), &["3", "16", "16"], Gives(&["3", "6", "6"])),
            // The last window would start in the right padding, so torch drops it.
            ("pool ceil mode drops last window", pool_with("AvgPool1d", 1, "2", "2", "1", "1", "True"), &["3", "5"], Gives(&["3", "3"])),
            ("pool ceil mode symbolic", pool_with("MaxPool1d", 1, "2", "2", "0", "1", "True"), &["c", "L"], Gives(&["c", UNKNOWN_DIM])),
            ("pool unproven ceil mode", pool_with("MaxPool1d", 1, "2", "2", "0", "1", "flag"), &["3", "8"], Gives(&["3", UNKNOWN_DIM])),
            // Effective kernel 2*(3-1)+1 = 5.
            ("maxpool dilation", pool_with("MaxPool1d", 1, "3", "1", "0", "2", "False"), &["3", "9"], Gives(&["3", "5"])),
            ("pool rank too low", pool("MaxPool2d", 2, "2", "2", "0"), &["16"], Fails(&["requires input with at least 3 dims"])),
            ("flatten middle dims", flatten("1", "2"), &["2", "3", "4", "5"], Gives(&["2", "12", "5"])),
            ("flatten symbolic, negative end", flatten("1", "-1"), &["batch", "c", "d"], Gives(&["batch", "c*d"])),
            ("flatten unresolvable start", flatten("axis", "-1"), &["2", "3", "4"], Unknown),
            ("unflatten", unflatten("1"), &["batch", "6", "feat"], Gives(&["batch", "2", "3", "feat"])),
            ("unflatten non-literal dim", unflatten("axis"), &["batch", "6", "feat"], Unknown),
            ("upsample scalar scale", upsample(Some("2"), None), &["1", "3", "8", "8"], Gives(&["1", "3", "16", "16"])),
            ("upsample size tuple", upsample(None, Some("(20, 20)")), &["1", "3", "8", "8"], Gives(&["1", "3", "20", "20"])),
            ("upsample rank too low", upsample(Some("2"), None), &["3", "8"], Fails(&["requires input with at least 3 dims"])),
            // conv_spatial_dim(8, k=4, s=2, p=1) == 4; the transpose inverts it.
            ("conv transpose2d", conv_transpose(2, "3", "16", "4", "2", "1"), &["3", "4", "4"], Gives(&["16", "8", "8"])),
            (
                "conv transpose2d output padding",
                conv_transpose_with(2, "3", "3", "3", "2", "1", "1", "1"),
                &["1", "3", "16", "16"],
                Gives(&["1", "3", "32", "32"]),
            ),
            // (5-1)*1 - 0 + 2*(3-1) + 1 = 9.
            (
                "conv transpose1d dilation",
                conv_transpose_with(1, "3", "3", "3", "1", "0", "0", "2"),
                &["3", "5"],
                Gives(&["3", "9"]),
            ),
            (
                "conv transpose symbolic output padding",
                conv_transpose_with(1, "3", "3", "3", "2", "1", "1", "1"),
                &["3", "L"],
                Gives(&["3", "(L-1)*2-2+3+1"]),
            ),
            // torch refuses an output_padding that is not below the stride.
            (
                "conv transpose output padding too large",
                conv_transpose_with(1, "3", "3", "3", "2", "1", "2", "1"),
                &["3", "5"],
                Unknown,
            ),
            ("conv transpose2d channels", conv_transpose(2, "3", "16", "4", "2", "1"), &["5", "4", "4"], Fails(&["expected 3 input channels"])),
            // `(output, h_n)` is a tuple: one name gets no shape.
            ("rnn output is a tuple", rnn("GRU", "10", "20"), &["seq", "10"], Unknown),
            ("lstm output is a tuple", rnn_with("LSTM", "8", "16", ["1", "True", "0", "False"]), &["5", "3", "8"], Unknown),
            ("rnn rank 4", rnn("GRU", "10", "20"), &["2", "5", "3", "10"], Fails(&["expects input with 2 or 3 dims"])),
            ("rnn input size mismatch", rnn("GRU", "10", "20"), &["seq", "9"], Fails(&["expected input last dim 10"])),
            (
                "rnn cell unbatched",
                LayerKind::RnnCell { name: s("GRUCell"), input_size: s("10"), hidden_size: s("20") },
                &["10"],
                Gives(&["20"]),
            ),
            (
                "lstm cell output is a tuple",
                LayerKind::RnnCell { name: s("LSTMCell"), input_size: s("10"), hidden_size: s("20") },
                &["10"],
                Unknown,
            ),
            (
                "lstm cell input size mismatch",
                LayerKind::RnnCell { name: s("LSTMCell"), input_size: s("10"), hidden_size: s("20") },
                &["9"],
                Fails(&["expected input last dim 10"]),
            ),
            ("pixel shuffle", LayerKind::PixelShuffle { upscale_factor: s("2") }, &["8", "4", "4"], Gives(&["2", "8", "8"])),
            ("pixel shuffle indivisible", LayerKind::PixelShuffle { upscale_factor: s("2") }, &["9", "4", "4"], Fails(&["not evenly divisible"])),
            ("pixel unshuffle", LayerKind::PixelUnshuffle { downscale_factor: s("2") }, &["2", "8", "8"], Gives(&["8", "4", "4"])),
            ("pad uniform int", pad("1"), &["3", "8", "8"], Gives(&["3", "10", "10"])),
            // The first (left, right) pair pads the last spatial dim.
            ("pad tuple reverse axes", pad("(1, 1, 2, 2)"), &["3", "8", "8"], Gives(&["3", "12", "10"])),
            ("pad symbolic", pad("p"), &["3", "8", "8"], Unknown),
            ("bilinear first input", bilinear("10"), &["batch", "10"], Gives(&["batch", "30"])),
            ("bilinear in1 mismatch", bilinear("10"), &["batch", "9"], Fails(&["expected input last dim 10"])),
            ("cosine similarity", cosine("1"), &["batch", "10"], Gives(&["batch"])),
            ("cosine similarity non-literal dim", cosine("axis"), &["batch", "10"], Unknown),
            ("mlp", mlp("10", "5"), &["10"], Gives(&["5"])),
            ("mlp in size mismatch", mlp("10", "5"), &["9"], Fails(&["expected input last dim 10"])),
            ("instance norm2d min rank", kept("InstanceNorm2d"), &["8"], Fails(&["requires input with at least 3 dims"])),
            ("identity any rank", kept("Identity"), &[], Gives(&[])),
            ("local response norm min rank", kept("LocalResponseNorm"), &["4", "8"], Fails(&["requires input with at least 3 dims"])),
            ("silu any rank", kept("SiLU"), &["4"], Gives(&["4"])),
            ("transformer encoder stack", kept("TransformerEncoder"), &["seq", "batch", "512"], Gives(&["seq", "batch", "512"])),
            ("alpha dropout any rank", kept("AlphaDropout"), &[], Gives(&[])),
            ("transformer encoder any rank", kept("TransformerEncoderLayer"), &[], Gives(&[])),
            ("transformer decoder any rank", kept("TransformerDecoderLayer"), &[], Gives(&[])),
            ("rms norm min rank", kept("RMSNorm"), &[], Fails(&["requires input with at least 1 dims"])),
            (
                "sequential linear relu linear",
                seq(vec![linear("784", "256"), kept("ReLU"), linear("256", "10")]),
                &["batch", "784"],
                Gives(&["batch", "10"]),
            ),
            (
                "sequential reports the failing child",
                seq(vec![linear("784", "256"), linear("999", "10")]),
                &["batch", "784"],
                Fails(&["layer[1]", "expected input last dim 999", "got 256"]),
            ),
            ("sequential scalar input", seq(vec![linear("784", "256")]), &[], Fails(&["layer[0]"])),
            // MultiheadAttention applied directly is unknown, so the chain is too.
            ("sequential unknown child", seq(vec![mha("512"), linear("512", "10")]), &["batch", "seq", "512"], Unknown),
            (
                "nested sequential",
                seq(vec![seq(vec![linear("4", "8"), kept("ReLU")]), linear("8", "2")]),
                &["batch", "4"],
                Gives(&["batch", "2"]),
            ),
            ("empty sequential", seq(Vec::new()), &["batch", "4"], Gives(&["batch", "4"])),
            ("einops rearrange", einops("Rearrange", "\"b c h w -> b (c h w)\"", &[]), &["2", "3", "4", "5"], Gives(&["2", "60"])),
            (
                "einops reduce axis kwargs",
                einops("Reduce", "\"b (h w) c -> b h w c\"", &[("h", "14")]),
                &["2", "196", "8"],
                Gives(&["2", "14", "14", "8"]),
            ),
            ("einops rank mismatch", einops("Rearrange", "\"b c h w -> b (c h w)\"", &[]), &["2", "3", "4"], Fails(&["expects rank 4"])),
            (
                "sequential rearrange then linear",
                seq(vec![einops("Rearrange", "\"b c h w -> b (c h w)\"", &[]), linear("60", "10")]),
                &["2", "3", "4", "5"],
                Gives(&["2", "10"]),
            ),
            // 6 blocks per axis: L = 36, channels = 3*3*3 = 27.
            ("unfold scalar args", unfold("3", "1", "0", "1"), &["3", "8", "8"], Gives(&["27", "36"])),
            ("unfold batched", unfold("3", "1", "0", "1"), &["2", "3", "8", "8"], Gives(&["2", "27", "36"])),
            // blocks_h = 5, blocks_w = 4, L = 20, channels = 3*2*3 = 18.
            ("unfold per-axis tuples", unfold("(2, 3)", "(2, 3)", "0", "1"), &["3", "10", "12"], Gives(&["18", "20"])),
            // Effective kernel 5 gives 3 blocks per axis. Channels use the raw kernel.
            ("unfold dilation", unfold("3", "1", "0", "2"), &["3", "7", "7"], Gives(&["27", "9"])),
            ("unfold symbolic spatial dims", unfold("3", "1", "0", "1"), &["c", "H", "W"], Gives(&["c*3*3", "(H-2)*(W-2)"])),
            ("unfold dilation with symbolic kernel", unfold("k", "1", "0", "2"), &["3", "8", "8"], Unknown),
            ("unfold rank too low", unfold("2", "1", "0", "1"), &["8"], Fails(&["requires input with at least 3 dims"])),
            ("unfold bad tuple arity", unfold("(1, 2, 3)", "1", "0", "1"), &["3", "8", "8"], Unknown),
            ("fold divisible channels", fold("(6, 6)", "2"), &["12", "20"], Gives(&["3", "6", "6"])),
            ("fold batched", fold("(6, 6)", "2"), &["8", "12", "20"], Gives(&["8", "3", "6", "6"])),
            ("fold per-axis kernel", fold("(5, 7)", "(2, 3)"), &["18", "35"], Gives(&["3", "5", "7"])),
            ("fold not divisible", fold("(4, 4)", "3"), &["10", "20"], Fails(&["not evenly divisible by 9"])),
            ("fold symbolic", fold("(H, W)", "k"), &["c", "L"], Gives(&["c/(k*k)", "H", "W"])),
            ("fold rank too low", fold("(4, 4)", "2"), &[], Fails(&["requires input with at least 2 dims"])),
            ("fold bad output size arity", fold("(4, 5, 6)", "2"), &["12", "20"], Unknown),
        ];
        for (case, kind, input, expected) in rows {
            let shapes = HashMap::from([(s("x"), shape(input))]);
            let result = apply_layer_application(&app("x", kind.clone()), &shapes);
            match expected {
                Gives(dims) => assert_eq!(result, Ok(Some(shape(dims))), "case: {case}"),
                Unknown => assert_eq!(result, Ok(None), "case: {case}"),
                Fails(parts) => {
                    let error = result.expect_err(case);
                    for part in *parts {
                        assert!(error.contains(part), "case: {case}: error was: {error}");
                    }
                }
            }
        }
    }

    /// Case, options (num_layers, bidirectional, proj_size, batch_first), input,
    /// output last dim, h_n, c_n.
    type RnnRow<'a> = (&'a str, [&'a str; 4], &'a [&'a str], &'a str, Option<&'a [&'a str]>, Option<&'a [&'a str]>);

    #[test]
    fn rnn_output_cases() {
        let u = UNKNOWN_DIM;
        let rows: &[RnnRow] = &[
            ("default batched", ["1", "False", "0", "False"], &["5", "3", "8"], "16", Some(&["1", "3", "16"]), Some(&["1", "3", "16"])),
            ("unbatched", ["1", "False", "0", "False"], &["5", "8"], "16", Some(&["1", "16"]), Some(&["1", "16"])),
            ("bidirectional layers", ["2", "True", "0", "False"], &["5", "3", "8"], "32", Some(&["4", "3", "16"]), Some(&["4", "3", "16"])),
            ("batch first keeps batch at axis 1", ["1", "False", "0", "True"], &["3", "5", "8"], "16", Some(&["1", "3", "16"]), Some(&["1", "3", "16"])),
            ("projection", ["1", "True", "4", "False"], &["5", "3", "8"], "8", Some(&["2", "3", "4"]), Some(&["2", "3", "16"])),
            ("unproven num_layers", ["n", "False", "0", "False"], &["5", "3", "8"], "16", Some(&[u, "3", "16"]), Some(&[u, "3", "16"])),
            ("unproven batch_first", ["1", "False", "0", "bf"], &["5", "3", "8"], "16", Some(&["1", u, "16"]), Some(&["1", u, "16"])),
            ("unproven proj_size", ["1", "False", "p", "False"], &["5", "8"], u, Some(&["1", u]), Some(&["1", "16"])),
            ("rank 4", ["1", "False", "0", "False"], &["2", "5", "3", "8"], "16", None, None),
        ];
        for (case, options, input, output, hidden, cell) in rows {
            let outputs = rnn_outputs(&rnn_with("LSTM", "8", "16", *options), &shape(input));
            assert_eq!(outputs.output, *output, "case: {case}");
            assert_eq!(outputs.hidden, hidden.map(shape), "case: {case}");
            assert_eq!(outputs.cell, cell.map(shape), "case: {case}");
        }
    }

    /// Case, layer, query, key, (need_weights, average_attn_weights), weights.
    type AttentionRow<'a> = (&'a str, LayerKind, &'a [&'a str], &'a [&'a str], [&'a str; 2], Option<&'a [&'a str]>);

    #[test]
    fn attention_weight_cases() {
        let rows: &[AttentionRow] = &[
            ("sequence first", mha("16"), &["7", "2", "16"], &["9", "2", "16"], ["True", "True"], Some(&["2", "7", "9"])),
            ("batch first", mha_with("16", "4", "True", "0"), &["2", "7", "16"], &["2", "9", "16"], ["True", "True"], Some(&["2", "7", "9"])),
            ("unbatched", mha("16"), &["7", "16"], &["9", "16"], ["True", "True"], Some(&["7", "9"])),
            ("per head", mha_with("16", "4", "False", "0"), &["7", "2", "16"], &["9", "2", "16"], ["True", "False"], Some(&["2", "4", "7", "9"])),
            ("need_weights false", mha("16"), &["7", "2", "16"], &["9", "2", "16"], ["False", "True"], None),
            ("unproven need_weights", mha("16"), &["7", "2", "16"], &["9", "2", "16"], ["flag", "True"], None),
            ("unproven average", mha("16"), &["7", "2", "16"], &["9", "2", "16"], ["True", "avg"], None),
            ("unproven batch_first", mha_with("16", "4", "bf", "0"), &["7", "2", "16"], &["9", "2", "16"], ["True", "True"], None),
            ("added keys", mha_with("16", "4", "False", "2"), &["7", "2", "16"], &["9", "2", "16"], ["True", "True"], Some(&["2", "7", "11"])),
            ("unproven added keys", mha_with("16", "4", "False", UNPROVEN_DIM), &["7", "2", "16"], &["9", "2", "16"], ["True", "True"], None),
            ("equinox", eqx_mha("16", "4"), &["7", "16"], &["9", "16"], ["True", "True"], None),
        ];
        for (case, kind, query, key, [need, average], expected) in rows {
            let weights = attention_weights(kind, &shape(query), &shape(key), need, average);
            assert_eq!(weights, expected.map(shape), "case: {case}");
        }
    }

    #[test]
    fn apply_without_input_shape_is_unknown() {
        let output = apply_layer_application(&app("x", linear("3", "5")), &HashMap::new()).unwrap();
        assert_eq!(output, None);
    }

    #[test]
    fn apply_error_mentions_layer_and_input_names() {
        let mut app = app("input", linear("3", "5"));
        app.layer = s("projection");
        let shapes = HashMap::from([(s("input"), shape(&["batch", "4"]))]);

        let error = apply_layer_application(&app, &shapes).unwrap_err();

        assert!(error.contains("projection"));
        assert!(error.contains("input"));
    }

    fn call(module_parts: &[&str], owner: Option<&str>, name: &str, bindings: &[(&str, &str)]) -> ResolvedCallSignature {
        ResolvedCallSignature {
            implementation: ResolvedImplementation {
                target: ResolvedModuleTarget {
                    dots: 0,
                    module_parts: module_parts.iter().map(|p| p.to_string()).collect(),
                    file_path: PathBuf::from("unused.py"),
                    symbol_parts: vec![s(owner.unwrap_or(name))],
                },
                symbol: owner.map(|owner| PythonSymbol::Class { name: s(owner) }),
            },
            signature: PythonCallableSignature {
                owner: owner.map(s),
                name: s(name),
                params: Vec::new(),
            },
            arguments: Vec::new(),
            bindings: bindings.iter().map(|(name, value)| (s(name), s(value))).collect(),
        }
    }

    /// Case, module, owner, method, bindings, expected layer.
    type ClassifyRow<'a> = (&'a str, &'a [&'a str], Option<&'a str>, &'a str, &'a [(&'a str, &'a str)], Option<LayerKind>);

    #[test]
    fn classify_cases() {
        const EQX_LINEAR: &[&str] = &["equinox", "nn", "_linear"];
        const IN_OUT: &[(&str, &str)] = &[("in_features", "3"), ("out_features", "5")];
        const REARRANGE: &str = "\"b c h w -> b (c h w)\"";
        const REDUCE: &str = "\"b (h w) c -> b h w c\"";
        let init = "__init__";
        let rows: &[ClassifyRow] = &[
            ("equinox linear", EQX_LINEAR, Some("Linear"), init, IN_OUT, Some(linear("3", "5"))),
            (
                "symbolic equinox linear",
                EQX_LINEAR,
                Some("Linear"),
                init,
                &[("in_features", "features"), ("out_features", "hidden")],
                Some(linear("features", "hidden")),
            ),
            ("missing in_features", EQX_LINEAR, Some("Linear"), init, &[("out_features", "5")], None),
            ("missing out_features", EQX_LINEAR, Some("Linear"), init, &[("in_features", "3")], None),
            ("non-equinox linear", &["my_project", "layers"], Some("Linear"), init, IN_OUT, None),
            ("non-constructor", EQX_LINEAR, Some("Linear"), "forward", IN_OUT, None),
            ("nested equinox.nn module", &["equinox", "nn", "layers", "linear"], Some("Linear"), init, IN_OUT, Some(linear("3", "5"))),
            ("equinox not nn", &["equinox", "experimental"], Some("Linear"), init, IN_OUT, None),
            ("wrong owner", &["equinox", "nn"], Some("Dense"), init, IN_OUT, None),
            ("function call", &["jax", "numpy"], None, "concatenate", &[("arrays", "xs")], None),
            (
                "torch multihead attention reads embed_dim",
                &["torch", "nn"],
                Some("MultiheadAttention"),
                init,
                &[("embed_dim", "512"), ("num_heads", "8")],
                Some(mha("512")),
            ),
            // equinox orders (num_heads, query_size, ...), the reverse of torch.
            (
                "equinox multihead attention reads query_size",
                &["equinox", "nn"],
                Some("MultiheadAttention"),
                init,
                &[("num_heads", "8"), ("query_size", "512")],
                Some(eqx_mha("512", "8")),
            ),
            (
                "equinox multihead attention ignores embed_dim",
                &["equinox", "nn"],
                Some("MultiheadAttention"),
                init,
                &[("embed_dim", "8"), ("num_heads", "8"), ("query_size", "512")],
                Some(eqx_mha("512", "8")),
            ),
            (
                "einops torch rearrange",
                &["einops", "layers", "torch"],
                Some("Rearrange"),
                init,
                &[("pattern", REARRANGE)],
                Some(einops("Rearrange", REARRANGE, &[])),
            ),
            (
                "einops flax reduce with axis kwargs",
                &["einops", "layers", "flax"],
                Some("Reduce"),
                init,
                &[("pattern", REDUCE), ("reduction", "\"mean\""), ("h", "14"), ("w", "14")],
                Some(einops("Reduce", REDUCE, &[("h", "14"), ("w", "14")])),
            ),
            ("einops rearrange without pattern", &["einops", "layers", "torch"], Some("Rearrange"), init, &[], None),
            ("einops rearrange wrong module", &["einops", "layers", "chainer"], Some("Rearrange"), init, &[("pattern", "\"a b -> b a\"")], None),
            ("torch unfold scalar args", &["torch", "nn", "fold"], Some("Unfold"), init, &[("kernel_size", "3")], Some(unfold("3", "1", "0", "1"))),
            (
                "torch unfold tuple args",
                &["torch", "nn", "fold"],
                Some("Unfold"),
                init,
                &[("kernel_size", "(2, 3)"), ("stride", "(1, 2)"), ("padding", "1")],
                Some(unfold("(2, 3)", "(1, 2)", "1", "1")),
            ),
            ("unfold without kernel_size", &["torch", "nn", "fold"], Some("Unfold"), init, &[], None),
            (
                "torch fold",
                &["torch", "nn", "fold"],
                Some("Fold"),
                init,
                &[("output_size", "(4, 5)"), ("kernel_size", "2")],
                Some(fold("(4, 5)", "2")),
            ),
            ("fold without output_size", &["torch", "nn", "fold"], Some("Fold"), init, &[("kernel_size", "2")], None),
            (
                "local response norm",
                &["torch", "nn", "normalization"],
                Some("LocalResponseNorm"),
                init,
                &[("size", "5")],
                Some(kept("LocalResponseNorm")),
            ),
            // `Linear(self.d_inner, ...)` must match a `d_inner` annotation dim.
            (
                "strips self prefix",
                EQX_LINEAR,
                Some("Linear"),
                init,
                &[("in_features", "self.d_inner"), ("out_features", "10")],
                Some(linear("d_inner", "10")),
            ),
            (
                "strips self prefix in compound arg",
                EQX_LINEAR,
                Some("Linear"),
                init,
                &[("in_features", "self.d_inner*2"), ("out_features", "10")],
                Some(linear("d_inner*2", "10")),
            ),
        ];
        for (case, module_parts, owner, name, bindings, expected) in rows {
            let call = call(module_parts, *owner, name, bindings);
            assert_eq!(&classify_layer_call(&call), expected, "case: {case}");
        }
    }

    enum Sig {
        /// The params start with the first list and contain the second.
        Params(&'static [&'static str], &'static [&'static str]),
        Exact(&'static [&'static str]),
        Missing,
    }

    #[test]
    fn signature_cases() {
        const EQX: &[&str] = &["equinox", "nn"];
        const TORCH: &[&str] = &["torch", "nn"];
        const CONV_PREFIX: &[&str] = &["self", "in_channels", "out_channels", "kernel_size"];
        const CONV_T: &[&str] = &["ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"];
        const CONV_T_PREFIX: &[&str] = &["self", "in_channels", "out_channels", "kernel_size", "stride"];
        const RNN_PREFIX: &[&str] = &["self", "input_size", "hidden_size"];
        const KEPT: &[&str] = &[
            "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "BatchNorm", "BatchNorm1d", "BatchNorm2d",
            "BatchNorm3d", "LayerNorm", "GroupNorm", "ReLU", "GELU", "Sigmoid", "Tanh", "Softmax", "PReLU",
            "Identity", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "RMSNorm", "AlphaDropout",
            "TransformerEncoderLayer", "TransformerDecoderLayer",
        ];
        let rows: &[(&[&str], &[&str], Sig)] = &[
            (EQX, &["Linear"], Sig::Params(&["self", "in_features", "out_features"], &[])),
            (TORCH, &["Linear"], Sig::Params(&["self", "in_features", "out_features"], &[])),
            (EQX, &["Conv1d", "Conv2d", "Conv3d"], Sig::Params(CONV_PREFIX, &["stride", "padding"])),
            (TORCH, &["Conv1d", "Conv2d", "Conv3d"], Sig::Params(CONV_PREFIX, &[])),
            (EQX, KEPT, Sig::Params(&[], &[])),
            (TORCH, KEPT, Sig::Params(&[], &[])),
            (&["jax", "nn"], &["Linear"], Sig::Missing),
            (&["flax", "nn"], &["Dense"], Sig::Missing),
            (&["flax", "linen"], &["Dense"], Sig::Params(&[], &["features"])),
            (EQX, &["Mystery"], Sig::Missing),
            (TORCH, &["Mystery"], Sig::Missing),
            (&[], &["Linear"], Sig::Missing),
            (&["torch"], &["nn"], Sig::Missing),
            (&["equinox", "experimental"], &["Linear"], Sig::Missing),
            (&["torch", "functional"], &["Linear"], Sig::Missing),
            (TORCH, &["MultiheadAttention"], Sig::Params(&["self", "embed_dim", "num_heads"], &[])),
            // Reversed vs. torch: num_heads first, then query_size.
            (EQX, &["MultiheadAttention"], Sig::Params(&["self", "num_heads", "query_size"], &[])),
            (TORCH, &["Flatten"], Sig::Exact(&["self", "start_dim", "end_dim"])),
            (TORCH, &["Unflatten"], Sig::Exact(&["self", "dim", "sizes"])),
            (TORCH, &["Upsample"], Sig::Params(&[], &["scale_factor", "size"])),
            (EQX, CONV_T, Sig::Params(CONV_T_PREFIX, &["padding"])),
            (TORCH, CONV_T, Sig::Params(CONV_T_PREFIX, &["padding"])),
            (TORCH, &["RNN", "LSTM", "GRU", "RNNCell", "LSTMCell", "GRUCell"], Sig::Params(RNN_PREFIX, &[])),
            (EQX, &["LSTMCell", "GRUCell"], Sig::Params(RNN_PREFIX, &[])),
            (TORCH, &["PixelShuffle"], Sig::Exact(&["self", "upscale_factor"])),
            (TORCH, &["PixelUnshuffle"], Sig::Exact(&["self", "downscale_factor"])),
            (TORCH, &["ConstantPad2d"], Sig::Exact(&["self", "padding", "value"])),
            (TORCH, &["ZeroPad2d", "ReflectionPad1d", "ReplicationPad3d"], Sig::Exact(&["self", "padding"])),
            (TORCH, &["Bilinear"], Sig::Params(&["self", "in1_features", "in2_features", "out_features"], &[])),
            (TORCH, &["CosineSimilarity"], Sig::Exact(&["self", "dim", "eps"])),
            (EQX, &["MLP"], Sig::Params(&["self", "in_size", "out_size"], &[])),
            // MLP is an equinox.nn concept; torch has no torch.nn.MLP.
            (TORCH, &["MLP"], Sig::Missing),
            (&["einops", "layers", "torch"], &["Rearrange"], Sig::Params(&["self", "pattern"], &[])),
            (&["einops", "layers", "torch"], &["Reduce"], Sig::Params(&["self", "pattern", "reduction"], &[])),
            (&["einops", "layers", "flax"], &["Rearrange"], Sig::Params(&["self", "pattern"], &[])),
            (&["einops", "layers", "flax"], &["Reduce"], Sig::Params(&["self", "pattern", "reduction"], &[])),
            (&["einops", "layers", "chainer"], &["Rearrange"], Sig::Missing),
        ];
        for (module, classes, expected) in rows {
            for class in *classes {
                let parts: Vec<String> = module.iter().chain([class]).map(|p| p.to_string()).collect();
                let signature = known_layer_signature(&parts);
                if let Sig::Missing = expected {
                    assert_eq!(signature, None, "{parts:?}");
                    continue;
                }
                let signature = signature.unwrap_or_else(|| panic!("no signature for {parts:?}"));
                assert_eq!(signature.owner.as_deref(), Some(*class), "{parts:?}");
                assert_eq!(signature.name, "__init__", "{parts:?}");
                match expected {
                    Sig::Exact(params) => assert_eq!(signature.params, *params, "{parts:?}"),
                    Sig::Params(prefix, contained) => {
                        assert_eq!(&signature.params[..prefix.len()], *prefix, "{parts:?}");
                        for param in *contained {
                            assert!(signature.params.iter().any(|p| p == param), "{parts:?} has no {param}");
                        }
                    }
                    Sig::Missing => unreachable!(),
                }
            }
        }
    }

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    /// Case, source code, expected layers by name.
    type ExtractRow<'a> = (&'a str, &'a str, &'a [(&'a str, LayerKind)]);

    fn check_layers(case: &str, layers: HashMap<String, LayerKind>, expected: &[(&str, LayerKind)]) {
        let expected: HashMap<String, LayerKind> = expected.iter().map(|(name, kind)| (s(name), kind.clone())).collect();
        assert_eq!(layers, expected, "case: {case}");
    }

    /// Layers resolved from source files on disk: a minimal `equinox.nn.Linear`
    /// and an unrelated `helpers.make_layer`.
    #[test]
    fn disk_extract_cases() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("equinox/nn")).unwrap();
        fs::write(tmp.path().join("equinox/nn/__init__.py"), "from ._linear import Linear").unwrap();
        fs::write(
            tmp.path().join("equinox/nn/_linear.py"),
            "class Linear:\n    def __init__(self, in_features, out_features, use_bias=True): pass",
        )
        .unwrap();
        fs::write(tmp.path().join("helpers.py"), "def make_layer(): pass").unwrap();
        let roots = vec![tmp.path().to_path_buf()];
        let rows: &[ExtractRow] = &[
            ("single linear", "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)", &[("layer", linear("3", "5"))]),
            (
                "multiple linears",
                "import equinox as eqx\na = eqx.nn.Linear(3, 5)\nb = eqx.nn.Linear(features, hidden)",
                &[("a", linear("3", "5")), ("b", linear("features", "hidden"))],
            ),
            ("from-import alias", "from equinox.nn import Linear as Lin\nlayer = Lin(in_features=3, out_features=5)", &[("layer", linear("3", "5"))]),
            (
                "skips non-layer calls",
                "import equinox as eqx\nimport helpers\nlayer = eqx.nn.Linear(3, 5)\nother = helpers.make_layer()",
                &[("layer", linear("3", "5"))],
            ),
            // Not in the built-in catalog, and not on disk.
            ("skips missing implementation", "import my_layers\nlayer = my_layers.Linear(3, 5)", &[]),
            ("reversed keyword order", "import equinox as eqx\nlayer = eqx.nn.Linear(out_features=5, in_features=3)", &[("layer", linear("3", "5"))]),
            ("extra constructor kwargs", "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5, use_bias=False, key=key)", &[("layer", linear("3", "5"))]),
            ("keyword overrides positional", "import equinox as eqx\nlayer = eqx.nn.Linear(3, 4, out_features=5)", &[("layer", linear("3", "5"))]),
            ("inside a function", "import equinox as eqx\ndef make():\n    layer = eqx.nn.Linear(3, 5)", &[("layer", linear("3", "5"))]),
            ("required feature missing", "import equinox as eqx\nlayer = eqx.nn.Linear(in_features=3)", &[]),
            (
                "last assignment wins",
                "import equinox as eqx\nlayer = eqx.nn.Linear(3, 5)\nlayer = eqx.nn.Linear(7, 11)",
                &[("layer", linear("7", "11"))],
            ),
        ];
        for (case, code, expected) in rows {
            let read = |path: &PathBuf| fs::read_to_string(path).ok();
            let layers = extract_layer_assignments(parse(code).root_node(), code, &roots, read, 5).unwrap();
            check_layers(case, layers, expected);
        }
    }

    /// Layers resolved from the built-in catalog, with no search roots and no
    /// file reads.
    #[test]
    fn catalog_extract_cases() {
        let conv2d_3_16_3 = || conv(2, "3", "16", "3", "1", "0");
        let rows: &[ExtractRow] = &[
            ("equinox linear", "import equinox as eqx\nlayer = eqx.nn.Linear(64, 128)", &[("layer", linear("64", "128"))]),
            ("torch linear", "import torch\nlayer = torch.nn.Linear(128, 256)", &[("layer", linear("128", "256"))]),
            ("equinox embedding", "import equinox as eqx\nemb = eqx.nn.Embedding(10000, 512)", &[("emb", embedding("512"))]),
            ("torch embedding kwarg", "import torch\nemb = torch.nn.Embedding(10000, embedding_dim=768)", &[("emb", embedding("768"))]),
            ("torch maxpool2d", "import torch\npool = torch.nn.MaxPool2d(2)", &[("pool", pool("MaxPool2d", 2, "2", "2", "0"))]),
            ("equinox maxpool2d default stride", "import equinox as eqx\npool = eqx.nn.MaxPool2d(2)", &[("pool", pool("MaxPool2d", 2, "2", "1", "0"))]),
            ("equinox use_ceil", "import equinox as eqx\npool = eqx.nn.AvgPool1d(2, 2, 0, True)", &[("pool", pool_with("AvgPool1d", 1, "2", "2", "0", "1", UNPROVEN_DIM))]),
            ("torch stride none", "import torch\npool = torch.nn.AvgPool2d(3, None)", &[("pool", pool("AvgPool2d", 2, "3", "3", "0"))]),
            (
                "torch maxpool positional dilation",
                "import torch\npool = torch.nn.MaxPool1d(3, 1, 0, 2, False, True)",
                &[("pool", pool_with("MaxPool1d", 1, "3", "1", "0", "2", "True"))],
            ),
            ("torch avgpool positional ceil mode", "import torch\npool = torch.nn.AvgPool1d(2, 2, 0, True)", &[("pool", pool_with("AvgPool1d", 1, "2", "2", "0", "1", "True"))]),
            ("torch maxpool return indices", "import torch\npool = torch.nn.MaxPool2d(2, return_indices=True)", &[]),
            ("flax embed", "import flax.linen as nn\nemb = nn.Embed(1000, 64)", &[("emb", embedding("64"))]),
            (
                "torch lstm options",
                "import torch.nn as nn\nr = nn.LSTM(8, 16, 2, True, True, 0.0, True, 4)",
                &[("r", rnn_with("LSTM", "8", "16", ["2", "True", "4", "True"]))],
            ),
            ("gru ignores proj_size", "import torch.nn as nn\nr = nn.GRU(8, 16, proj_size=4)", &[("r", rnn("GRU", "8", "16"))]),
            ("rnn fifth positional", "import torch.nn as nn\nr = nn.RNN(8, 16, 1, 'relu')", &[]),
            (
                "torch multihead attention options",
                "import torch.nn as nn\na = nn.MultiheadAttention(16, 4, add_zero_attn=True, batch_first=True)",
                &[("a", mha_with("16", "4", "True", "1"))],
            ),
            (
                "multihead attention unproven add_bias_kv",
                "import torch.nn as nn\na = nn.MultiheadAttention(16, 4, add_bias_kv=flag)",
                &[("a", mha_with("16", "4", "False", UNPROVEN_DIM))],
            ),
            (
                "adaptive avgpool",
                "import torch\npool = torch.nn.AdaptiveAvgPool2d(1)",
                &[("pool", LayerKind::AdaptivePool { name: s("AdaptiveAvgPool2d"), spatial_rank: 2, output_size: s("1") })],
            ),
            ("equinox conv2d", "import equinox as eqx\nlayer = eqx.nn.Conv2d(3, 16, 3)", &[("layer", conv2d_3_16_3())]),
            (
                "torch conv2d stride padding",
                "import torch\nlayer = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)",
                &[("layer", conv(2, "3", "16", "3", "2", "1"))],
            ),
            ("torch dropout", "import torch\nlayer = torch.nn.Dropout()", &[("layer", kept("Dropout"))]),
            ("from-import alias", "from equinox.nn import Linear as Lin\nlayer = Lin(in_features=3, out_features=5)", &[("layer", linear("3", "5"))]),
            ("non-layer call", "import jax.numpy as jnp\ny = jnp.sum(x)", &[]),
            // Neither equinox.nn.Linear nor torch.nn.Linear, and nothing on disk.
            ("user layer", "import my_module\nlayer = my_module.Linear(3, 5)", &[]),
            ("torch multihead attention", "import torch.nn as nn\nattn = nn.MultiheadAttention(512, 8)", &[("attn", mha("512"))]),
            // equinox binds (num_heads, query_size) positionally.
            ("equinox multihead attention", "import equinox as eqx\nattn = eqx.nn.MultiheadAttention(8, 512)", &[("attn", eqx_mha("512", "8"))]),
            (
                "equinox multihead attention keywords",
                "import equinox as eqx\nattn = eqx.nn.MultiheadAttention(num_heads=4, query_size=256)",
                &[("attn", eqx_mha("256", "4"))],
            ),
            (
                "torch flatten",
                "import torch\nlayer = torch.nn.Flatten(1, 2)",
                &[("layer", LayerKind::Flatten { start_dim: s("1"), end_dim: s("2") })],
            ),
            (
                "torch conv transpose2d",
                "import torch\nlayer = torch.nn.ConvTranspose2d(3, 16, 4, stride=2, padding=1)",
                &[("layer", conv_transpose(2, "3", "16", "4", "2", "1"))],
            ),
            (
                "torch conv transpose output padding and dilation",
                "import torch\nlayer = torch.nn.ConvTranspose2d(3, 16, 4, 2, 1, 1, 1, True, 2)",
                &[("layer", conv_transpose_with(2, "3", "16", "4", "2", "1", "1", "2"))],
            ),
            // equinox orders (stride, output_padding, padding, dilation).
            (
                "equinox conv transpose positional order",
                "import equinox as eqx\nlayer = eqx.nn.ConvTranspose2d(3, 16, 4, 2, 1, 0, 2)",
                &[("layer", conv_transpose_with(2, "3", "16", "4", "2", "0", "1", "2"))],
            ),
            (
                "torch sequential",
                "import torch.nn as nn\nnet = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))",
                &[("net", seq(vec![linear("784", "256"), kept("ReLU"), linear("256", "10")]))],
            ),
            (
                "equinox sequential list argument",
                "import equinox as eqx\nnet = eqx.nn.Sequential([eqx.nn.Linear(4, 8), eqx.nn.Linear(8, 2)])",
                &[("net", seq(vec![linear("4", "8"), linear("8", "2")]))],
            ),
            (
                "nested sequential",
                "import torch.nn as nn\nnet = nn.Sequential(nn.Sequential(nn.Linear(4, 8), nn.ReLU()), nn.Linear(8, 2))",
                &[("net", seq(vec![seq(vec![linear("4", "8"), kept("ReLU")]), linear("8", "2")]))],
            ),
            // A child that is not a layer constructor call leaves the whole
            // Sequential unclassified.
            ("sequential with callable child", "import torch.nn as nn\nnet = nn.Sequential(nn.Linear(4, 8), some_fn)", &[]),
            ("sequential with star args", "import torch.nn as nn\nnet = nn.Sequential(*layer_list)", &[]),
            (
                "sequential with unknown inner call",
                "import torch.nn as nn\nimport mystery\nnet = nn.Sequential(nn.Linear(4, 8), mystery.Thing(1, 2))",
                &[],
            ),
            (
                "spectral norm",
                "import equinox as eqx\nlayer = eqx.nn.SpectralNorm(eqx.nn.Linear(4, 8), \"weight\", key=None)",
                &[],
            ),
            ("weight norm", "import equinox as eqx\nlayer = eqx.nn.WeightNorm(eqx.nn.Conv2d(3, 16, 3))", &[("layer", conv2d_3_16_3())]),
            (
                "spectral norm of unknown layer",
                "import equinox as eqx\nimport mystery\nlayer = eqx.nn.SpectralNorm(mystery.Thing(1, 2), \"weight\")",
                &[],
            ),
            (
                "einops torch rearrange",
                "from einops.layers.torch import Rearrange\nlayer = Rearrange(\"b c h w -> b (c h w)\")",
                &[("layer", einops("Rearrange", "\"b c h w -> b (c h w)\"", &[]))],
            ),
            (
                "einops flax reduce kwargs",
                "from einops.layers.flax import Reduce\nlayer = Reduce(\"b (h w) c -> b h w c\", \"max\", h=14, w=14)",
                &[("layer", einops("Reduce", "\"b (h w) c -> b h w c\"", &[("h", "14"), ("w", "14")]))],
            ),
            ("torch unfold", "import torch\nlayer = torch.nn.Unfold(kernel_size=3, stride=1, padding=0)", &[("layer", unfold("3", "1", "0", "1"))]),
            ("torch fold", "import torch\nlayer = torch.nn.Fold(output_size=(4, 4), kernel_size=2)", &[("layer", fold("(4, 4)", "2"))]),
            ("local response norm", "import torch\nlayer = torch.nn.LocalResponseNorm(5)", &[("layer", kept("LocalResponseNorm"))]),
            ("self prefix", "import torch\nlayer = torch.nn.Linear(self.d_inner, 10)", &[("layer", linear("d_inner", "10"))]),
            (
                "self prefix in sequential child",
                "import torch.nn as nn\nnet = nn.Sequential(nn.Linear(self.d_inner, 10), nn.ReLU())",
                &[("net", seq(vec![linear("d_inner", "10"), kept("ReLU")]))],
            ),
        ];
        for (case, code, expected) in rows {
            let layers = extract_layer_assignments(parse(code).root_node(), code, &[], |_| None, 5).unwrap();
            check_layers(case, layers, expected);
        }
    }
}

#[cfg(test)]
mod overflow_tests {
    use super::*;

    const MAX: i64 = i64::MAX;

    #[test]
    fn test_integer_rules_give_unknown_on_overflow() {
        let max = MAX.to_string();
        let unknown = UNKNOWN_DIM.to_string();
        assert_eq!(apply_offset(&max, 1), unknown);
        assert_eq!(mul_dim(&u64::MAX.to_string(), 2), unknown);
        assert_eq!(conv_spatial_dim(&max, "0", "1", "0"), unknown);
        assert_eq!(conv_spatial_dim("5", "3", "1", &max), unknown);
        assert_eq!(conv_spatial_dim("n", "3", "1", &max), unknown);
        assert_eq!(conv_spatial_dim("n", "3", "2", &max), unknown);
        assert_eq!(conv_spatial_dim("n", "k", "2", &max), unknown);
        assert_eq!(conv_transpose_spatial_dim("5", "3", &max, "0", "0"), unknown);
        assert_eq!(conv_transpose_spatial_dim("5", "3", "1", &max, "0"), unknown);
        assert_eq!(conv_transpose_spatial_dim("n", "3", "1", &max, "0"), unknown);
        assert_eq!(conv_transpose_spatial_dim("5", "3", "1", "0", &max), unknown);
        assert_eq!(pool_ceil_spatial_dim(&max, "1", "1", "1"), unknown);
        assert_eq!(effective_kernel_size("3", &max), Some(unknown.clone()));
        assert_eq!(div_dim_by_dims("6", &[u64::MAX.to_string(), "2".to_string()]), Ok(unknown));
    }

    #[test]
    fn test_integer_rules_keep_an_exact_size() {
        assert_eq!(apply_offset("5", -2), "3");
        assert_eq!(apply_offset("n", -2), "n-2");
        assert_eq!(mul_dim("4", 2), "8");
        assert_eq!(conv_spatial_dim("10", "3", "2", "1"), "5");
        assert_eq!(conv_transpose_spatial_dim("5", "3", "2", "1", "0"), "9");
        assert_eq!(conv_transpose_spatial_dim("16", "3", "2", "1", "1"), "32");
        assert_eq!(effective_kernel_size("3", "2"), Some("5".to_string()));
    }

    #[test]
    fn test_pixel_rules_stay_defined_for_a_large_or_zero_factor() {
        let shape: Vec<String> = ["4", "8", "8"].map(String::from).to_vec();
        let shuffle = |factor: &str| LayerKind::PixelShuffle { upscale_factor: factor.to_string() };
        let unshuffle = |factor: &str| LayerKind::PixelUnshuffle { downscale_factor: factor.to_string() };
        let apply = |kind: LayerKind| apply_layer_kind(&kind, &shape, "l", "x").unwrap().unwrap();
        let large = (1u64 << 40).to_string();
        assert_eq!(apply(shuffle(&large))[0], UNKNOWN_DIM);
        assert_eq!(apply(shuffle("0"))[0], UNKNOWN_DIM);
        assert_eq!(apply(unshuffle("0"))[1], UNKNOWN_DIM);
        let pad = LayerKind::Pad { name: "ZeroPad1d".to_string(), spatial_rank: 1, padding: MAX.to_string() };
        assert_eq!(apply(pad)[2], UNKNOWN_DIM);
    }
}
