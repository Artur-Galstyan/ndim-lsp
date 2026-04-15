use tree_sitter::Node;

use crate::{
    helpers::get_arg,
    layers::layers::{Framework, LayerInfo, LayerType},
};

pub fn equinox_linear(args_node: Node, text: &str) -> Option<LayerInfo> {
    let Some(in_node) = get_arg(args_node, 0, "in_features", text) else {
        return None;
    };
    let Some(out_node) = get_arg(args_node, 1, "out_features", text) else {
        return None;
    };
    let Ok(in_features) = in_node.utf8_text(text.as_bytes()) else {
        return None;
    };
    let Ok(out_features) = out_node.utf8_text(text.as_bytes()) else {
        return None;
    };
    Some(LayerInfo {
        framework: Framework::Equinox,
        layer_type: LayerType::Linear,
        in_features: in_features.to_string(),
        out_features: out_features.to_string(),
    })
}
