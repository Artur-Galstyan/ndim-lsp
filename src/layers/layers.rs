use std::collections::HashMap;

use tree_sitter::Node;

use crate::{helpers::get_arg, layers::equinox::equinox_linear};

#[derive(Debug)]
pub enum Framework {
    Equinox,
    Flax,
    PyTorch,
}

#[derive(Debug)]
pub enum LayerType {
    Linear,
    Unknown,
}

pub struct LayerInfo {
    pub layer_type: LayerType,
    pub in_features: String,
    pub out_features: String,
    pub framework: Framework,
}

pub fn try_parse_layer_constructor(
    node: Node<'_>,
    import_alias_map: &HashMap<String, String>,
    text: &str,
) -> Option<LayerInfo> {
    let Some(func_node) = node.child_by_field_name("function") else {
        return None;
    };
    let Some(args_node) = node.child_by_field_name("arguments") else {
        return None;
    };

    let Some(obj) = func_node.child_by_field_name("object") else {
        return None;
    };
    let Some(attr) = func_node.child_by_field_name("attribute") else {
        return None;
    };
    let Some(obj_name) = obj.utf8_text(text.as_bytes()).ok() else {
        return None;
    };
    let Some(attr_name) = attr.utf8_text(text.as_bytes()).ok() else {
        return None;
    };
    let resolved_object = if let Some((prefix, rest)) = obj_name.split_once('.') {
        match import_alias_map.get(prefix) {
            Some(resolved) => format!("{}.{}", resolved, rest),
            None => obj_name.to_string(),
        }
    } else {
        import_alias_map
            .get(obj_name)
            .cloned()
            .unwrap_or_else(|| obj_name.to_string())
    };

    match (resolved_object.as_str(), attr_name) {
        ("equinox.nn", "Linear") => equinox_linear(args_node, text),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::calls::get_calls;

    use super::*;
    use tree_sitter::Parser;

    fn get_ast(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    #[test]
    fn test_parse_equinox_linear_positional() {
        let code = "layer = equinox.nn.Linear(128, 64)";
        let tree = get_ast(code);
        let node = get_calls(tree.root_node(), code).unwrap();
        let map = HashMap::new();

        assert_eq!(node.len(), 1);

        let info = try_parse_layer_constructor(node[0], &map, code).unwrap();
        assert!(matches!(info.framework, Framework::Equinox));
        assert!(matches!(info.layer_type, LayerType::Linear));
        assert_eq!(info.in_features, "128");
        assert_eq!(info.out_features, "64");
    }

    #[test]
    fn test_parse_equinox_linear_kwargs() {
        let code = "layer = equinox.nn.Linear(out_features=64, in_features=128)";
        let tree = get_ast(code);
        let node = get_calls(tree.root_node(), code).unwrap();
        let map = HashMap::new();

        assert_eq!(node.len(), 1);

        let info = try_parse_layer_constructor(node[0], &map, code).unwrap();
        assert_eq!(info.in_features, "128");
        assert_eq!(info.out_features, "64");
    }

    #[test]
    fn test_parse_equinox_linear_aliased_import() {
        // e.g. `import equinox as eqx`
        let code = "layer = eqx.nn.Linear(128, 64)";
        let tree = get_ast(code);
        let node = get_calls(tree.root_node(), code).unwrap();

        let mut map = HashMap::new();
        map.insert("eqx".to_string(), "equinox".to_string());

        assert_eq!(node.len(), 1);

        let info = try_parse_layer_constructor(node[0], &map, code).unwrap();
        assert_eq!(info.in_features, "128");
    }

    #[test]
    fn test_parse_equinox_linear_aliased_module_import() {
        // e.g. `import equinox.nn as nn`
        let code = "layer = nn.Linear(128, 64)";
        let tree = get_ast(code);
        let node = get_calls(tree.root_node(), code).unwrap();

        let mut map = HashMap::new();
        map.insert("nn".to_string(), "equinox.nn".to_string());

        assert_eq!(node.len(), 1);

        let info = try_parse_layer_constructor(node[0], &map, code).unwrap();
        assert_eq!(info.in_features, "128");
    }

    #[test]
    fn test_parse_unknown_function() {
        let code = "layer = equinox.nn.Dropout(0.5)";
        let tree = get_ast(code);
        let node = get_calls(tree.root_node(), code).unwrap();
        let map = HashMap::new();

        assert_eq!(node.len(), 1);

        let info = try_parse_layer_constructor(node[0], &map, code);
        assert!(info.is_none());
    }

    #[test]
    fn test_parse_not_a_layer_call() {
        let code = "layer = print('hello')";
        let tree = get_ast(code);
        let node = get_calls(tree.root_node(), code).unwrap();
        let map = HashMap::new();

        assert_eq!(node.len(), 1);

        let info = try_parse_layer_constructor(node[0], &map, code);
        assert!(info.is_none());
    }

    #[test]
    fn test_parse_missing_args() {
        // Missing out_features
        let code = "layer = equinox.nn.Linear(128)";
        let tree = get_ast(code);
        let node = get_calls(tree.root_node(), code).unwrap();
        let map = HashMap::new();

        assert_eq!(node.len(), 1);

        let info = try_parse_layer_constructor(node[0], &map, code);
        assert!(info.is_none());
    }
}
