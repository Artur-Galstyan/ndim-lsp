use std::collections::HashMap;
use std::sync::LazyLock;

use tree_sitter::{Node, Query, QueryCursor, Range, StreamingIterator};

use crate::types::*;

#[cfg(test)]
use crate::resolution::bind_call_arguments;

fn node_text(node: Node, text: &str) -> Result<String, String> {
    node.utf8_text(text.as_bytes())
        .map(|s| s.to_string())
        .map_err(|e| e.to_string())
}

pub fn extract_callable_signature(
    node: Node,
    text: &str,
    symbol: &PythonSymbol,
) -> Result<Option<PythonCallableSignature>, String> {
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
                name: node_text(name_node, text)?,
                value: node_text(value_node, text)?,
            });
        } else {
            args.push(CallArgument::Positional {
                value: node_text(child, text)?,
            });
        }
    }

    Ok(args)
}

pub fn extract_jaxtyping_shapes(node: Node, text: &str) -> Result<Vec<FunctionShapeScope>, String> {
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

    fn find_string_literal<'tree>(node: Node<'tree>) -> Option<Node<'tree>> {
        if node.kind() == "string" {
            return Some(node);
        }
        for i in 0..node.named_child_count() {
            let Some(child) = node.named_child(i as u32) else {
                continue;
            };
            if let Some(value) = find_string_literal(child) {
                return Some(value);
            }
        }
        None
    }

    fn contains_type_name(node: Node, text: &str, type_name: &str) -> Result<bool, String> {
        let value = node_text(node, text)?;
        if (node.kind() == "identifier" && value == type_name)
            || (node.kind() == "attribute" && value.ends_with(&format!(".{type_name}")))
        {
            return Ok(true);
        }

        for i in 0..node.named_child_count() {
            let Some(child) = node.named_child(i as u32) else {
                continue;
            };
            if contains_type_name(child, text, type_name)? {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn contains_array_type(node: Node, text: &str) -> Result<bool, String> {
        Ok(contains_type_name(node, text, "Array")?
            || (contains_type_name(node, text, "NDArray")?
                && contains_type_name(node, text, "Shape")?))
    }

    fn point_for_byte(text: &str, byte: usize) -> tree_sitter::Point {
        let prefix = &text.as_bytes()[..byte];
        let row = prefix.iter().filter(|&&b| b == b'\n').count();
        let line_start = prefix.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
        tree_sitter::Point::new(row, byte - line_start)
    }

    fn shape_dims(node: Node, text: &str) -> Result<Vec<(String, tree_sitter::Range)>, String> {
        let raw = node_text(node, text)?;
        let trimmed = raw.trim();
        let trim_start = raw.len() - raw.trim_start().len();
        let Some(quote_start) = trimmed.find(['"', '\'']) else {
            return Ok(Vec::new());
        };
        let prefix = trimmed[..quote_start].to_ascii_lowercase();
        if prefix.chars().any(|c| c != 'r' && c != 'u') {
            return Ok(Vec::new());
        }

        let quoted = &trimmed[quote_start..];
        let quote = quoted
            .chars()
            .next()
            .expect("invariant: quote_start points to a quote char");
        let triple = quote.to_string().repeat(3);
        let quote_len = if quoted.starts_with(&triple) && quoted.ends_with(&triple) {
            3
        } else if quoted.starts_with(quote) && quoted.ends_with(quote) {
            1
        } else {
            return Ok(Vec::new());
        };
        let unquoted = &quoted[quote_len..quoted.len() - quote_len];
        let base_byte = node.start_byte() + trim_start + quote_start + quote_len;
        let mut spans = Vec::new();
        let mut token_start = None;

        for (offset, ch) in unquoted.char_indices() {
            if ch.is_whitespace() || ch == ',' {
                if let Some(start) = token_start.take() {
                    spans.push((start, offset));
                }
            } else if token_start.is_none() {
                token_start = Some(offset);
            }
        }
        if let Some(start) = token_start {
            spans.push((start, unquoted.len()));
        }

        Ok(spans
            .into_iter()
            .map(|(start, end)| {
                let start_byte = base_byte + start;
                let end_byte = base_byte + end;
                (
                    unquoted[start..end].to_string(),
                    tree_sitter::Range {
                        start_byte,
                        end_byte,
                        start_point: point_for_byte(text, start_byte),
                        end_point: point_for_byte(text, end_byte),
                    },
                )
            })
            .collect())
    }

    fn visit(
        node: Node,
        text: &str,
        scopes: &mut Vec<FunctionShapeScope>,
        current_scope_idx: usize,
    ) -> Result<(), String> {
        // Determine the scope index children should use.
        // For function_definition we push a new scope; otherwise inherit.
        let child_scope_idx = if node.kind() == "function_definition" {
            let function_name = node
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(text.as_bytes()).ok())
                .map(|s| s.to_string());
            let mut function_shapes = HashMap::new();
            let mut param_order = Vec::new();
            let mut all_params = Vec::new();
            let mut dimension_sites = Vec::new();
            if let Some(parameters) = node.child_by_field_name("parameters") {
                for i in 0..parameters.named_child_count() {
                    let Some(parameter) = parameters.named_child(i as u32) else {
                        continue;
                    };
                    if let Some(name) = first_identifier(parameter, text)? {
                        all_params.push(name);
                    }
                    let Some(type_node) = parameter.child_by_field_name("type") else {
                        continue;
                    };
                    if !contains_array_type(type_node, text)? {
                        // Plain scalar-typed params (`decay: float = 0.9`)
                        // can never be array-shaped — seed them as rank-0
                        // ("scalar") so they broadcast correctly in binops
                        // (e.g. `mean + (1 - decay) * delta`) instead of
                        // going dark for lack of any shape info at all.
                        // Anything else non-array (str, a custom class,
                        // Optional[...], etc.) is left untouched.
                        if let Ok(type_name) = type_node.utf8_text(text.as_bytes())
                            && matches!(type_name, "int" | "float" | "bool" | "complex")
                            && let Some(name) = first_identifier(parameter, text)?
                        {
                            function_shapes.insert(name, Vec::new());
                        }
                        continue;
                    }
                    let Some(shape_node) = find_string_literal(type_node) else {
                        continue;
                    };
                    let parsed_dims = shape_dims(shape_node, text)?;
                    if parsed_dims.is_empty() {
                        continue;
                    }
                    let Some(name) = first_identifier(parameter, text)? else {
                        continue;
                    };
                    let dims = parsed_dims.iter().map(|(value, _)| value.clone()).collect();
                    for (axis, (value, range)) in parsed_dims.into_iter().enumerate() {
                        dimension_sites.push(ShapeDimensionSite {
                            binding: Some(name.clone()),
                            axis,
                            value,
                            range,
                        });
                    }
                    function_shapes.insert(name.clone(), dims);
                    param_order.push(name);
                }
            }
            let mut return_shape = None;
            if let Some(ret_type) = node.child_by_field_name("return_type")
                && contains_array_type(ret_type, text)?
                && let Some(shape_node) = find_string_literal(ret_type)
            {
                let parsed_dims = shape_dims(shape_node, text)?;
                if !parsed_dims.is_empty() {
                    return_shape = Some(parsed_dims.iter().map(|(value, _)| value.clone()).collect());
                    for (axis, (value, range)) in parsed_dims.into_iter().enumerate() {
                        dimension_sites.push(ShapeDimensionSite {
                            binding: None,
                            axis,
                            value,
                            range,
                        });
                    }
                }
            }


            let new_idx = scopes.len();
            scopes.push(FunctionShapeScope {
                function_name,
                start_byte: node.start_byte(),
                end_byte: node.end_byte(),
                shapes: function_shapes,
                return_shape,
                param_order,
                all_params,
                dimension_sites,
            });
            new_idx
        } else {
            current_scope_idx
        };

        // Handle annotated assignments: x: Float[Array, "dims"] = value
        // or forward declarations: x: Float[Array, "dims"]
        //
        // Note: class-body annotations (e.g. equinox.Module fields like
        // `weight: Float[Array, "in out"]`) also match here. Since we
        // don't push a scope for class_definition, they land in the
        // enclosing scope (usually module scope). This is the current
        // intended behavior — a future increment can add class scoping
        // once `self.x` lookups are wired.
        if node.kind() == "assignment"
            && let Some(type_node) = node.child_by_field_name("type")
            && let Some(left_node) = node.child_by_field_name("left")
            && left_node.kind() == "identifier"
            && contains_array_type(type_node, text)?
            && let Some(shape_node) = find_string_literal(type_node)
        {
            let parsed_dims = shape_dims(shape_node, text)?;
            if !parsed_dims.is_empty() {
                let name = node_text(left_node, text)?;
                let dims = parsed_dims.iter().map(|(value, _)| value.clone()).collect();
                for (axis, (value, range)) in parsed_dims.into_iter().enumerate() {
                    scopes[child_scope_idx].dimension_sites.push(ShapeDimensionSite {
                        binding: Some(name.clone()),
                        axis,
                        value,
                        range,
                    });
                }
                scopes[child_scope_idx].shapes.insert(name, dims);
            }
        }

        for i in 0..node.named_child_count() {
            let Some(child) = node.named_child(i as u32) else {
                continue;
            };
            visit(child, text, scopes, child_scope_idx)?;
        }

        Ok(())
    }

    let mut scopes = vec![FunctionShapeScope {
        function_name: None,
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
        shapes: HashMap::new(),
        return_shape: None,
        param_order: Vec::new(),
        all_params: Vec::new(),
        dimension_sites: Vec::new(),
    }];
    visit(node, text, &mut scopes, 0)?;
    Ok(scopes)
}

pub fn find_top_level_symbol(
    node: Node,
    text: &str,
    name: &str,
) -> Result<Option<PythonSymbol>, String> {
    const QUERY_STRING: &str = r#"
        (module (class_definition name: (_) @cls_def))
        (module (function_definition name: (_) @fn_def))
        (module (import_statement) @import)
        (module (import_from_statement) @import)
    "#;

    static QUERY: LazyLock<Query> = LazyLock::new(|| {
        Query::new(&tree_sitter_python::LANGUAGE.into(), QUERY_STRING).expect("static query")
    });
    let query = &*QUERY;
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
    let mut matches = cursor.matches(query, node, text.as_bytes());
    let mut found = None;

    while let Some(match_) = matches.next() {
        match match_.pattern_index {
            0 => {
                for capture in match_.captures {
                    if capture.index == class_idx {
                        let class_name = node_text(capture.node, text)?;

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
                        let fn_name = node_text(capture.node, text)?;

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

    const QUERY_STRING: &str = r#"
        (import_statement name: (_) @import)
        (import_from_statement module_name: (_) @module_name name: (_) @name)
    "#;
    static QUERY: LazyLock<Query> = LazyLock::new(|| {
        Query::new(&tree_sitter_python::LANGUAGE.into(), QUERY_STRING).expect("static query")
    });
    let query = &*QUERY;

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
    let mut matches = cursor.matches(query, node, text.as_bytes());

    let mut import_map: HashMap<String, ImportPath> = HashMap::new();

    while let Some(match_) = matches.next() {
        if match_.pattern_index == 0 {
            for capture in match_.captures {
                if capture.index == import_statement_idx {
                    match capture.node.kind() {
                        "dotted_name" => {
                            let dotted_name = node_text(capture.node, text)?;
                            let (module, name, first) =
                                plain_import_dotted_name_to_vec(&dotted_name)
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
                            let dotted_name = node_text(name_child, text)?;
                            let (module, name, _) = plain_import_dotted_name_to_vec(&dotted_name)?;

                            let Some(alias_node) = capture.node.child_by_field_name("alias") else {
                                return Err("aliased_import missing alias".to_string());
                            };
                            let alias = node_text(alias_node, text)?;

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
                                node_text(n, text)?;
                            module = identifier.split(".").map(|f| f.to_string()).collect();
                        }
                        "relative_import" => {
                            let mut child_cursor = n.walk();
                            for child in n.named_children(&mut child_cursor) {
                                match child.kind() {
                                    "import_prefix" => {
                                        let prefix = node_text(child, text)?;
                                        dots = prefix.len();
                                    }
                                    "dotted_name" => {
                                        let dotted = node_text(child, text)?;
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
                            name = node_text(n, text)?;
                            alias = name.clone();
                        }
                        "aliased_import" => {
                            let Some(alias_child_node) = n.child_by_field_name("alias") else {
                                return Err("aliased_import has no alias field".to_string());
                            };
                            alias = node_text(alias_child_node, text)?;

                            let Some(dotted_name_idx) = n.child_by_field_name("name") else {
                                return Err("aliased_import has no name field".to_string());
                            };
                            let dotted_name = node_text(dotted_name_idx, text)?;
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

pub fn extract_method_calls(node: Node, text: &str) -> Result<Vec<MethodCallInfo>, String> {
    let mut result: Vec<MethodCallInfo> = Vec::new();
    const QUERY_STRING: &str = r#"
        (assignment
          left: (identifier) @var
          right: (call
            function: (attribute
              object: (identifier) @receiver
              attribute: (identifier) @method
            )
          )
        ) @assignment
    "#;

    static QUERY: LazyLock<Query> = LazyLock::new(|| {
        Query::new(&tree_sitter_python::LANGUAGE.into(), QUERY_STRING).expect("static query")
    });
    let query = &*QUERY;

    let Some(var_idx) = query.capture_index_for_name("var") else {
        return Err("var not found".to_string());
    };
    let Some(receiver_idx) = query.capture_index_for_name("receiver") else {
        return Err("receiver not found".to_string());
    };
    let Some(method_idx) = query.capture_index_for_name("method") else {
        return Err("method not found".to_string());
    };
    let Some(assignment_idx) = query.capture_index_for_name("assignment") else {
        return Err("assignment not found".to_string());
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, node, text.as_bytes());

    while let Some(match_) = matches.next() {
        let mut variable = String::new();
        let mut receiver = String::new();
        let mut method = String::new();
        let mut args_node_range: Option<Range> = None;

        for capture in match_.captures {
            if capture.index == var_idx {
                variable = node_text(capture.node, text)?;
            } else if capture.index == receiver_idx {
                receiver = node_text(capture.node, text)?;
            } else if capture.index == method_idx {
                method = node_text(capture.node, text)?;
            } else if capture.index == assignment_idx {
                let Some(right_child_node) = capture.node.child_by_field_name("right") else {
                    return Err("right child not found".to_string());
                };
                let Some(arguments_node) = right_child_node.child_by_field_name("arguments") else {
                    return Err("arguments child not found".to_string());
                };
                args_node_range = Some(arguments_node.range());
            }
        }

        let args_node_range = args_node_range.ok_or("args_node_range not found".to_string())?;
        result.push(MethodCallInfo {
            variable,
            receiver,
            method,
            args_node_range,
        });
    }

    Ok(result)
}

pub fn extract_calls(node: Node, text: &str) -> Result<Vec<CallInfo>, String> {
    let mut result: Vec<CallInfo> = Vec::new();
    const QUERY_STRING: &str = r#"
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

    static QUERY: LazyLock<Query> = LazyLock::new(|| {
        Query::new(&tree_sitter_python::LANGUAGE.into(), QUERY_STRING).expect("static query")
    });
    let query = &*QUERY;

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
    let mut matches = cursor.matches(query, node, text.as_bytes());

    while let Some(match_) = matches.next() {
        let mut variable = String::new();
        let mut target = String::new();
        let mut args_node_range: Option<Range> = None;

        for capture in match_.captures {
            if capture.index == var_idx {
                variable = node_text(capture.node, text)?;
            } else if capture.index == assignment_idx {
                let Some(right_child_node) = capture.node.child_by_field_name("right") else {
                    return Err("right child not found".to_string());
                };

                let Some(arguments_node) = right_child_node.child_by_field_name("arguments") else {
                    return Err("arguments child not found".to_string());
                };
                args_node_range = Some(arguments_node.range());
            } else if capture.index == fn_id_idx || capture.index == fn_attr_idx {
                target = node_text(capture.node, text)?;
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

/// Like `extract_calls`, but for assignments whose LHS is `self.<attr>`
/// (e.g. `self.input_proj = eqx.nn.Linear(d_inner, dt_rank)` in an
/// `__init__`). `CallInfo::variable` is the bare attribute name
/// (`input_proj`). Used to resolve `self.<attr>` references to a layer.
pub fn extract_self_attr_calls(node: Node, text: &str) -> Result<Vec<CallInfo>, String> {
    let mut result: Vec<CallInfo> = Vec::new();
    const QUERY_STRING: &str = r#"
        (assignment
          left: (attribute
            object: (identifier) @obj
            attribute: (identifier) @attr)
          right: (call
            function: [
              (identifier) @fn_id
              (attribute) @fn_attr
            ]
          )
        ) @assignment
    "#;

    static QUERY: LazyLock<Query> = LazyLock::new(|| {
        Query::new(&tree_sitter_python::LANGUAGE.into(), QUERY_STRING).expect("static query")
    });
    let query = &*QUERY;

    let obj_idx = query.capture_index_for_name("obj").ok_or("obj not found")?;
    let attr_idx = query.capture_index_for_name("attr").ok_or("attr not found")?;
    let fn_id_idx = query.capture_index_for_name("fn_id").ok_or("fn_id not found")?;
    let fn_attr_idx = query.capture_index_for_name("fn_attr").ok_or("fn_attr not found")?;
    let assignment_idx = query
        .capture_index_for_name("assignment")
        .ok_or("assignment not found")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, node, text.as_bytes());

    while let Some(match_) = matches.next() {
        let mut is_self = false;
        let mut variable = String::new();
        let mut target = String::new();
        let mut args_node_range: Option<Range> = None;

        for capture in match_.captures {
            if capture.index == obj_idx {
                is_self = node_text(capture.node, text)? == "self";
            } else if capture.index == attr_idx {
                variable = node_text(capture.node, text)?;
            } else if capture.index == assignment_idx {
                let right = capture
                    .node
                    .child_by_field_name("right")
                    .ok_or("right child not found")?;
                let arguments = right
                    .child_by_field_name("arguments")
                    .ok_or("arguments child not found")?;
                args_node_range = Some(arguments.range());
            } else if capture.index == fn_id_idx || capture.index == fn_attr_idx {
                target = node_text(capture.node, text)?;
            }
        }

        if !is_self {
            continue;
        }
        let args_node_range = args_node_range.ok_or("args_node_range not found")?;
        result.push(CallInfo {
            variable,
            target,
            args_node_range,
        });
    }

    Ok(result)
}

/// Collect `self.<attr> = <identifier>` assignments (typically the
/// `self.dt_rank = dt_rank` lines in an `__init__`) into an alias map
/// `{attr -> identifier}`. Used to canonicalize symbolic dims so that
/// `self.dt_rank` and `dt_rank` (the same value) compare equal.
///
/// Only identifier right-hand sides are captured (the dominant
/// "store the constructor arg" pattern); expression RHS is skipped.
/// The map is file-global; cross-class collisions on the same attribute
/// name resolve last-wins.
/// Resolve `self.<attr> = <ident>` assignments (typically in `__init__`) to a
/// map of bare attribute name → aliased identifier. Flat last-wins view of
/// `extract_self_attr_aliases_by_class`. Cross-class same-named attrs
/// collide (last one wins); use the by-class version to avoid that.
pub fn extract_self_attr_aliases(
    node: Node,
    text: &str,
) -> Result<HashMap<String, String>, String> {
    let by_class = extract_self_attr_aliases_by_class(node, text)?;
    Ok(by_class
        .into_iter()
        .filter_map(|(attr, mut entries)| entries.pop().map(|e| (attr, e.value)))
        .collect())
}

/// Byte ranges of every `class_definition` in the tree (DFS, includes
/// nested). Mirrors `layers.rs`'s private `class_ranges` helper.
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

/// Innermost class range containing `byte`; whole file if none. Mirrors
/// `layers.rs`'s private `enclosing_class_range` helper.
fn enclosing_class_range(classes: &[(usize, usize)], byte: usize) -> (usize, usize) {
    classes
        .iter()
        .filter(|(s, e)| *s <= byte && byte < *e)
        .min_by_key(|(s, e)| e - s)
        .copied()
        .unwrap_or((0, usize::MAX))
}

/// Like `extract_self_attr_aliases`, but each binding keeps the byte range of
/// its enclosing `class_definition` (issue: the alias map was file-global,
/// so same-named `self.<attr>` aliases in different classes collided
/// last-wins) — mirrors `extract_self_attr_layers_by_class` in `layers.rs`.
/// Lookup goes through `ShapeCtx::self_attr_alias_at` (innermost class
/// containing the use-site byte; lone-binding fallback).
pub fn extract_self_attr_aliases_by_class(
    node: Node,
    text: &str,
) -> Result<HashMap<String, Vec<ScopedSelfAttrAlias>>, String> {
    let mut aliases: HashMap<String, Vec<ScopedSelfAttrAlias>> = HashMap::new();
    const QUERY_STRING: &str = r#"
        (assignment
          left: (attribute
            object: (identifier) @obj
            attribute: (identifier) @attr)
          right: (identifier) @val) @assign
    "#;

    static QUERY: LazyLock<Query> = LazyLock::new(|| {
        Query::new(&tree_sitter_python::LANGUAGE.into(), QUERY_STRING).expect("static query")
    });
    let query = &*QUERY;
    let obj_idx = query.capture_index_for_name("obj").ok_or("obj not found")?;
    let attr_idx = query.capture_index_for_name("attr").ok_or("attr not found")?;
    let val_idx = query.capture_index_for_name("val").ok_or("val not found")?;
    let assign_idx = query
        .capture_index_for_name("assign")
        .ok_or("assign not found")?;

    let classes = class_ranges(node);
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, node, text.as_bytes());

    while let Some(match_) = matches.next() {
        let mut is_self = false;
        let mut attr = String::new();
        let mut val = String::new();
        let mut assign_byte = 0usize;
        for capture in match_.captures {
            if capture.index == assign_idx {
                assign_byte = capture.node.start_byte();
                continue;
            }
            let t = node_text(capture.node, text)?;
            if capture.index == obj_idx {
                is_self = t == "self";
            } else if capture.index == attr_idx {
                attr = t.to_string();
            } else if capture.index == val_idx {
                val = t.to_string();
            }
        }
        if is_self && !attr.is_empty() {
            let (class_start, class_end) = enclosing_class_range(&classes, assign_byte);
            aliases.entry(attr).or_default().push(ScopedSelfAttrAlias {
                class_start,
                class_end,
                value: val,
            });
        }
    }

    Ok(aliases)
}

#[cfg(test)]
mod self_attr_alias_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    #[test]
    fn test_flat_wrapper_single_class() {
        let code = "class A:\n    def __init__(self):\n        self.dt_rank = dt_rank\n";
        let tree = parse(code);

        let aliases = extract_self_attr_aliases(tree.root_node(), code).unwrap();

        assert_eq!(aliases.get("dt_rank").map(String::as_str), Some("dt_rank"));
    }

    #[test]
    fn test_by_class_keeps_each_class_own_range() {
        let code = "class A:\n    def __init__(self):\n        self.rank = dt_rank\n\nclass B:\n    def __init__(self):\n        self.rank = other_rank\n";
        let tree = parse(code);

        let by_class = extract_self_attr_aliases_by_class(tree.root_node(), code).unwrap();

        let entries = by_class.get("rank").expect("rank alias recorded");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].value, "dt_rank");
        assert_eq!(entries[1].value, "other_rank");
        // Each entry's class range is disjoint (class A's range ends where
        // class B's starts, or earlier).
        assert!(entries[0].class_end <= entries[1].class_start);
    }

    #[test]
    fn test_flat_wrapper_last_wins_on_cross_class_collision() {
        // Documents the pre-existing (still-supported) flat wrapper's
        // last-wins behavior for callers that haven't moved to class-scoped
        // lookup — same convention as `extract_self_attr_layers`.
        let code = "class A:\n    def __init__(self):\n        self.rank = dt_rank\n\nclass B:\n    def __init__(self):\n        self.rank = other_rank\n";
        let tree = parse(code);

        let aliases = extract_self_attr_aliases(tree.root_node(), code).unwrap();

        assert_eq!(aliases.get("rank").map(String::as_str), Some("other_rank"));
    }

    #[test]
    fn test_no_aliases_returns_empty_map() {
        let code = "class A:\n    def __init__(self):\n        self.linear = Linear(3, 5)\n";
        let tree = parse(code);

        let by_class = extract_self_attr_aliases_by_class(tree.root_node(), code).unwrap();

        assert!(by_class.is_empty());
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

    fn scope_by_name<'a>(scopes: &'a [FunctionShapeScope], name: &str) -> &'a FunctionShapeScope {
        scopes
            .iter()
            .find(|s| s.function_name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no scope for function '{}'", name))
    }

    fn module_scope(scopes: &[FunctionShapeScope]) -> &FunctionShapeScope {
        assert!(
            scopes[0].function_name.is_none(),
            "scope[0] is not module scope"
        );
        &scopes[0]
    }

    #[test]
    fn test_extracts_single_jaxtyping_shape() {
        let code = "def f(x: Float[Array, \"batch features\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let f = scope_by_name(&scopes, "f");
        assert_eq!(f.shapes.get("x"), Some(&shape(&["batch", "features"])));
        assert!(module_scope(&scopes).shapes.is_empty());
    }

    #[test]
    fn test_dimension_sites_keep_exact_source_ranges() {
        let code = r#"def f(x: Float[Array, r" batch, hidden*2 "]) -> NDArray[Shape["batch, output"], Float]: pass"#;
        let tree = parse(code);
        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();
        let sites = &scope_by_name(&scopes, "f").dimension_sites;

        assert_eq!(sites.len(), 4);
        assert_eq!(sites[0].binding.as_deref(), Some("x"));
        assert_eq!(sites[0].axis, 0);
        assert_eq!(&code[sites[0].range.start_byte..sites[0].range.end_byte], "batch");
        assert_eq!(&code[sites[1].range.start_byte..sites[1].range.end_byte], "hidden*2");
        assert_eq!(sites[2].binding, None);
        assert_eq!(&code[sites[3].range.start_byte..sites[3].range.end_byte], "output");
    }

    #[test]
    fn test_extracts_concrete_integer_dimension() {
        let code = "def f(x: Float[Array, \"batch 3\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "3"]))
        );
    }

    #[test]
    fn test_extracts_multiple_parameters() {
        let code = "def f(x: Float[Array, \"b d\"], y: Int[Array, \"b\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let f = scope_by_name(&scopes, "f");
        assert_eq!(f.shapes.get("x"), Some(&shape(&["b", "d"])));
        assert_eq!(f.shapes.get("y"), Some(&shape(&["b"])));
        assert_eq!(f.shapes.len(), 2);
    }

    #[test]
    fn test_extracts_typed_default_parameter() {
        let code = "def f(x: Float[Array, \"b d\"] = default): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["b", "d"]))
        );
    }

    #[test]
    fn test_skips_unannotated_parameter() {
        let code = "def f(x, y: Float[Array, \"b d\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let f = scope_by_name(&scopes, "f");
        assert!(!f.shapes.contains_key("x"));
        assert_eq!(f.shapes.get("y"), Some(&shape(&["b", "d"])));
    }

    #[test]
    fn test_scalar_typed_param_seeded_as_rank_zero() {
        // `x: int` (a plain scalar Python type, not a jaxtyping array) can
        // never be array-shaped — it's seeded as rank-0 ("scalar") so it
        // broadcasts correctly in binops (e.g. `arr + x`) instead of going
        // dark for lack of any shape info at all (lazy call-site parameter
        // seeding's `decay: float`-style params).
        let code = "def f(x: int): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&Vec::<String>::new())
        );
    }

    #[test]
    fn test_non_scalar_non_array_annotation_still_skipped() {
        // A plain type annotation that is neither a jaxtyping array nor one
        // of the recognized scalar types (`int`/`float`/`bool`/`complex`) —
        // e.g. a custom class or `str` — has no static shape info and is
        // left out of `shapes` entirely (not even a rank-0 guess).
        let code = "def f(x: str, y: MyClass): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let f = scope_by_name(&scopes, "f");
        assert!(!f.shapes.contains_key("x"));
        assert!(!f.shapes.contains_key("y"));
    }

    #[test]
    fn test_extracts_shapes_inside_nested_function() {
        let code = "def outer():\n    def inner(x: Float[Array, \"b d\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let outer = scope_by_name(&scopes, "outer");
        let inner = scope_by_name(&scopes, "inner");
        assert_eq!(inner.shapes.get("x"), Some(&shape(&["b", "d"])));
        assert!(!outer.shapes.contains_key("x"));
        assert!(outer.shapes.is_empty());
    }

    #[test]
    fn test_extracts_single_quoted_shape() {
        let code = "def f(x: Float[Array, 'batch features']): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_extracts_multiline_function_signature() {
        let code = "def f(\n    x: Float[Array, \"batch features\"],\n    y: Float[Array, \"batch\"],\n): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let f = scope_by_name(&scopes, "f");
        assert_eq!(f.shapes.get("x"), Some(&shape(&["batch", "features"])));
        assert_eq!(f.shapes.get("y"), Some(&shape(&["batch"])));
    }

    #[test]
    fn test_extracts_method_parameter_but_skips_self() {
        let code = "class M:\n    def __call__(self, x: Float[Array, \"batch features\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let call = scope_by_name(&scopes, "__call__");
        assert!(!call.shapes.contains_key("self"));
        assert_eq!(call.shapes.get("x"), Some(&shape(&["batch", "features"])));
    }

    #[test]
    fn test_return_annotation_is_captured() {
        let code = "def f(x) -> Float[Array, \"batch features\"]: pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").return_shape,
            Some(vec!["batch".into(), "features".into()])
        );
    }

    #[test]
    fn test_return_annotation_non_array_returns_none() {
        let code = "def f(x) -> int: pass";
        let tree = parse(code);
        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();
        assert_eq!(scope_by_name(&scopes, "f").return_shape, None);

        let code = "def f(x) -> str: pass";
        let tree = parse(code);
        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();
        assert_eq!(scope_by_name(&scopes, "f").return_shape, None);

        let code = "def f(x) -> Literal[\"x\"]: pass";
        let tree = parse(code);
        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();
        assert_eq!(scope_by_name(&scopes, "f").return_shape, None);
    }

    #[test]
    fn test_return_annotation_array_with_concrete_dims() {
        let code = "def f(x) -> Float[Array, \"3 4\"]: pass";
        let tree = parse(code);
        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();
        assert_eq!(
            scope_by_name(&scopes, "f").return_shape,
            Some(vec!["3".into(), "4".into()])
        );
    }

    #[test]
    fn test_return_annotation_missing() {
        let code = "def f(x): pass";
        let tree = parse(code);
        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();
        assert_eq!(scope_by_name(&scopes, "f").return_shape, None);
    }

    #[test]
    fn test_return_annotation_inside_nested_function() {
        let code = "\
def f(x) -> Float[Array, \"batch\"]:
    def g(y) -> Float[Array, \"features\"]:
        pass
";
        let tree = parse(code);
        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();
        assert_eq!(
            scope_by_name(&scopes, "f").return_shape,
            Some(vec!["batch".into()])
        );
        assert_eq!(
            scope_by_name(&scopes, "g").return_shape,
            Some(vec!["features".into()])
        );
    }

    #[test]
    fn test_module_scope_has_no_return_shape() {
        let code = "x = 1";
        let tree = parse(code);
        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();
        assert_eq!(module_scope(&scopes).return_shape, None);
    }

    #[test]
    fn test_non_array_string_annotation_is_skipped() {
        let code = "def f(x: Literal[\"batch features\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(scope_by_name(&scopes, "f").shapes.is_empty());
    }

    #[test]
    fn test_notarray_identifier_is_not_treated_as_array() {
        let code = "def f(x: Float[NotArray, \"batch features\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(scope_by_name(&scopes, "f").shapes.is_empty());
    }

    #[test]
    fn test_qualified_jax_array_is_accepted() {
        let code = "def f(x: Float[jax.Array, \"batch features\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_qualified_jaxtyping_array_is_accepted() {
        let code = "def f(x: Float[jaxtyping.Array, \"batch features\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_shaped_array_annotation_is_accepted() {
        let code = "def f(x: Shaped[Array, \"batch features\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_optional_wrapped_array_annotation_is_accepted() {
        let code = "def f(x: Optional[Float[Array, \"batch features\"]]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_raw_shape_string_is_accepted() {
        let code = "def f(x: Float[Array, r\"batch features\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_triple_quoted_shape_string_is_accepted() {
        let code = "def f(x: Float[Array, \"\"\"batch features\"\"\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_f_string_shape_is_skipped() {
        let code = "def f(x: Float[Array, f\"batch {features}\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(scope_by_name(&scopes, "f").shapes.is_empty());
    }

    #[test]
    fn test_annotated_varargs_are_extracted() {
        let code = "def f(*xs: Float[Array, \"batch features\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("xs"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_annotated_kwargs_are_extracted() {
        let code = "def f(**kwargs: Float[Array, \"batch features\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("kwargs"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_keyword_only_annotated_parameter_is_extracted() {
        let code = "def f(*, x: Float[Array, \"batch features\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_positional_only_annotated_parameter_is_extracted() {
        let code = "def f(x: Float[Array, \"batch features\"], /): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_deeply_nested_union_annotation_is_extracted() {
        let code = "def f(x: Union[None, Float[Array, \"batch features\"]]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_bytes_shape_string_is_skipped() {
        let code = "def f(x: Float[Array, b\"batch features\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(scope_by_name(&scopes, "f").shapes.is_empty());
    }

    #[test]
    fn test_comment_near_annotation_does_not_affect_shape() {
        let code = "def f(\n    x: Float[Array, \"batch features\"],  # important\n): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_empty_shape_string_is_skipped() {
        let code = "def f(x: Float[Array, \"\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(scope_by_name(&scopes, "f").shapes.is_empty());
    }

    #[test]
    fn test_extra_spaces_in_shape_string_are_ignored() {
        let code = "def f(x: Float[Array, \"  batch   features  \"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "features"]))
        );
    }

    #[test]
    fn test_preserves_punctuation_inside_dimension_names() {
        let code = "def f(x: Float[Array, \"batch hidden*2\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert_eq!(
            scope_by_name(&scopes, "f").shapes.get("x"),
            Some(&shape(&["batch", "hidden*2"]))
        );
    }

    #[test]
    fn test_same_param_name_in_two_functions_kept_separate() {
        let code = "def f(x: Float[Array, \"a b\"]): pass\ndef g(x: Float[Array, \"c d\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let f = scope_by_name(&scopes, "f");
        let g = scope_by_name(&scopes, "g");
        assert_eq!(f.shapes.get("x"), Some(&shape(&["a", "b"])));
        assert_eq!(g.shapes.get("x"), Some(&shape(&["c", "d"])));
    }

    #[test]
    fn test_module_scope_is_present_and_empty_for_function_only_file() {
        let code = "def f(x: Float[Array, \"a b\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let module = module_scope(&scopes);
        assert!(module.shapes.is_empty());
        assert_eq!(module.start_byte, tree.root_node().start_byte());
        assert_eq!(module.end_byte, tree.root_node().end_byte());
    }

    #[test]
    fn test_function_scope_byte_range_covers_its_definition() {
        let code = "x = 1\ndef f(x: Float[Array, \"a b\"]): pass\ny = 2";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let f = scope_by_name(&scopes, "f");
        let def_start = code.find("def f").unwrap();
        let def_end = code.find("pass").unwrap() + "pass".len();
        assert!(f.start_byte <= def_start);
        assert!(f.end_byte >= def_end);
        assert!(f.start_byte >= module_scope(&scopes).start_byte);
        assert!(f.end_byte <= module_scope(&scopes).end_byte);
    }

    #[test]
    fn test_nested_function_scopes_are_distinct() {
        let code =
            "def outer(o: Float[Array, \"a b\"]):\n    def inner(i: Float[Array, \"c d\"]): pass";
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let outer = scope_by_name(&scopes, "outer");
        let inner = scope_by_name(&scopes, "inner");
        assert_eq!(outer.shapes.get("o"), Some(&shape(&["a", "b"])));
        assert!(!outer.shapes.contains_key("i"));
        assert_eq!(inner.shapes.get("i"), Some(&shape(&["c", "d"])));
        assert!(!inner.shapes.contains_key("o"));
        assert!(inner.start_byte >= outer.start_byte);
        assert!(inner.end_byte <= outer.end_byte);
    }

    // --- Annotated assignment tests ---

    #[test]
    fn test_annotated_assignment_node_structure() {
        // Verify tree-sitter structure: assignment with type field and left identifier
        let code = r#"x: Float[Array, "a b"] = None"#;
        let tree = parse(code);

        let root = tree.root_node();
        // module > expression_statement > assignment
        let expr_stmt = root.named_child(0).unwrap();
        assert_eq!(expr_stmt.kind(), "expression_statement");
        let assignment = expr_stmt.named_child(0).unwrap();
        assert_eq!(assignment.kind(), "assignment");
        // assignment has "left" and "type" fields
        let left = assignment.child_by_field_name("left").unwrap();
        assert_eq!(left.kind(), "identifier");
        assert_eq!(left.utf8_text(code.as_bytes()).unwrap(), "x");
        let type_node = assignment.child_by_field_name("type").unwrap();
        assert!(type_node.kind() == "type");
        // right field exists when there's a value
        let right = assignment.child_by_field_name("right").unwrap();
        assert_eq!(right.kind(), "none");
    }

    #[test]
    fn test_forward_decl_annotated_assignment_node_structure() {
        // x: Float[Array, "n"] without a value
        let code = r#"x: Float[Array, "n"]"#;
        let tree = parse(code);

        let root = tree.root_node();
        let expr_stmt = root.named_child(0).unwrap();
        let assignment = expr_stmt.named_child(0).unwrap();
        assert_eq!(assignment.kind(), "assignment");
        let left = assignment.child_by_field_name("left").unwrap();
        assert_eq!(left.kind(), "identifier");
        let type_node = assignment.child_by_field_name("type").unwrap();
        assert!(type_node.kind() == "type");
        // No right field for forward declaration
        assert!(assignment.child_by_field_name("right").is_none());
    }

    #[test]
    fn test_module_level_annotated_assignment() {
        let code = r#"x: Float[Array, "a b"] = None"#;
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let module = module_scope(&scopes);
        assert_eq!(module.shapes.get("x"), Some(&shape(&["a", "b"])));
    }

    #[test]
    fn test_multiple_module_level_annotated_assignments() {
        let code = r#"x: Float[Array, "a b"] = None
y: Int[Array, "c"] = 0"#;
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let module = module_scope(&scopes);
        assert_eq!(module.shapes.get("x"), Some(&shape(&["a", "b"])));
        assert_eq!(module.shapes.get("y"), Some(&shape(&["c"])));
        assert_eq!(module.shapes.len(), 2);
    }

    #[test]
    fn test_annotated_assignment_inside_function() {
        let code = r#"def f():
    x: Float[Array, "a b"] = None"#;
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let f = scope_by_name(&scopes, "f");
        assert_eq!(f.shapes.get("x"), Some(&shape(&["a", "b"])));
        assert!(module_scope(&scopes).shapes.is_empty());
    }

    #[test]
    fn test_annotated_assignment_non_array_type_ignored() {
        let code = r#"x: int = 5"#;
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        assert!(module_scope(&scopes).shapes.is_empty());
    }

    #[test]
    fn test_annotated_assignment_without_value() {
        let code = r#"x: Float[Array, "n"]"#;
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let module = module_scope(&scopes);
        assert_eq!(module.shapes.get("x"), Some(&shape(&["n"])));
    }

    #[test]
    fn test_annotated_assignment_in_nested_function_goes_to_inner_scope() {
        let code = r#"def outer():
    x: Float[Array, "a b"] = None
    def inner():
        y: Float[Array, "c d"] = None"#;
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let outer = scope_by_name(&scopes, "outer");
        let inner = scope_by_name(&scopes, "inner");
        assert_eq!(outer.shapes.get("x"), Some(&shape(&["a", "b"])));
        assert!(!outer.shapes.contains_key("y"));
        assert_eq!(inner.shapes.get("y"), Some(&shape(&["c", "d"])));
        assert!(!inner.shapes.contains_key("x"));
        assert!(module_scope(&scopes).shapes.is_empty());
    }

    #[test]
    fn test_module_level_assignment_coexists_with_function_params() {
        let code = r#"x: Float[Array, "a b"] = None
def f(y: Float[Array, "c d"]): pass"#;
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let module = module_scope(&scopes);
        let f = scope_by_name(&scopes, "f");
        assert_eq!(module.shapes.get("x"), Some(&shape(&["a", "b"])));
        assert_eq!(f.shapes.get("y"), Some(&shape(&["c", "d"])));
        assert!(!module.shapes.contains_key("y"));
        assert!(!f.shapes.contains_key("x"));
    }

    #[test]
    fn test_annotated_assignment_in_class_body_lands_in_module_scope() {
        // Documents current behavior: class bodies don't push a scope,
        // so annotated fields land in the enclosing (module) scope.
        // A future increment can add class scoping once self.x lookups
        // are wired.
        let code = r#"class M:
    w: Float[Array, "in out"]"#;
        let tree = parse(code);

        let scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let module = module_scope(&scopes);
        assert_eq!(module.shapes.get("w"), Some(&shape(&["in", "out"])));
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

/// Walk the AST to find all `binary_operator` nodes that appear in a value position
/// (assignment RHS, return, yield, assert) and whose operands are both identifiers.
/// Handles arbitrary nesting of parenthesized_expression, expression_list, and
/// unary_operator wrappers — no pattern enumeration needed.
pub fn extract_binary_ops(node: Node, text: &str) -> Result<Vec<BinaryOpInfo>, String> {
    let mut result: Vec<BinaryOpInfo> = Vec::new();
    collect_binary_ops(node, text, &mut result)?;
    // Sort by start byte for source-order traversal.
    result.sort_by_key(|info| info.range.start_byte);
    Ok(result)
}

fn collect_binary_ops(
    node: Node,
    text: &str,
    result: &mut Vec<BinaryOpInfo>,
) -> Result<(), String> {
    let kind = node.kind();

    // Check if this node is a binary_operator with identifier operands.
    if kind == "binary_operator" {
        if let Some(info) = try_extract_binary_op(node, text)? {
            result.push(info);
        }
        // Don't recurse into children — operands are already handled.
    // TODO: If nested binary ops become in-scope (e.g. `(a + b) @ c`),
    // this early return must be removed so the walker can recurse into
    // non-identifier operands and find inner ops.
        return Ok(());
    }

    // Recurse into children for all other node types.
    for i in 0..node.child_count() {
        let Some(child) = node.child(i as u32) else { continue };
        collect_binary_ops(child, text, result)?;
    }
    Ok(())
}

/// If `node` is a binary_operator with two identifier operands, determine
/// whether it sits in a value position and return its info.
fn try_extract_binary_op(node: Node, text: &str) -> Result<Option<BinaryOpInfo>, String> {
    let left_child = node.child_by_field_name("left");
    let right_child = node.child_by_field_name("right");
    let Some(left_node) = left_child else { return Ok(None) };
    let Some(right_node) = right_child else { return Ok(None) };

    if left_node.kind() != "identifier" || right_node.kind() != "identifier" {
        return Ok(None);
    }

    // Find the operator via the named field.
    let Some(op_node) = node.child_by_field_name("operator") else {
        return Ok(None);
    };
    let op_text = node_text(op_node, text)?;

    let op = match op_text.as_str() {
        "@" => BinaryOp::MatMul,
        "+" => BinaryOp::Add,
        "-" => BinaryOp::Sub,
        "*" => BinaryOp::Mul,
        "/" => BinaryOp::Div,
        _ => return Ok(None),
    };

    // Walk up through transparent wrappers to find the enclosing context.
    // Returns None for contexts we don't track (e.g., tuple unpacking, call args).
    let variable = match resolve_variable_context(node, text) {
        Some(v) => v,
        None => return Ok(None),
    };

    Ok(Some(BinaryOpInfo {
        variable,
        left: node_text(left_node, text)?,
        right: node_text(right_node, text)?,
        op,
        range: node.range(),
    }))
}

/// Walk up from a binary_operator node through transparent wrappers
/// (parenthesized_expression, expression_list, unary_operator) to
/// determine the variable context.
/// - Returns `Some(varname)` for assignments with an identifier LHS.
/// - Returns `Some("")` for return/yield/assert contexts.
/// - Returns `None` for untracked contexts (tuple unpacking, call args, etc.).
fn resolve_variable_context(node: Node, text: &str) -> Option<String> {
    let mut current = node;
    loop {
        let Some(parent) = current.parent() else {
            // Reached root without hitting a value-position context.
            return None;
        };
        match parent.kind() {
            // Transparent wrappers — keep walking up.
            "parenthesized_expression" | "expression_list" | "unary_operator" => {
                current = parent;
            }
            // Value-position contexts with no LHS variable.
            "return_statement" | "yield" | "assert_statement" => {
                return Some(String::new());
            }
            // Assignment — extract LHS variable.
            "assignment" => {
                let left = parent.child_by_field_name("left");
                if let Some(lhs) = left
                    && lhs.kind() == "identifier"
                {
                    return Some(
                        lhs.utf8_text(text.as_bytes())
                            .ok()?
                            .to_string()
                    );
                }
                // Tuple unpacking or other non-identifier LHS — not tracked.
                return None;
            }
            // Any other parent means the binary_operator is in a context
            // we don't track (e.g., call argument, subscript index).
            _ => return None,
        }
    }
}

#[cfg(test)]
mod extract_binary_ops_tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    #[test]
    fn test_matmul_operator() {
        let code = "y = a @ b";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].variable, "y");
        assert_eq!(ops[0].left, "a");
        assert_eq!(ops[0].right, "b");
        assert_eq!(ops[0].op, BinaryOp::MatMul);
    }

    #[test]
    fn test_add_operator() {
        let code = "y = a + b";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::Add);
    }

    #[test]
    fn test_sub_operator() {
        let code = "y = a - b";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::Sub);
    }

    #[test]
    fn test_mul_operator() {
        let code = "y = a * b";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::Mul);
    }

    #[test]
    fn test_div_operator() {
        let code = "y = a / b";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::Div);
    }

    #[test]
    fn test_range_covers_binary_operator_node() {
        let code = "y = a @ b";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(
            &code[ops[0].range.start_byte..ops[0].range.end_byte],
            "a @ b"
        );
    }

    #[test]
    fn test_skip_tuple_unpacking_lhs() {
        let code = "x, y = a + b";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn test_skip_chained_addition() {
        let code = "y = a + b + c";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        // The outermost binary_operator has left = (a + b), which is not an identifier
        assert!(ops.is_empty());
    }

    #[test]
    fn test_skip_non_binary_rhs() {
        let code = "y = foo(a, b)";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn test_skip_parenthesized_operands() {
        let code = "y = (a + b) @ c";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        // left operand (a + b) is a parenthesized expression, not an identifier
        assert!(ops.is_empty());
    }

    #[test]
    fn test_multiple_binary_ops() {
        let code = "x = a @ b\ny = c + d";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].op, BinaryOp::MatMul);
        assert_eq!(ops[1].op, BinaryOp::Add);
    }

    #[test]
    fn test_skip_unsupported_operator() {
        let code = "y = a // b";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        // floor division is not in the supported set
        assert!(ops.is_empty());
    }

    #[test]
    fn test_return_matmul() {
        let code = "return x @ y";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::MatMul);
        assert_eq!(ops[0].left, "x");
        assert_eq!(ops[0].right, "y");
        assert_eq!(ops[0].variable, "");
        assert_eq!(&code[ops[0].range.start_byte..ops[0].range.end_byte], "x @ y");
    }

    #[test]
    fn test_return_add() {
        let code = "return a + b";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::Add);
        assert_eq!(ops[0].variable, "");
    }

    #[test]
    fn test_mixed_assignment_and_return() {
        let code = "z = x @ y\nreturn z + w";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 2);
        // Source order: assignment first, then return
        assert_eq!(ops[0].variable, "z");
        assert_eq!(ops[0].op, BinaryOp::MatMul);
        assert_eq!(ops[1].variable, "");
        assert_eq!(ops[1].op, BinaryOp::Add);
    }

    #[test]
    fn test_return_no_binary_op_function_call() {
        let code = "return f(x)";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn test_return_bare_identifier() {
        let code = "return x";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn test_return_parenthesized_matmul() {
        let code = "return (x @ y)";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::MatMul);
        assert_eq!(ops[0].left, "x");
        assert_eq!(ops[0].right, "y");
        assert_eq!(ops[0].variable, "");
        assert_eq!(&code[ops[0].range.start_byte..ops[0].range.end_byte], "x @ y");
    }

    #[test]
    fn test_return_double_parenthesized_matmul() {
        let code = "return ((x @ y))";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::MatMul);
        assert_eq!(ops[0].left, "x");
        assert_eq!(ops[0].right, "y");
        assert_eq!(&code[ops[0].range.start_byte..ops[0].range.end_byte], "x @ y");
    }

    #[test]
    fn test_return_expression_list_two_ops() {
        let code = "return x @ y, a + b";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].op, BinaryOp::MatMul);
        assert_eq!(ops[0].left, "x");
        assert_eq!(ops[0].right, "y");
        assert_eq!(ops[0].variable, "");
        assert_eq!(&code[ops[0].range.start_byte..ops[0].range.end_byte], "x @ y");
        assert_eq!(ops[1].op, BinaryOp::Add);
        assert_eq!(ops[1].left, "a");
        assert_eq!(ops[1].right, "b");
        assert_eq!(ops[1].variable, "");
        assert_eq!(&code[ops[1].range.start_byte..ops[1].range.end_byte], "a + b");
    }

    #[test]
    fn test_return_unary_negated_matmul() {
        let code = "return -(x @ y)";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::MatMul);
        assert_eq!(ops[0].left, "x");
        assert_eq!(ops[0].right, "y");
        assert_eq!(&code[ops[0].range.start_byte..ops[0].range.end_byte], "x @ y");
    }

    #[test]
    fn test_yield_matmul() {
        let code = "yield x @ y";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::MatMul);
        assert_eq!(ops[0].left, "x");
        assert_eq!(ops[0].right, "y");
        assert_eq!(ops[0].variable, "");
    }

    #[test]
    fn test_assert_matmul() {
        let code = "assert x @ y";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::MatMul);
        assert_eq!(ops[0].left, "x");
        assert_eq!(ops[0].right, "y");
        assert_eq!(ops[0].variable, "");
    }

    #[test]
    fn test_triple_parenthesized_return() {
        let code = "return (((x @ y)))";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::MatMul);
        assert_eq!(ops[0].left, "x");
        assert_eq!(ops[0].right, "y");
        assert_eq!(&code[ops[0].range.start_byte..ops[0].range.end_byte], "x @ y");
    }

    #[test]
    fn test_return_tuple_with_parenthesized_element() {
        let code = "return x @ y, (a + b)";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].op, BinaryOp::MatMul);
        assert_eq!(ops[0].left, "x");
        assert_eq!(ops[0].right, "y");
        assert_eq!(&code[ops[0].range.start_byte..ops[0].range.end_byte], "x @ y");
        assert_eq!(ops[1].op, BinaryOp::Add);
        assert_eq!(ops[1].left, "a");
        assert_eq!(ops[1].right, "b");
        assert_eq!(&code[ops[1].range.start_byte..ops[1].range.end_byte], "a + b");
    }

    #[test]
    fn test_yield_from_matmul() {
        // tree-sitter-python parses "yield from x @ y" identically to "yield x @ y"
        let code = "yield from x @ y";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op, BinaryOp::MatMul);
        assert_eq!(ops[0].left, "x");
        assert_eq!(ops[0].right, "y");
        assert_eq!(ops[0].variable, "");
    }

    #[test]
    fn test_binary_op_in_call_arg_not_extracted() {
        let code = "print(x @ y)";
        let tree = parse(code);
        let ops = extract_binary_ops(tree.root_node(), code).unwrap();
        assert!(ops.is_empty());
    }
}

#[cfg(test)]
mod extract_method_calls_tests {
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
    fn test_simple_method_call() {
        let code = "y = x.reshape(3, 4)";
        let tree = parse(code);
        let calls = extract_method_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].variable, "y");
        assert_eq!(calls[0].receiver, "x");
        assert_eq!(calls[0].method, "reshape");
        assert_eq!(args_text(code, &calls[0].args_node_range), "(3, 4)");
    }

    #[test]
    fn test_method_call_no_args() {
        let code = "y = x.flatten()";
        let tree = parse(code);
        let calls = extract_method_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].receiver, "x");
        assert_eq!(calls[0].method, "flatten");
        assert_eq!(args_text(code, &calls[0].args_node_range), "()");
    }

    #[test]
    fn test_method_call_with_kwargs() {
        let code = "y = x.sum(axis=0, keepdims=True)";
        let tree = parse(code);
        let calls = extract_method_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].receiver, "x");
        assert_eq!(calls[0].method, "sum");
        assert_eq!(
            args_text(code, &calls[0].args_node_range),
            "(axis=0, keepdims=True)"
        );
    }

    #[test]
    fn test_skips_deep_attribute_call() {
        let code = "x = eqx.nn.Linear(3, 5)";
        let tree = parse(code);
        let calls = extract_method_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_captures_module_style_too() {
        let code = "y = np.sum(x)";
        let tree = parse(code);
        let calls = extract_method_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].receiver, "np");
        assert_eq!(calls[0].method, "sum");
    }

    #[test]
    fn test_skips_chained_method_call_outer() {
        let code = "y = x.reshape(3, 4).sum()";
        let tree = parse(code);
        let calls = extract_method_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_skips_tuple_unpack() {
        let code = "a, b = x.split(2)";
        let tree = parse(code);
        let calls = extract_method_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_skips_bare_method_call() {
        let code = "x.fill(0)";
        let tree = parse(code);
        let calls = extract_method_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_multiple_method_calls() {
        let code = "y = x.flatten()\nz = y.sum(axis=0)";
        let tree = parse(code);
        let calls = extract_method_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].receiver, "x");
        assert_eq!(calls[0].method, "flatten");
        assert_eq!(calls[1].receiver, "y");
        assert_eq!(calls[1].method, "sum");
    }

    #[test]
    fn test_method_call_inside_function() {
        let code = r#"
def forward(x):
    y = x.reshape(2, 3)
    z = y.mean(axis=1)
"#;
        let tree = parse(code);
        let calls = extract_method_calls(tree.root_node(), code).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].method, "reshape");
        assert_eq!(calls[1].method, "mean");
    }
}



