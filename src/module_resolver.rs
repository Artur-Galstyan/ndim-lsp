use std::path::PathBuf;

use tree_sitter::{Node, Query, QueryCursor, StreamingIterator};

#[derive(Debug, PartialEq, Clone)]
pub enum ReExport {
    Absolute {
        path: String,
        name: String,
    },
    Relative {
        dots: usize,
        path: Option<String>,
        name: String,
    },
}

pub fn import_finder(node: Node, attribute_name: &str, text: &str) -> Option<ReExport> {
    let language = tree_sitter_python::LANGUAGE.into();

    let query_str = r#"
        (import_from_statement
          module_name: [
            (dotted_name) @absolute_module
            (relative_import) @relative_module
          ]
          name: [
            (dotted_name) @name
            (aliased_import) @aliased_name
          ]
        )
    "#;

    let query = Query::new(&language, query_str).unwrap();
    let mut cursor = QueryCursor::new();

    let absolute_mod_idx = query.capture_index_for_name("absolute_module")?;
    let relative_mod_idx = query.capture_index_for_name("relative_module")?;
    let name_idx = query.capture_index_for_name("name")?;
    let aliased_idx = query.capture_index_for_name("aliased_name")?;

    let mut matches = cursor.matches(&query, node, text.as_bytes());
    while let Some(match_) = matches.next() {
        let mut abs_module: Option<&str> = None;
        let mut rel_module: Option<Node> = None;
        let mut names: Vec<Node> = Vec::new();
        let mut aliased_names: Vec<Node> = Vec::new();
        for capture in match_.captures {
            if capture.index == absolute_mod_idx {
                abs_module = capture.node.utf8_text(text.as_bytes()).ok();
            } else if capture.index == relative_mod_idx {
                rel_module = Some(capture.node);
            } else if capture.index == name_idx {
                names.push(capture.node);
            } else if capture.index == aliased_idx {
                aliased_names.push(capture.node);
            }
        }

        let mut name_in_source: Option<String> = None;

        for name_node in &names {
            let text_val = name_node.utf8_text(text.as_bytes()).ok()?;
            if text_val == attribute_name {
                name_in_source = Some(text_val.to_string());
                break;
            }
        }

        if name_in_source.is_none() {
            for aliased_node in &aliased_names {
                let alias_node = aliased_node.child_by_field_name("alias")?;
                let alias_text = alias_node.utf8_text(text.as_bytes()).ok()?;
                if alias_text == attribute_name {
                    let original_node = aliased_node.child_by_field_name("name")?;
                    let original_text = original_node.utf8_text(text.as_bytes()).ok()?;
                    name_in_source = Some(original_text.to_string());
                    break;
                }
            }
        }

        if let Some(name) = name_in_source {
            if let Some(abs) = abs_module {
                return Some(ReExport::Absolute {
                    path: abs.to_string(),
                    name,
                });
            } else if let Some(rel_node) = rel_module {
                let mut cursor = rel_node.walk();
                let mut dots = 0;
                let mut path: Option<String> = None;
                for child in rel_node.children(&mut cursor) {
                    match child.kind() {
                        "import_prefix" => {
                            let text_val = child.utf8_text(text.as_bytes()).ok()?;
                            dots = text_val.len();
                        }
                        "dotted_name" => {
                            path = child.utf8_text(text.as_bytes()).ok().map(|s| s.to_string());
                        }
                        _ => {}
                    }
                }
                return Some(ReExport::Relative { dots, path, name });
            }
        }
    }

    None
}

pub fn resolve_module(dotted_path: &str, roots: &[PathBuf]) -> Option<(PathBuf, Vec<String>)> {
    let paths = dotted_path.replace(".", "/");
    for root in roots {
        let mut stripped_paths = Vec::new();
        let mut split_paths: Vec<&str> = paths.split('/').collect();

        while !split_paths.is_empty() {
            let py_test = split_paths.join("/");
            let path = root.join(&py_test).with_extension("py");
            if path.exists() {
                return Some((path, stripped_paths));
            }

            let init_test = format!("{}/__init__.py", py_test);
            let path = root.join(&init_test);
            if path.exists() {
                return Some((path, stripped_paths));
            }

            let Some((_, to_strip)) = py_test.rsplit_once('/') else {
                break;
            };
            stripped_paths.insert(0, to_strip.to_string());
            split_paths.remove(split_paths.len() - 1);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_file(path: &PathBuf) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, "").unwrap();
    }

    #[test]
    fn test_resolves_top_level_py_file() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("foo.py"));

        let result = resolve_module("foo", std::slice::from_ref(&root));
        assert_eq!(result, Some((root.join("foo.py"), Vec::new())));
    }

    #[test]
    fn test_resolves_nested_py_file() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("a/b.py"));

        let result = resolve_module("a.b", std::slice::from_ref(&root));
        assert_eq!(result, Some((root.join("a/b.py"), Vec::new())));
    }

    #[test]
    fn test_returns_none_when_module_does_not_exist() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();

        let result = resolve_module("missing", &[root]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_returns_none_with_empty_roots() {
        let result = resolve_module("foo", &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_resolves_package_with_init() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("mypkg/__init__.py"));

        let result = resolve_module("mypkg", std::slice::from_ref(&root));
        assert_eq!(result, Some((root.join("mypkg/__init__.py"), Vec::new())));
    }

    #[test]
    fn test_resolves_nested_package_with_init() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("torch/nn/__init__.py"));

        let result = resolve_module("torch.nn", std::slice::from_ref(&root));
        assert_eq!(
            result,
            Some((root.join("torch/nn/__init__.py"), Vec::new()))
        );
    }

    #[test]
    fn test_strips_attribute_to_find_package() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("torch/nn/__init__.py"));

        let result = resolve_module("torch.nn.Linear", std::slice::from_ref(&root));
        assert_eq!(
            result,
            Some((
                root.join("torch/nn/__init__.py"),
                vec!["Linear".to_string()]
            ))
        );
    }

    #[test]
    fn test_strips_multiple_attributes() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("a/__init__.py"));

        let result = resolve_module("a.b.c", std::slice::from_ref(&root));
        assert_eq!(
            result,
            Some((
                root.join("a/__init__.py"),
                vec!["b".to_string(), "c".to_string()]
            ))
        );
    }

    #[test]
    fn test_py_file_wins_over_init_at_same_level() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("foo.py"));
        write_file(&root.join("foo/__init__.py"));

        let result = resolve_module("foo", std::slice::from_ref(&root));
        assert_eq!(result, Some((root.join("foo.py"), Vec::new())));
    }

    #[test]
    fn test_longest_match_wins() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("torch/__init__.py"));
        write_file(&root.join("torch/nn/__init__.py"));

        let result = resolve_module("torch.nn", std::slice::from_ref(&root));
        println!("path: {:?}", result.as_ref().unwrap());
        assert_eq!(
            result,
            Some((root.join("torch/nn/__init__.py"), Vec::new()))
        );
    }

    #[test]
    fn test_longest_match_wins_with_attribute() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("torch/__init__.py"));
        write_file(&root.join("torch/nn/__init__.py"));

        let result = resolve_module("torch.nn.Linear", std::slice::from_ref(&root));
        assert_eq!(
            result,
            Some((
                root.join("torch/nn/__init__.py"),
                vec!["Linear".to_string()]
            ))
        );
    }

    #[test]
    fn test_first_root_wins_when_module_exists_in_multiple() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        let root1 = tmp1.path().to_path_buf();
        let root2 = tmp2.path().to_path_buf();
        write_file(&root1.join("foo.py"));
        write_file(&root2.join("foo.py"));

        let result = resolve_module("foo", &[root1.clone(), root2.clone()]);
        assert_eq!(result, Some((root1.join("foo.py"), Vec::new())));
    }

    #[test]
    fn test_falls_through_to_second_root_when_first_misses() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        let root1 = tmp1.path().to_path_buf();
        let root2 = tmp2.path().to_path_buf();
        write_file(&root2.join("foo.py"));

        let result = resolve_module("foo", &[root1.clone(), root2.clone()]);
        assert_eq!(result, Some((root2.join("foo.py"), Vec::new())));
    }

    #[test]
    fn test_stripped_paths_not_leaked_between_roots() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        let root1 = tmp1.path().to_path_buf();
        let root2 = tmp2.path().to_path_buf();
        write_file(&root2.join("a/__init__.py"));

        let result = resolve_module("a.b.c", &[root1.clone(), root2.clone()]);
        assert_eq!(
            result,
            Some((
                root2.join("a/__init__.py"),
                vec!["b".to_string(), "c".to_string()]
            ))
        );
    }

    #[test]
    fn test_deeply_nested_attribute_stripping() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("a/b/__init__.py"));

        let result = resolve_module("a.b.c.d.e.f", std::slice::from_ref(&root));
        assert_eq!(
            result,
            Some((
                root.join("a/b/__init__.py"),
                vec![
                    "c".to_string(),
                    "d".to_string(),
                    "e".to_string(),
                    "f".to_string()
                ]
            ))
        );
    }

    #[test]
    fn test_returns_none_for_unresolvable_path() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("other.py"));

        let result = resolve_module("missing.stuff", std::slice::from_ref(&root));
        assert_eq!(result, None);
    }
}

#[cfg(test)]
mod import_finder_tests {
    use super::*;
    use tree_sitter::Parser;

    fn get_ast(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    #[test]
    fn test_absolute_import() {
        let code = "from torch.nn.modules.linear import Linear";
        let tree = get_ast(code);
        let result = import_finder(tree.root_node(), "Linear", code);
        assert_eq!(
            result,
            Some(ReExport::Absolute {
                path: "torch.nn.modules.linear".to_string(),
                name: "Linear".to_string(),
            })
        );
    }

    #[test]
    fn test_relative_import_single_dot() {
        let code = "from .modules.linear import Linear";
        let tree = get_ast(code);
        let result = import_finder(tree.root_node(), "Linear", code);
        assert_eq!(
            result,
            Some(ReExport::Relative {
                dots: 1,
                path: Some("modules.linear".to_string()),
                name: "Linear".to_string(),
            })
        );
    }

    #[test]
    fn test_relative_import_many_dots() {
        let code = "from ...modules import Linear";
        let tree = get_ast(code);
        let result = import_finder(tree.root_node(), "Linear", code);
        assert_eq!(
            result,
            Some(ReExport::Relative {
                dots: 3,
                path: Some("modules".to_string()),
                name: "Linear".to_string(),
            })
        );
    }

    #[test]
    fn test_relative_import_no_path() {
        let code = "from . import Linear";
        let tree = get_ast(code);
        let result = import_finder(tree.root_node(), "Linear", code);
        assert_eq!(
            result,
            Some(ReExport::Relative {
                dots: 1,
                path: None,
                name: "Linear".to_string(),
            })
        );
    }

    #[test]
    fn test_aliased_relative_import() {
        let code = "from .modules.linear import Linear as L";
        let tree = get_ast(code);
        let result = import_finder(tree.root_node(), "L", code);
        assert_eq!(
            result,
            Some(ReExport::Relative {
                dots: 1,
                path: Some("modules.linear".to_string()),
                name: "Linear".to_string(),
            })
        );
    }

    #[test]
    fn test_aliased_absolute_import() {
        let code = "from torch.nn import Linear as _Linear";
        let tree = get_ast(code);
        let result = import_finder(tree.root_node(), "_Linear", code);
        assert_eq!(
            result,
            Some(ReExport::Absolute {
                path: "torch.nn".to_string(),
                name: "Linear".to_string(),
            })
        );
    }

    #[test]
    fn test_multi_import_finds_each() {
        let code = "from .modules.linear import Linear, Conv2d, ReLU";
        let tree = get_ast(code);

        let result = import_finder(tree.root_node(), "Linear", code);
        assert_eq!(
            result,
            Some(ReExport::Relative {
                dots: 1,
                path: Some("modules.linear".to_string()),
                name: "Linear".to_string(),
            })
        );

        let result = import_finder(tree.root_node(), "Conv2d", code);
        assert_eq!(
            result,
            Some(ReExport::Relative {
                dots: 1,
                path: Some("modules.linear".to_string()),
                name: "Conv2d".to_string(),
            })
        );

        let result = import_finder(tree.root_node(), "ReLU", code);
        assert_eq!(
            result,
            Some(ReExport::Relative {
                dots: 1,
                path: Some("modules.linear".to_string()),
                name: "ReLU".to_string(),
            })
        );
    }

    #[test]
    fn test_returns_none_when_name_not_imported() {
        let code = "from .modules.linear import Linear";
        let tree = get_ast(code);
        let result = import_finder(tree.root_node(), "Conv2d", code);
        assert_eq!(result, None);
    }

    #[test]
    fn test_returns_none_when_no_imports() {
        let code = "x = 5";
        let tree = get_ast(code);
        let result = import_finder(tree.root_node(), "Linear", code);
        assert_eq!(result, None);
    }

    #[test]
    fn test_finds_correct_import_among_many() {
        let code = r#"
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
from .activations import ReLU
"#;
        let tree = get_ast(code);

        let result = import_finder(tree.root_node(), "Conv2d", code);
        assert_eq!(
            result,
            Some(ReExport::Absolute {
                path: "torch.nn.modules.conv".to_string(),
                name: "Conv2d".to_string(),
            })
        );

        let result = import_finder(tree.root_node(), "ReLU", code);
        assert_eq!(
            result,
            Some(ReExport::Relative {
                dots: 1,
                path: Some("activations".to_string()),
                name: "ReLU".to_string(),
            })
        );
    }
}
