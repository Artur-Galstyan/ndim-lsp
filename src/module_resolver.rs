use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use tree_sitter::{Node, Parser, Query, QueryCursor, Range, StreamingIterator};

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

fn import_finder(node: Node, attribute_name: &str, text: &str) -> Option<ReExport> {
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

fn resolve_module(dotted_path: &str, roots: &[PathBuf]) -> Option<(PathBuf, Vec<String>)> {
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

fn find_definition<'a>(node: Node<'a>, name: &str, text: &str) -> Option<Node<'a>> {
    let query = r#"
        (module (class_definition name: (identifier) @def_name) @def_node)
        (module (function_definition name: (identifier) @def_name) @def_node)
        (module (expression_statement (assignment left: (identifier) @def_name) @def_node))
    "#;

    let language = tree_sitter_python::LANGUAGE.into();
    let mut definition: Option<Node<'a>> = None;
    let query = Query::new(&language, query).unwrap();
    let mut cursor = QueryCursor::new();

    let def_idx = query.capture_index_for_name("def_name")?;
    let def_node_idx = query.capture_index_for_name("def_node")?;

    let mut matches = cursor.matches(&query, node, text.as_bytes());
    while let Some(match_) = matches.next() {
        let mut name_match = false;
        let mut def_node: Option<Node<'a>> = None;

        for capture in match_.captures {
            if capture.index == def_idx {
                let text_val = capture.node.utf8_text(text.as_bytes()).ok()?;
                if text_val == name {
                    name_match = true;
                }
            } else if capture.index == def_node_idx {
                def_node = Some(capture.node);
            }
        }

        if name_match {
            return def_node;
        }
    }

    None
}

fn reexport_to_absolute_path(
    current_file: &Path,
    package_roots: &[PathBuf],
    reexport: &ReExport,
) -> Option<String> {
    match reexport {
        ReExport::Absolute { path, name } => {
            return Some(format!("{}.{}", path, name));
        }
        ReExport::Relative { dots, path, name } => {
            for package in package_roots {
                if let Ok(rel) = current_file.strip_prefix(package) {
                    let parent = rel.parent()?.to_string_lossy().into_owned();
                    let mut to_dotted = parent.replace('/', ".");
                    let mut dot_counter = *dots;
                    while dot_counter > 1 {
                        let mut parts = to_dotted.split('.').collect::<Vec<_>>();
                        parts.pop();
                        to_dotted = parts.join(".");
                        dot_counter -= 1;
                    }
                    let mut result = to_dotted;
                    if let Some(p) = path {
                        result = format!("{}.{}", result, p);
                    }
                    return Some(format!("{}.{}", result, name));
                }
            }
        }
    };
    None
}

pub fn resolve_definition(
    dotted_path: &str,
    roots: &[PathBuf],
    visited: &mut HashSet<PathBuf>,
    depth: usize,
) -> Option<(PathBuf, Range)> {
    if depth == 0 {
        return None;
    }
    let (file_path, leftover_args) = resolve_module(dotted_path, roots)?;

    if visited.contains(&file_path) {
        return None;
    }
    visited.insert(file_path.clone());
    let last_arg = leftover_args.last()?;
    let mut parser = Parser::new();
    let language = tree_sitter_python::LANGUAGE;
    parser.set_language(&language.into()).unwrap();

    let file_content = fs::read_to_string(&file_path).ok()?;
    let tree = parser.parse(&file_content, None)?;

    let def_found = find_definition(tree.root_node(), last_arg, &file_content);

    match def_found {
        Some(def) => Some((file_path, def.range())),
        None => {
            let import = import_finder(tree.root_node(), last_arg, &file_content)?;
            let abs_path = reexport_to_absolute_path(&file_path, roots, &import)?;
            resolve_definition(&abs_path, roots, visited, depth - 1)
        }
    }
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

#[cfg(test)]
mod find_definition_tests {
    use super::*;
    use tree_sitter::Parser;

    fn get_ast(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    #[test]
    fn test_finds_top_level_class() {
        let code = "class Linear:\n    pass";
        let tree = get_ast(code);
        let result = find_definition(tree.root_node(), "Linear", code);
        assert!(result.is_some());
        assert_eq!(result.unwrap().kind(), "class_definition");
    }

    #[test]
    fn test_finds_top_level_function() {
        let code = "def my_func():\n    pass";
        let tree = get_ast(code);
        let result = find_definition(tree.root_node(), "my_func", code);
        assert!(result.is_some());
        assert_eq!(result.unwrap().kind(), "function_definition");
    }

    #[test]
    fn test_finds_top_level_assignment() {
        let code = "Linear = _Linear";
        let tree = get_ast(code);
        let result = find_definition(tree.root_node(), "Linear", code);
        assert!(result.is_some());
        assert_eq!(result.unwrap().kind(), "assignment");
    }

    #[test]
    fn test_returns_none_when_not_defined() {
        let code = "class Foo:\n    pass";
        let tree = get_ast(code);
        let result = find_definition(tree.root_node(), "Bar", code);
        assert!(result.is_none());
    }

    #[test]
    fn test_returns_none_in_empty_file() {
        let code = "";
        let tree = get_ast(code);
        let result = find_definition(tree.root_node(), "anything", code);
        assert!(result.is_none());
    }

    #[test]
    fn test_ignores_nested_class_inside_function() {
        let code = r#"
def wrapper():
    class Hidden:
        pass
"#;
        let tree = get_ast(code);
        let result = find_definition(tree.root_node(), "Hidden", code);
        assert!(result.is_none());
    }

    #[test]
    fn test_ignores_nested_function_inside_function() {
        let code = r#"
def outer():
    def inner():
        pass
"#;
        let tree = get_ast(code);
        let result = find_definition(tree.root_node(), "inner", code);
        assert!(result.is_none());
    }

    #[test]
    fn test_ignores_method_inside_class() {
        let code = r#"
class Foo:
    def method(self):
        pass
"#;
        let tree = get_ast(code);
        let result = find_definition(tree.root_node(), "method", code);
        assert!(result.is_none());
    }

    #[test]
    fn test_ignores_nested_assignment_inside_function() {
        let code = r#"
def wrapper():
    hidden = 5
"#;
        let tree = get_ast(code);
        let result = find_definition(tree.root_node(), "hidden", code);
        assert!(result.is_none());
    }

    #[test]
    fn test_finds_among_many_definitions() {
        let code = r#"
class Foo:
    pass

def bar():
    pass

Baz = something
"#;
        let tree = get_ast(code);

        assert!(find_definition(tree.root_node(), "Foo", code).is_some());
        assert!(find_definition(tree.root_node(), "bar", code).is_some());
        assert!(find_definition(tree.root_node(), "Baz", code).is_some());
        assert!(find_definition(tree.root_node(), "Missing", code).is_none());
    }

    #[test]
    fn test_top_level_wins_over_nested_same_name() {
        let code = r#"
class Linear:
    pass

def wrapper():
    class Linear:
        pass
"#;
        let tree = get_ast(code);
        let result = find_definition(tree.root_node(), "Linear", code);
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.kind(), "class_definition");
        assert_eq!(node.start_position().row, 1);
    }
}

#[cfg(test)]
mod reexport_to_absolute_path_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_absolute_reexport_passthrough() {
        let roots = vec![PathBuf::from("/site-packages")];
        let current = Path::new("/site-packages/torch/nn/__init__.py");
        let reexport = ReExport::Absolute {
            path: "torch.nn.modules.linear".to_string(),
            name: "Linear".to_string(),
        };

        let result = reexport_to_absolute_path(current, &roots, &reexport);
        assert_eq!(result, Some("torch.nn.modules.linear.Linear".to_string()));
    }

    #[test]
    fn test_relative_single_dot_with_path() {
        let roots = vec![PathBuf::from("/site-packages")];
        let current = Path::new("/site-packages/torch/nn/__init__.py");
        let reexport = ReExport::Relative {
            dots: 1,
            path: Some("modules.linear".to_string()),
            name: "Linear".to_string(),
        };

        let result = reexport_to_absolute_path(current, &roots, &reexport);
        assert_eq!(result, Some("torch.nn.modules.linear.Linear".to_string()));
    }

    #[test]
    fn test_relative_from_dot_import_name() {
        let roots = vec![PathBuf::from("/site-packages")];
        let current = Path::new("/site-packages/torch/nn/__init__.py");
        let reexport = ReExport::Relative {
            dots: 1,
            path: None,
            name: "Linear".to_string(),
        };

        let result = reexport_to_absolute_path(current, &roots, &reexport);
        assert_eq!(result, Some("torch.nn.Linear".to_string()));
    }

    #[test]
    fn test_relative_two_dots_pops_parent() {
        let roots = vec![PathBuf::from("/site-packages")];
        let current = Path::new("/site-packages/torch/nn/modules/linear.py");
        let reexport = ReExport::Relative {
            dots: 2,
            path: Some("functional".to_string()),
            name: "relu".to_string(),
        };

        let result = reexport_to_absolute_path(current, &roots, &reexport);
        assert_eq!(result, Some("torch.nn.functional.relu".to_string()));
    }

    #[test]
    fn test_picks_correct_root_when_multiple() {
        let roots = vec![
            PathBuf::from("/project/src"),
            PathBuf::from("/site-packages"),
        ];
        let current = Path::new("/site-packages/torch/nn/__init__.py");
        let reexport = ReExport::Relative {
            dots: 1,
            path: Some("modules".to_string()),
            name: "Linear".to_string(),
        };

        let result = reexport_to_absolute_path(current, &roots, &reexport);
        assert_eq!(result, Some("torch.nn.modules.Linear".to_string()));
    }

    #[test]
    fn test_works_with_project_root() {
        let roots = vec![PathBuf::from("/project/src")];
        let current = Path::new("/project/src/mypkg/utils/__init__.py");
        let reexport = ReExport::Relative {
            dots: 1,
            path: Some("helpers".to_string()),
            name: "foo".to_string(),
        };

        let result = reexport_to_absolute_path(current, &roots, &reexport);
        assert_eq!(result, Some("mypkg.utils.helpers.foo".to_string()));
    }

    #[test]
    fn test_returns_none_when_file_not_under_any_root() {
        let roots = vec![PathBuf::from("/site-packages")];
        let current = Path::new("/somewhere/else/foo.py");
        let reexport = ReExport::Relative {
            dots: 1,
            path: Some("bar".to_string()),
            name: "baz".to_string(),
        };

        let result = reexport_to_absolute_path(current, &roots, &reexport);
        assert_eq!(result, None);
    }

    #[test]
    fn test_regular_py_file_not_init() {
        let roots = vec![PathBuf::from("/site-packages")];
        let current = Path::new("/site-packages/torch/nn/modules/linear.py");
        let reexport = ReExport::Relative {
            dots: 1,
            path: None,
            name: "Linear".to_string(),
        };

        let result = reexport_to_absolute_path(current, &roots, &reexport);
        assert_eq!(result, Some("torch.nn.modules.Linear".to_string()));
    }

    #[test]
    fn test_three_dots_pops_two_levels() {
        let roots = vec![PathBuf::from("/site-packages")];
        let current = Path::new("/site-packages/a/b/c/d/__init__.py");
        let reexport = ReExport::Relative {
            dots: 3,
            path: Some("x".to_string()),
            name: "y".to_string(),
        };

        let result = reexport_to_absolute_path(current, &roots, &reexport);
        assert_eq!(result, Some("a.b.x.y".to_string()));
    }

    #[test]
    fn test_absolute_variant_ignores_current_file() {
        let roots = vec![PathBuf::from("/site-packages")];
        let current = Path::new("/anywhere/random.py");
        let reexport = ReExport::Absolute {
            path: "jax.numpy".to_string(),
            name: "zeros".to_string(),
        };

        let result = reexport_to_absolute_path(current, &roots, &reexport);
        assert_eq!(result, Some("jax.numpy.zeros".to_string()));
    }
}

#[cfg(test)]
mod resolve_definition_tests {
    use super::*;
    use std::collections::HashSet;
    use std::fs;
    use tempfile::TempDir;

    fn write(path: &PathBuf, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    #[test]
    fn test_finds_class_defined_in_file() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write(&root.join("foo.py"), "class Linear:\n    pass");

        let mut visited = HashSet::new();
        let result =
            resolve_definition("foo.Linear", std::slice::from_ref(&root), &mut visited, 10);

        assert!(result.is_some());
        let (path, _) = result.unwrap();
        assert_eq!(path, root.join("foo.py"));
    }

    #[test]
    fn test_finds_function_defined_in_file() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write(&root.join("utils.py"), "def helper():\n    pass");

        let mut visited = HashSet::new();
        let result = resolve_definition(
            "utils.helper",
            std::slice::from_ref(&root),
            &mut visited,
            10,
        );

        assert!(result.is_some());
        let (path, _) = result.unwrap();
        assert_eq!(path, root.join("utils.py"));
    }

    #[test]
    fn test_follows_absolute_reexport() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write(
            &root.join("mypkg/__init__.py"),
            "from mypkg.impl import Linear",
        );
        write(&root.join("mypkg/impl.py"), "class Linear:\n    pass");

        let mut visited = HashSet::new();
        let result = resolve_definition(
            "mypkg.Linear",
            std::slice::from_ref(&root),
            &mut visited,
            10,
        );

        assert!(result.is_some());
        let (path, _) = result.unwrap();
        assert_eq!(path, root.join("mypkg/impl.py"));
    }

    #[test]
    fn test_follows_relative_reexport() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write(&root.join("mypkg/__init__.py"), "from .impl import Linear");
        write(&root.join("mypkg/impl.py"), "class Linear:\n    pass");

        let mut visited = HashSet::new();
        let result = resolve_definition(
            "mypkg.Linear",
            std::slice::from_ref(&root),
            &mut visited,
            10,
        );

        assert!(result.is_some());
        let (path, _) = result.unwrap();
        assert_eq!(path, root.join("mypkg/impl.py"));
    }

    #[test]
    fn test_returns_none_when_name_not_found() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write(&root.join("foo.py"), "class Other:\n    pass");

        let mut visited = HashSet::new();
        let result =
            resolve_definition("foo.Missing", std::slice::from_ref(&root), &mut visited, 10);

        assert!(result.is_none());
    }
    #[test]
    fn test_follows_multi_hop_chain() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write(
            &root.join("mypkg/__init__.py"),
            "from .middle import Linear",
        );
        write(&root.join("mypkg/middle.py"), "from .impl import Linear");
        write(&root.join("mypkg/impl.py"), "class Linear:\n    pass");

        let mut visited = HashSet::new();
        let result = resolve_definition(
            "mypkg.Linear",
            std::slice::from_ref(&root),
            &mut visited,
            10,
        );

        assert!(result.is_some());
        let (path, _) = result.unwrap();
        assert_eq!(path, root.join("mypkg/impl.py"));
    }

    #[test]
    fn test_detects_cycle() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write(&root.join("a.py"), "from b import X");
        write(&root.join("b.py"), "from a import X");

        let mut visited = HashSet::new();
        let result = resolve_definition("a.X", std::slice::from_ref(&root), &mut visited, 10);

        assert!(result.is_none());
    }

    #[test]
    fn test_respects_depth_limit() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write(&root.join("a.py"), "from b import X");
        write(&root.join("b.py"), "from c import X");
        write(&root.join("c.py"), "from d import X");
        write(&root.join("d.py"), "class X:\n    pass");

        let mut visited = HashSet::new();
        let result = resolve_definition("a.X", std::slice::from_ref(&root), &mut visited, 2);

        assert!(result.is_none());
    }

    #[test]
    fn test_reaches_def_within_depth() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write(&root.join("a.py"), "from b import X");
        write(&root.join("b.py"), "class X:\n    pass");

        let mut visited = HashSet::new();
        let result = resolve_definition("a.X", std::slice::from_ref(&root), &mut visited, 2);

        assert!(result.is_some());
        let (path, _) = result.unwrap();
        assert_eq!(path, root.join("b.py"));
    }

    #[test]
    fn test_aliased_reexport_chain() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write(
            &root.join("mypkg/__init__.py"),
            "from .impl import _Linear as Linear",
        );
        write(&root.join("mypkg/impl.py"), "class _Linear:\n    pass");

        let mut visited = HashSet::new();
        let result = resolve_definition(
            "mypkg.Linear",
            std::slice::from_ref(&root),
            &mut visited,
            10,
        );

        assert!(result.is_some());
        let (path, _) = result.unwrap();
        assert_eq!(path, root.join("mypkg/impl.py"));
    }
}
