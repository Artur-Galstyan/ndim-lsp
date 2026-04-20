use std::path::PathBuf;

pub fn resolve_module(dotted_path: &str, roots: &[PathBuf]) -> Option<(PathBuf, Vec<String>)> {
    for root in roots {
        let py_test = dotted_path.replace(".", "/") + ".py";
        let path = root.join(&py_test);
        if path.exists() {
            return Some((path, Vec::new()));
        }
        let Some((prefix, to_strip)) = py_test.rsplit_once('/') else {
            continue;
        };
        let init_test = format!("{}/__init__.py", prefix);
        let path = root.join(&init_test);
        if path.exists() {
            return Some((path, vec![to_strip.to_string()]));
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

        let result = resolve_module("foo", &[root.clone()]);
        assert_eq!(result, Some((root.join("foo.py"), Vec::new())));
    }

    #[test]
    fn test_resolves_nested_py_file() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("a/b.py"));

        let result = resolve_module("a.b", &[root.clone()]);
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

        let result = resolve_module("mypkg", &[root.clone()]);
        assert_eq!(result, Some((root.join("mypkg/__init__.py"), Vec::new())));
    }

    #[test]
    fn test_resolves_nested_package_with_init() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("torch/nn/__init__.py"));

        let result = resolve_module("torch.nn", &[root.clone()]);
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

        let result = resolve_module("torch.nn.Linear", &[root.clone()]);
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

        let result = resolve_module("a.b.c", &[root.clone()]);
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

        let result = resolve_module("foo", &[root.clone()]);
        assert_eq!(result, Some((root.join("foo.py"), Vec::new())));
    }

    #[test]
    fn test_longest_match_wins() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        write_file(&root.join("torch/__init__.py"));
        write_file(&root.join("torch/nn/__init__.py"));

        let result = resolve_module("torch.nn", &[root.clone()]);
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

        let result = resolve_module("torch.nn.Linear", &[root.clone()]);
        assert_eq!(
            result,
            Some((
                root.join("torch/nn/__init__.py"),
                vec!["Linear".to_string()]
            ))
        );
    }
}
