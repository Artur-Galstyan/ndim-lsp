//! What an imported name means at a place of the file. Python binds an import
//! in the scope of the statement, so the import map of a place is the module
//! map, changed by the scopes around that place. A scope that changes nothing
//! has no region: a file without such a scope pays one length check per lookup.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::Arc;

use tree_sitter::Node;

use crate::analysis::LIBRARIES;
use crate::python_ast::{build_import_map, visit_imports};
use crate::types::ImportPath;

pub type ImportMap = HashMap<String, ImportPath>;

/// The name of a map entry for a name that a scope binds to a value. The entry
/// stays in the map because a target with no entry reads as a library path:
/// `torch.zeros` for a parameter `torch`. No path or known function has it.
const SHADOWED: &str = "<local>";

/// Whether `name` is an import in `map`.
pub fn is_imported(map: &ImportMap, name: &str) -> bool {
    map.get(name).is_some_and(|import| import.name != SHADOWED)
}

/// A byte range with its own import map. `None` is the module map.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportRegion {
    pub(crate) start: usize,
    pub(crate) end: usize,
    /// The distance in the list back to the region around this one. 0 is none.
    /// It stays valid when the lists of two units are joined.
    up: usize,
    map: Option<Arc<ImportMap>>,
}

/// The imports of a file by place.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImportScopes {
    module: ImportMap,
    file: ImportMap,
    /// In order of start. A region after another is nested in it or after it.
    regions: Vec<ImportRegion>,
}

impl ImportScopes {
    pub fn new(root: Node, text: &str) -> Result<Self, String> {
        let (file, module) = import_maps(root, text)?;
        let regions = import_regions(root, text, &module, &file)?;
        Ok(Self::from_parts(module, file, regions))
    }

    pub(crate) fn from_parts(module: ImportMap, file: ImportMap, regions: Vec<ImportRegion>) -> Self {
        Self { module, file, regions }
    }

    /// The imports that a name at `byte` can read.
    pub fn at(&self, byte: usize) -> &ImportMap {
        let last = self.regions.partition_point(|region| region.start <= byte);
        let Some(mut i) = last.checked_sub(1) else {
            return &self.module;
        };
        loop {
            let region = &self.regions[i];
            if byte < region.end {
                return region.map.as_deref().unwrap_or(&self.module);
            }
            if region.up == 0 {
                return &self.module;
            }
            i -= region.up;
        }
    }

    /// Every import of the file, in any scope: what the file uses, not what a
    /// name means.
    pub fn file(&self) -> &ImportMap {
        &self.file
    }
}

/// Every import under `node`, and the imports that bind at module level. A
/// `global` statement makes an import in a function one of these.
pub fn import_maps(node: Node, text: &str) -> Result<(ImportMap, ImportMap), String> {
    let (mut file, mut module) = (ImportMap::new(), ImportMap::new());
    visit_imports(node, text, |at, alias, path| {
        if binds_module_name(at, text, &alias) {
            module.insert(alias.clone(), path.clone());
        }
        file.insert(alias, path);
    })?;
    Ok((file, module))
}

fn binds_module_name(at: Node, text: &str, alias: &str) -> bool {
    let mut up = at.parent();
    while let Some(owner) = up {
        if matches!(owner.kind(), "function_definition" | "class_definition") {
            return declares_global(owner, text, alias);
        }
        up = owner.parent();
    }
    true
}

fn declares_global(owner: Node, text: &str, alias: &str) -> bool {
    let mut stack: Vec<Node> = owner.child_by_field_name("body").into_iter().collect();
    while let Some(n) = stack.pop() {
        match n.kind() {
            "global_statement" => {
                let mut cursor = n.walk();
                if n.named_children(&mut cursor).any(|name| name.utf8_text(text.as_bytes()) == Ok(alias)) {
                    return true;
                }
            }
            "function_definition" | "class_definition" => {}
            _ => {
                let mut cursor = n.walk();
                stack.extend(n.named_children(&mut cursor));
            }
        }
    }
    false
}

#[derive(Clone, Copy, PartialEq)]
enum ScopeKind {
    Function,
    Class,
    Comprehension,
}

struct Scope {
    kind: ScopeKind,
    start: usize,
    end: usize,
    parent: usize,
    imports: Vec<(String, ImportPath)>,
    bound: Vec<String>,
    globals: Vec<String>,
}

impl Scope {
    fn new(kind: ScopeKind, range: Node, parent: usize) -> Self {
        Self {
            kind,
            start: range.start_byte(),
            end: range.end_byte(),
            parent,
            imports: Vec::new(),
            bound: Vec::new(),
            globals: Vec::new(),
        }
    }
}

fn identifiers<'t>(node: Node<'t>, out: &mut Vec<Node<'t>>) {
    if node.kind() == "identifier" {
        out.push(node);
        return;
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        identifiers(child, out);
    }
}

/// The names of a parameter list, and the default values, which are read in
/// the scope around the function.
fn parameters<'t>(list: Node<'t>, names: &mut Vec<Node<'t>>, defaults: &mut Vec<Node<'t>>) {
    let mut cursor = list.walk();
    for param in list.named_children(&mut cursor) {
        let name = match param.kind() {
            "default_parameter" | "typed_default_parameter" => {
                defaults.extend(param.child_by_field_name("value"));
                param.child_by_field_name("name")
            }
            "typed_parameter" => param.named_child(0),
            _ => Some(param),
        };
        if let Some(name) = name {
            identifiers(name, names);
        }
    }
}

/// Whether an expression can hold a scope or a walrus. Most expressions do not,
/// and this check is much faster than a walk of their nodes.
fn may_bind(expression: &str) -> bool {
    ["for", "lambda", ":="].iter().any(|word| expression.contains(word))
}

/// The regions of the scopes under `node` that change what an import name
/// means. `module` is the module map, and `file` holds every import of the
/// file. The result is the same for a unit as for the whole tree.
pub fn import_regions(
    node: Node,
    text: &str,
    module: &ImportMap,
    file: &ImportMap,
) -> Result<Vec<ImportRegion>, String> {
    let tracked = |name: &str| file.contains_key(name) || LIBRARIES.contains(&name);
    let source = text.as_bytes();
    let bind = |scopes: &mut Vec<Scope>, name: Node, at: usize| {
        if at != 0
            && let Ok(name) = name.utf8_text(source)
            && tracked(name)
        {
            scopes[at].bound.push(name.to_string());
        }
    };
    let mut scopes = vec![Scope::new(ScopeKind::Function, node, 0)];
    let mut stack = vec![(node, 0, false)];
    // `target`: the node is in the place of a name that a statement binds.
    // `Node::parent` is too slow for each name of a file, so the walk carries
    // the place down from the statement.
    while let Some((n, at, target)) = stack.pop() {
        let mut cursor = n.walk();
        match n.kind() {
            "function_definition" | "lambda" | "class_definition" => {
                if let Some(name) = n.child_by_field_name("name") {
                    bind(&mut scopes, name, at);
                }
                let Some(body) = n.child_by_field_name("body") else {
                    continue;
                };
                let (mut names, mut defaults) = (Vec::new(), Vec::new());
                if let Some(list) = n.child_by_field_name("parameters") {
                    parameters(list, &mut names, &mut defaults);
                }
                stack.extend(n.child_by_field_name("superclasses").map(|bases| (bases, at, false)));
                stack.extend(defaults.into_iter().map(|value| (value, at, false)));
                let kind = if n.kind() == "class_definition" {
                    ScopeKind::Class
                } else {
                    ScopeKind::Function
                };
                let inner = scopes.len();
                scopes.push(Scope::new(kind, body, at));
                for name in names {
                    bind(&mut scopes, name, inner);
                }
                stack.push((body, inner, false));
            }
            "list_comprehension" | "set_comprehension" | "dictionary_comprehension"
            | "generator_expression" => {
                let inner = scopes.len();
                scopes.push(Scope::new(ScopeKind::Comprehension, n, at));
                stack.extend(n.named_children(&mut cursor).map(|child| (child, inner, false)));
            }
            "import_statement" | "import_from_statement" => {
                if at != 0 {
                    scopes[at].imports.extend(build_import_map(n, text)?);
                }
            }
            "global_statement" => {
                let mut names = Vec::new();
                identifiers(n, &mut names);
                let names = names.iter().filter_map(|name| name.utf8_text(source).ok());
                scopes[at].globals.extend(names.map(str::to_string));
            }
            "future_import_statement" | "string" | "comment" | "type" => {}
            "call" | "binary_operator" if !may_bind(&text[n.byte_range()]) => {}
            "identifier" => {
                if target {
                    bind(&mut scopes, n, at);
                }
            }
            "assignment" | "augmented_assignment" | "for_statement" | "for_in_clause" => {
                let left = n.child_by_field_name("left");
                stack.extend(n.named_children(&mut cursor).map(|child| (child, at, Some(child) == left)));
            }
            "named_expression" => {
                let mut owner = at;
                while scopes[owner].kind == ScopeKind::Comprehension {
                    owner = scopes[owner].parent;
                }
                if let Some(name) = n.child_by_field_name("name") {
                    bind(&mut scopes, name, owner);
                }
                stack.extend(n.child_by_field_name("value").map(|value| (value, at, false)));
            }
            "as_pattern" => {
                let alias = n.child_by_field_name("alias");
                stack.extend(n.named_children(&mut cursor).map(|child| (child, at, Some(child) == alias)));
            }
            "pattern_list" | "tuple_pattern" | "list_pattern" | "list_splat_pattern"
            | "parenthesized_expression" | "expression_list" | "as_pattern_target" => {
                stack.extend(n.named_children(&mut cursor).map(|child| (child, at, target)));
            }
            "delete_statement" | "nonlocal_statement" | "dotted_name" | "splat_pattern" => {
                stack.extend(n.named_children(&mut cursor).map(|child| (child, at, true)));
            }
            _ => {
                stack.extend(n.named_children(&mut cursor).map(|child| (child, at, false)));
            }
        }
    }
    Ok(regions_of(scopes, module))
}

fn regions_of(mut scopes: Vec<Scope>, module: &ImportMap) -> Vec<ImportRegion> {
    let n = scopes.len();
    // A parent comes before its children in `scopes`. `inherited` is what the
    // scopes inside a scope see, which skips a class.
    let mut maps: Vec<Option<Arc<ImportMap>>> = vec![None; n];
    let mut inherited: Vec<Option<Arc<ImportMap>>> = vec![None; n];
    let mut kept = vec![false; n];
    for i in 1..n {
        let parent = scopes[i].parent;
        let parent_is_class = scopes[parent].kind == ScopeKind::Class;
        let scope = &mut scopes[i];
        let Scope { imports, bound, globals, .. } = scope;
        imports.retain(|(alias, _)| !globals.contains(alias));
        bound.retain(|name| !globals.contains(name));
        let base = inherited[parent].clone();
        // A `global` name skips the functions around the scope.
        globals.retain(|name| base.as_deref().is_some_and(|map| map.get(name) != module.get(name)));
        let changed = !imports.is_empty() || !bound.is_empty() || !globals.is_empty();
        let map = if changed {
            let mut map = base.as_deref().unwrap_or(module).clone();
            for name in globals.drain(..) {
                match module.get(&name) {
                    Some(import) => map.insert(name, import.clone()),
                    None => map.remove(&name),
                };
            }
            // The walk went from the last statement to the first, and the
            // last import of a name wins.
            map.extend(imports.drain(..).rev());
            for name in bound.drain(..) {
                let shadowed = ImportPath { dots: 0, module: Vec::new(), name: SHADOWED.to_string() };
                map.insert(name, shadowed);
            }
            Some(Arc::new(map))
        } else {
            base.clone()
        };
        // The region of a class must not reach the scopes inside it.
        kept[i] = changed || (parent_is_class && kept[parent]);
        inherited[i] = if scope.kind == ScopeKind::Class { base } else { map.clone() };
        maps[i] = map;
    }
    let mut order: Vec<usize> = (1..n).filter(|&i| kept[i]).collect();
    order.sort_by_key(|&i| (scopes[i].start, Reverse(scopes[i].end), i));
    let mut place = vec![0; n];
    for (at, &i) in order.iter().enumerate() {
        place[i] = at;
    }
    order
        .iter()
        .enumerate()
        .map(|(at, &i)| {
            let mut around = scopes[i].parent;
            while around != 0 && !kept[around] {
                around = scopes[around].parent;
            }
            ImportRegion {
                start: scopes[i].start,
                end: scopes[i].end,
                up: if around == 0 { 0 } else { at - place[around] },
                map: maps[i].take(),
            }
        })
        .collect()
}
