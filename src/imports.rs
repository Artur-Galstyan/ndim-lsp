//! What an imported name means at a place of the file. Python binds an import
//! in the scope of the statement, so the import map of a place is the module
//! map, changed by the scopes around that place. A scope that changes nothing
//! has no region: a file without such a scope pays one length check per lookup.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::Arc;

use tree_sitter::Node;

use crate::analysis::LIBRARIES;
use crate::python_ast::{ModuleBind, SelfWrap, build_import_map, module_binds, visit_imports};
use crate::resolution::resolve_call_target;
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

/// Whether the result of `wrap` takes the same arguments as the function that
/// it wraps. A keyword of `partial` binds an argument. The other transforms
/// have keywords for options only.
pub(crate) fn keeps_signature(wrap: &SelfWrap, imports: &ImportMap) -> bool {
    if !wrap.callee.split('.').next().is_some_and(|root| is_imported(imports, root)) {
        return false;
    }
    let target = resolve_call_target(&wrap.callee, imports);
    let parts: Vec<&str> = target.parts.iter().map(String::as_str).collect();
    match parts[..] {
        _ if target.dots != 0 => false,
        ["jax", "jit" | "checkpoint" | "remat"] | ["equinox", "filter_jit"] => true,
        ["functools", "partial"] => !wrap.keywords,
        _ => false,
    }
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
    /// Module code changes the module map from each of these places on.
    rebound: Vec<Rebound>,
}

/// The first place where a statement gives the name `name` another value.
#[derive(Debug, Clone, PartialEq)]
pub struct Rebound {
    pub(crate) at: usize,
    pub(crate) name: String,
    /// The module map from `at` on.
    map: Arc<ImportMap>,
}

/// What a module rebinds of the names that `file` holds, in source order. A
/// module with none of these pays nothing.
pub(crate) fn rebound_names(binds: &[ModuleBind], module: &ImportMap, file: &ImportMap) -> Vec<Rebound> {
    let keeps = |bind: &ModuleBind| {
        bind.wrap.as_ref().is_some_and(|wrap| {
            let root = wrap.callee.split('.').next();
            keeps_signature(wrap, module)
                && !binds.iter().any(|other| other.hides_import && Some(other.name.as_str()) == root)
        })
    };
    let mut first: HashMap<&str, usize> = HashMap::new();
    for bind in binds.iter().filter(|bind| bind.hides_import && tracked(&bind.name, file) && !keeps(bind)) {
        let at = first.entry(&bind.name).or_insert(bind.at);
        *at = bind.at.min(*at);
    }
    let mut first: Vec<_> = first.into_iter().collect();
    first.sort_unstable_by_key(|&(name, at)| (at, name));
    let mut map = module.clone();
    first
        .into_iter()
        .map(|(name, at)| {
            map.insert(name.to_string(), shadowed());
            Rebound { at, name: name.to_string(), map: Arc::new(map.clone()) }
        })
        .collect()
}

fn tracked(name: &str, file: &ImportMap) -> bool {
    file.contains_key(name) || LIBRARIES.contains(&name)
}

fn shadowed() -> ImportPath {
    ImportPath { dots: 0, module: Vec::new(), name: SHADOWED.to_string() }
}

impl ImportScopes {
    pub fn new(root: Node, text: &str) -> Result<Self, String> {
        let (file, module) = import_maps(root, text)?;
        let rebound = rebound_names(&module_binds(root, text), &module, &file);
        let regions = import_regions(root, text, &module, &file, &rebound)?;
        Ok(Self::from_parts(module, file, regions, rebound))
    }

    pub(crate) fn from_parts(
        module: ImportMap,
        file: ImportMap,
        regions: Vec<ImportRegion>,
        rebound: Vec<Rebound>,
    ) -> Self {
        Self { module, file, regions, rebound }
    }

    /// The module map that module code at `byte` reads.
    fn module_at(&self, byte: usize) -> &ImportMap {
        match self.rebound.partition_point(|rebound| rebound.at <= byte).checked_sub(1) {
            Some(last) => &self.rebound[last].map,
            None => &self.module,
        }
    }

    /// The imports that a name at `byte` can read.
    pub fn at(&self, byte: usize) -> &ImportMap {
        let last = self.regions.partition_point(|region| region.start <= byte);
        let Some(mut i) = last.checked_sub(1) else {
            return self.module_at(byte);
        };
        loop {
            let region = &self.regions[i];
            if byte < region.end {
                return region.map.as_deref().unwrap_or_else(|| self.module_at(byte));
            }
            if region.up == 0 {
                return self.module_at(byte);
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
/// means. `module` is the module map, `file` holds every import of the file,
/// and `rebound` holds what the module rebinds. A body runs after the module
/// code above it, so it gets the last map of `rebound`, and so does a class or
/// a comprehension that changes a name. The result depends on the names in
/// `rebound`, not on their places, and is the same for a unit as for the tree.
pub fn import_regions(
    node: Node,
    text: &str,
    module: &ImportMap,
    file: &ImportMap,
    rebound: &[Rebound],
) -> Result<Vec<ImportRegion>, String> {
    let source = text.as_bytes();
    let bind = |scopes: &mut Vec<Scope>, name: Node, at: usize| {
        if at != 0
            && let Ok(name) = name.utf8_text(source)
            && tracked(name, file)
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
    Ok(regions_of(scopes, module, rebound.last().map(|last| &last.map)))
}

fn regions_of(mut scopes: Vec<Scope>, module: &ImportMap, last: Option<&Arc<ImportMap>>) -> Vec<ImportRegion> {
    let module = last.map_or(module, |map| &**map);
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
        let body = scope.kind == ScopeKind::Function;
        let Scope { imports, bound, globals, .. } = scope;
        imports.retain(|(alias, _)| !globals.contains(alias));
        bound.retain(|name| !globals.contains(name));
        let base = inherited[parent].clone();
        // A `global` name skips the functions around the scope.
        globals.retain(|name| base.as_deref().is_some_and(|map| map.get(name) != module.get(name)));
        let late = base.is_none() && body && last.is_some();
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
                map.insert(name, shadowed());
            }
            Some(Arc::new(map))
        } else if late {
            last.cloned()
        } else {
            base.clone()
        };
        // The region of a class must not reach the scopes inside it.
        kept[i] = changed || late || (parent_is_class && kept[parent]);
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
