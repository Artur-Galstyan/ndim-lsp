//! The names a top-level statement defines and reads. Two statements that
//! share a name are linked, and a change to one can change the analysis of
//! every statement in its group.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

use tree_sitter::Node;

use crate::types::ImportPath;

/// FNV-1a. A collision of two names only joins their groups.
pub(super) fn name_hash(name: &[u8]) -> u64 {
    name.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The names of a statement are hashes already.
#[derive(Default)]
pub(super) struct Passthrough(u64);

impl Hasher for Passthrough {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0 = name_hash(bytes);
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }

    fn write_usize(&mut self, n: usize) {
        self.0 = (n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

type NameMap<V> = HashMap<u64, V, BuildHasherDefault<Passthrough>>;
type ByteSet = HashSet<usize, BuildHasherDefault<Passthrough>>;

#[derive(Debug, Default, PartialEq)]
pub(super) struct NameSet {
    /// Every identifier except the ones that only bind a name.
    pub reads: Vec<u64>,
    /// Functions, classes, and `self.<attr>` targets, at any depth. A read of
    /// the name can reach any of them.
    pub defs: Vec<u64>,
    /// Names bound in the module scope (class bodies and blocks included),
    /// and `global` names. Statements that bind one name also depend on each
    /// other: the last writer decides.
    pub shared: Vec<u64>,
    /// The module scope names, to record the shape each ends with.
    pub assigned: Vec<String>,
    /// Plain assignments of a call, at any depth: the last segment of the
    /// callee, and the name assigned. A `vmap` binds a name for the whole file.
    pub calls: Vec<(u64, u64)>,
    pub globals: Vec<u64>,
    pub wildcard: bool,
}

/// Identifiers under `node`, not entering a nested scope or a block.
fn shallow_identifiers<'t>(node: Node<'t>, text: &'t str, out: &mut Vec<&'t str>) {
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "identifier" => out.push(&text[n.byte_range()]),
            "function_definition" | "class_definition" | "lambda" | "block" => {}
            _ => {
                let mut cursor = n.walk();
                stack.extend(n.children(&mut cursor));
            }
        }
    }
}

/// The nodes of an assignment target that the assignment itself binds: a
/// name or an attribute, alone or in a pattern. `a[i] = v` binds neither.
fn direct_targets<'t>(target: Node<'t>, out: &mut Vec<Node<'t>>) {
    match target.kind() {
        "identifier" | "attribute" => out.push(target),
        "pattern_list"
        | "tuple_pattern"
        | "list_pattern"
        | "list_splat_pattern"
        | "parenthesized_pattern" => {
            let mut cursor = target.walk();
            for child in target.named_children(&mut cursor) {
                direct_targets(child, out);
            }
        }
        _ => {}
    }
}

impl NameSet {
    pub(super) fn of(unit: Node, text: &str) -> Self {
        let mut set = NameSet::default();
        let mut skip = ByteSet::default();
        let mut assigned: Vec<&str> = Vec::new();
        let mut cursor = unit.walk();
        let mut flags = vec![true];
        let hash = |n: Node| name_hash(&text.as_bytes()[n.byte_range()]);
        'walk: loop {
            let node = cursor.node();
            let module_level = *flags.last().unwrap();
            let mut inner_module_level = module_level;
            let mut descend = true;
            match node.kind() {
                "identifier" => {
                    if !skip.contains(&node.start_byte()) {
                        set.reads.push(hash(node));
                    }
                    descend = false;
                }
                "function_definition" | "class_definition" => {
                    if let Some(name) = node.child_by_field_name("name") {
                        set.defs.push(hash(name));
                        skip.insert(name.start_byte());
                    }
                    inner_module_level = module_level && node.kind() == "class_definition";
                }
                "lambda" => inner_module_level = false,
                "assignment" | "augmented_assignment" | "for_statement" => {
                    let plain = node.kind() != "augmented_assignment";
                    if let Some(left) = node.child_by_field_name("left") {
                        let mut direct = Vec::new();
                        direct_targets(left, &mut direct);
                        let mut inside = vec![left];
                        while let Some(n) = inside.pop() {
                            if n.kind() == "attribute"
                                && let Some(attr) = n.child_by_field_name("attribute")
                            {
                                set.defs.push(hash(attr));
                            }
                            let mut cursor = n.walk();
                            inside.extend(n.children(&mut cursor));
                        }
                        if plain {
                            for t in &direct {
                                let bound = match t.kind() {
                                    "identifier" => Some(*t),
                                    _ => t.child_by_field_name("attribute"),
                                };
                                if let Some(bound) = bound {
                                    skip.insert(bound.start_byte());
                                }
                            }
                        }
                        if module_level {
                            shallow_identifiers(left, text, &mut assigned);
                        }
                        if node.kind() == "assignment"
                            && left.kind() == "identifier"
                            && let Some(call) = node.child_by_field_name("right")
                            && call.kind() == "call"
                            && let Some(func) = call.child_by_field_name("function")
                        {
                            let callee = &text[func.byte_range()];
                            let last = callee.rsplit('.').next().unwrap_or(callee);
                            set.calls.push((name_hash(last.as_bytes()), hash(left)));
                        }
                    }
                }
                "as_pattern" | "named_expression" => {
                    let field = if node.kind() == "as_pattern" {
                        "alias"
                    } else {
                        "name"
                    };
                    if module_level && let Some(target) = node.child_by_field_name(field) {
                        shallow_identifiers(target, text, &mut assigned);
                    }
                }
                "delete_statement"
                | "type_alias_statement"
                | "case_pattern"
                | "except_clause"
                | "except_group_clause" => {
                    if module_level {
                        shallow_identifiers(node, text, &mut assigned);
                    }
                }
                "global_statement" | "nonlocal_statement" => {
                    let mut names = Vec::new();
                    shallow_identifiers(node, text, &mut names);
                    for name in names {
                        let h = name_hash(name.as_bytes());
                        set.shared.push(h);
                        if node.kind() == "global_statement" {
                            set.globals.push(h);
                        }
                    }
                }
                "wildcard_import" => set.wildcard = true,
                "parameters" | "lambda_parameters" => {
                    let mut cursor = node.walk();
                    for param in node.named_children(&mut cursor) {
                        let name = match param.kind() {
                            "identifier" => Some(param),
                            "default_parameter" | "typed_default_parameter" => {
                                param.child_by_field_name("name")
                            }
                            "typed_parameter"
                            | "list_splat_pattern"
                            | "dictionary_splat_pattern" => {
                                param.named_child(0).filter(|n| n.kind() == "identifier")
                            }
                            _ => None,
                        };
                        if let Some(name) = name {
                            skip.insert(name.start_byte());
                        }
                    }
                }
                "keyword_argument" => {
                    if let Some(name) = node.child_by_field_name("name") {
                        skip.insert(name.start_byte());
                    }
                }
                _ => {}
            }
            if descend && cursor.goto_first_child() {
                flags.push(inner_module_level);
                continue;
            }
            while !cursor.goto_next_sibling() {
                if !cursor.goto_parent() {
                    break 'walk;
                }
                flags.pop();
            }
        }
        set.shared
            .extend(assigned.iter().map(|n| name_hash(n.as_bytes())));
        set.assigned = assigned.iter().map(|n| n.to_string()).collect();
        for list in [
            &mut set.reads,
            &mut set.defs,
            &mut set.shared,
            &mut set.globals,
        ] {
            list.sort_unstable();
            list.dedup();
        }
        set.assigned.sort_unstable();
        set.assigned.dedup();
        set.calls.sort_unstable();
        set.calls.dedup();
        set
    }
}

pub(super) struct Dsu(Vec<u32>);

impl Dsu {
    pub(super) fn new(len: usize) -> Self {
        Self((0..len as u32).collect())
    }

    pub(super) fn push(&mut self) -> usize {
        self.0.push(self.0.len() as u32);
        self.0.len() - 1
    }

    pub(super) fn find(&mut self, mut x: usize) -> usize {
        while self.0[x] as usize != x {
            let parent = self.0[x] as usize;
            self.0[x] = self.0[parent];
            x = parent;
        }
        x
    }

    pub(super) fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[a.max(b)] = a.min(b) as u32;
        }
    }
}

/// The names that make a call to `vmap` bind a name for the whole file.
pub(super) fn vmap_names(import_map: &HashMap<String, ImportPath>) -> Vec<u64> {
    let mut names = vec![name_hash(b"vmap"), name_hash(b"filter_vmap")];
    for (alias, import) in import_map {
        if import.name == "vmap" || import.name == "filter_vmap" {
            names.push(name_hash(alias.as_bytes()));
        }
    }
    names
}

/// Join, in `dsu`, node `base + i` with every node that shares a name with
/// statement `i`.
pub(super) fn link_units(sets: &[&NameSet], vmaps: &[u64], base: usize, dsu: &mut Dsu) {
    // Per name: its node, and whether a statement reads it.
    let mut names: NameMap<(usize, bool)> = NameMap::default();
    let shared_of = |set: &NameSet| -> Vec<u64> {
        let mut extra = Vec::new();
        for (callee, assigned) in &set.calls {
            if vmaps.contains(callee) {
                extra.push(*assigned);
            }
        }
        extra
    };
    let node_of = |names: &mut NameMap<(usize, bool)>, dsu: &mut Dsu, name: u64| -> usize {
        names.entry(name).or_insert_with(|| (dsu.push(), false)).0
    };
    for (i, set) in sets.iter().enumerate() {
        for &name in set.defs.iter().chain(&set.shared) {
            node_of(&mut names, dsu, name);
        }
        for name in set.shared.iter().copied().chain(shared_of(set)) {
            let node = node_of(&mut names, dsu, name);
            dsu.union(base + i, node);
        }
    }
    for (i, set) in sets.iter().enumerate() {
        for name in &set.reads {
            if let Some((node, read)) = names.get_mut(name) {
                *read = true;
                dsu.union(base + i, *node);
            }
        }
    }
    for (i, set) in sets.iter().enumerate() {
        for name in &set.defs {
            let (node, read) = names[name];
            if read {
                dsu.union(base + i, node);
            }
        }
    }
}
