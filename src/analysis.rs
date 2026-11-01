use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use tree_sitter::Node;

use crate::imports::{ImportScopes, is_imported};
use crate::known_functions::{
    apply_known_function, apply_known_kthvalue_shape, apply_known_linalg_lstsq_solution,
    apply_known_reduction, apply_known_topk_shape, apply_method_call, classify_known_function,
    classify_method_call, compute_array_split_shapes, compute_chunk_shapes,
    compute_einops_pack_shape, compute_fixed_axis_split_shapes, compute_split_shapes,
    compute_torch_split_shapes,
    compute_unbind_shape, multiply_dim,
};
use crate::incremental::{
    NodeRef, PrePass, PropPlan, PropagateOutput, Segment, UnitMemo, UnitRun, unit_nodes,
};
use crate::layers::{
    UNPROVEN_DIM, apply_layer_application, apply_layer_kind, classify_inline_constructor,
    strip_self_prefixes,
};
use crate::python_ast::{
    ClassHierarchy, SelfAttrWrite, dim_symbols, directly_in_class,
    extract_call_arguments, self_attr_aliases_from_writes,
};
use crate::resolution::{ResolutionCache, resolve_call_target, resolve_imported_function_shape};

use crate::types::*;

mod flow;
mod inner;
mod layer_dims;
mod modules;

pub(crate) use modules::LIBRARIES;
use flow::{Marked, Replay, collect_flow};
use modules::{
    Field, Module, apply_module, apply_sequence_call, changed_in_place, field_of, imported_class,
    imported_module_of, loop_variables, shape_of_indexed_module_call, shape_of_loop_module_call,
};
use inner::{
    called_lambda, comprehension_names, is_single_array_argument, lambda_argument,
    scan_lambda_output, shape_of_batched_lambda, shape_of_direct_lambda, shape_of_sum,
};

pub fn analyze_layer_shapes<F>(
    node: Node,
    text: &str,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<LayerShapeAnalysis, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    if let Some(cache) = cache {
        cache.revalidate(&[]);
    }
    analyze_tree(node, text, search_roots, &read_file, max_depth, cache)
}

/// `analyze_layer_shapes` without the check of the cache against the disk,
/// for a nested analysis that runs inside one (see `modules`).
fn analyze_tree<F>(
    node: Node,
    text: &str,
    search_roots: &[PathBuf],
    read_file: &F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<LayerShapeAnalysis, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let pre = PrePass::whole_tree(node, text, search_roots, read_file, max_depth, cache)?;
    finish_analysis(node, text, search_roots, read_file, max_depth, cache, pre, None)
}

/// Same result as `analyze_layer_shapes`, with the pre-passes taken from
/// `memo` for each top-level statement that did not change. `memo` becomes
/// the memo for the next call. Without a `cache` there is no way to know a
/// file on disk changed, so nothing is kept.
pub fn analyze_layer_shapes_with_memo<F>(
    node: Node,
    text: &str,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
    memo: &mut UnitMemo,
) -> Result<LayerShapeAnalysis, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let Some(cache) = cache else {
        *memo = UnitMemo::default();
        return analyze_layer_shapes(node, text, search_roots, read_file, max_depth, None);
    };
    cache.revalidate(&[]);
    let (pre, mut next) = match memo.prepass(node, text, search_roots, &read_file, max_depth, cache) {
        Ok((pre, next)) => (pre, Some(next)),
        Err(_) => (
            PrePass::whole_tree(node, text, search_roots, &read_file, max_depth, Some(cache))?,
            None,
        ),
    };
    let analysis = finish_analysis(
        node,
        text,
        search_roots,
        &read_file,
        max_depth,
        Some(cache),
        pre,
        next.as_mut(),
    );
    *memo = match analysis {
        Ok(_) => next.unwrap_or_default(),
        Err(_) => UnitMemo::default(),
    };
    analysis
}

#[allow(clippy::too_many_arguments)]
fn finish_analysis<F>(
    node: Node,
    text: &str,
    search_roots: &[PathBuf],
    read_file: &F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
    pre: PrePass,
    next: Option<&mut UnitMemo>,
) -> Result<LayerShapeAnalysis, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let PrePass {
        mut scopes,
        imports,
        layer_records,
        self_attr_layers,
        writes: self_attr_writes,
        hierarchy,
        plan,
    } = pre;
    let self_attr_aliases = self_attr_aliases_from_writes(&self_attr_writes, text);

    let (applications, errors, assignment_shapes) = match (plan, next) {
        (Some(plan), Some(next)) => propagate_planned(
            node,
            text,
            &imports,
            &layer_records,
            &self_attr_layers,
            &self_attr_aliases,
            &self_attr_writes,
            &hierarchy,
            search_roots,
            read_file,
            max_depth,
            cache,
            &mut scopes,
            plan,
            next,
        )?,
        _ => propagate_calls(
            node,
            text,
            &imports,
            &layer_records,
            &self_attr_layers,
            &self_attr_aliases,
            &self_attr_writes,
            &hierarchy,
            search_roots,
            read_file,
            max_depth,
            cache,
            &mut scopes,
        )?,
    };

    let mut layers = HashMap::new();
    for rec in &layer_records {
        layers.insert(rec.name.clone(), strip_self_prefixes(rec.kind.clone()));
    }

    Ok(LayerShapeAnalysis {
        scopes,
        layers,
        applications,
        errors,
        assignment_shapes,
    })
}

/// A non-annotated assignment whose RHS the analyzer could not shape.
/// `kind` buckets the cause (e.g. `"subscript"`, `"call:jnp.diff"`) so the
/// coverage harness can rank gaps by how often they actually occur.
#[derive(Debug, Clone, PartialEq)]
pub struct DarkSpot {
    pub line: u32,
    pub kind: String,
}

/// Shape-coverage of a single source file: how many assignments got a shape
/// vs. went dark, and what the dark ones were. Drives the corpus harness that
/// prioritizes which structural gaps / functions to implement next.
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageReport {
    /// Non-annotated, non-`_` assignments considered.
    pub total: usize,
    pub shaped: usize,
    pub dark: Vec<DarkSpot>,
}

/// Run the analyzer over a file and report which assignments went dark.
pub fn analyze_coverage<F>(
    node: Node,
    text: &str,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Result<CoverageReport, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let analysis = analyze_layer_shapes(node, text, search_roots, read_file, max_depth, cache)?;
    let shaped: std::collections::HashSet<(u32, &str)> = analysis
        .assignment_shapes
        .iter()
        .map(|r| (r.line, r.name.as_str()))
        .collect();

    let imports = ImportScopes::new(node, text)?;
    let items = collect_assignment_items(node, text)?;

    let mut total = 0;
    let mut shaped_count = 0;
    let mut dark = Vec::new();

    for (lhs, rhs, assignment) in items {
        // Annotated assignments carry a user-written shape — not a gap.
        if assignment.child_by_field_name("type").is_some() {
            continue;
        }
        let names: Vec<String> = match lhs {
            Lhs::Single(name) | Lhs::Augmented(name) => vec![name],
            Lhs::Tuple(targets) => targets
                .iter()
                .flat_map(|t| t.names())
                .map(str::to_string)
                .collect(),
        };
        if names.is_empty() {
            continue; // all `_` targets
        }
        let line = assignment.start_position().row as u32;
        total += 1;
        if names.iter().any(|n| shaped.contains(&(line, n.as_str()))) {
            shaped_count += 1;
        } else {
            dark.push(DarkSpot {
                line,
                kind: dark_spot_kind(rhs, text, imports.at(rhs.start_byte())),
            });
        }
    }

    Ok(CoverageReport {
        total,
        shaped: shaped_count,
        dark,
    })
}

/// Bucket a dark assignment's RHS by cause. Calls are labelled with their
/// (resolved) function name so missing functions surface by name.
fn dark_spot_kind(rhs: Node, text: &str, import_map: &HashMap<String, ImportPath>) -> String {
    match rhs.kind() {
        "subscript" => "subscript".to_string(),
        "unary_operator" => "unary".to_string(),
        "binary_operator" => "binary_operator".to_string(),
        "attribute" => "attribute".to_string(),
        "call" => {
            let Some(func) = rhs.child_by_field_name("function") else {
                return "call".to_string();
            };
            // Inline-call functions (e.g. `vmap(f)(x)`) have no plain name.
            if func.kind() == "call" {
                return "call:inline".to_string();
            }
            let Ok(raw) = func.utf8_text(text.as_bytes()) else {
                return "call".to_string();
            };
            let resolved = resolve_call_target(raw, import_map);
            let name = if resolved.parts.is_empty() {
                raw.to_string()
            } else {
                resolved.parts.join(".")
            };
            format!("call:{name}")
        }
        other => other.to_string(),
    }
}

fn find_scoped_layer<'a>(
    records: &'a [LayerAssignment],
    scopes: &[FunctionShapeScope],
    index: &ScopeIndex,
    call_byte: usize,
    target: &str,
) -> Option<&'a LayerAssignment> {
    let mut best: Option<(usize, usize)> = None;
    for (i, rec) in records.iter().enumerate() {
        if rec.name != target {
            continue;
        }
        let Some(rec_scope_idx) = index.lookup(rec.byte_position) else {
            continue;
        };
        let rec_scope = &scopes[rec_scope_idx];
        if rec_scope.start_byte <= call_byte && call_byte < rec_scope.end_byte {
            let size = rec_scope.end_byte - rec_scope.start_byte;
            match best {
                None => best = Some((i, size)),
                Some((_, prev)) if size <= prev => best = Some((i, size)),
                _ => {}
            }
        }
    }
    best.map(|(i, _)| &records[i])
}

// ── Recursive shape evaluator (#34) ─────────────────────────────────────
//
// Replaces the flat query-based extraction (extract_calls) with a
// recursive tree walk.
// For each assignment `lhs = rhs`, we call shape_of_expression(rhs) which
// recurses into nested calls, chained methods, unary wraps, parenthesised
// expressions, identifiers, and subscripts.

/// Context shared across recursive shape_of_expression calls within a single
/// propagate_calls pass.
struct ShapeCtx<'a> {
    text: &'a str,
    imports: &'a ImportScopes,
    layer_records: &'a [LayerAssignment],
    /// `self.<attr>` → layer bindings keyed by defining class range, from
    /// constructor assignments in `__init__`. Used to resolve
    /// `jax.vmap(self.<attr>)(x)` (issue #35) and direct `self.<attr>(x)` calls.
    self_attr_layers: &'a HashMap<String, Vec<ScopedSelfAttrLayer>>,
    /// `<attr>` → identifier bindings keyed by defining class range, from
    /// `self.<attr> = <ident>` assignments. Used to canonicalize symbolic
    /// dims (`self.dt_rank` ≡ `dt_rank`) before storage, resolved by the
    /// call-site byte via `self_attr_alias_at` — mirrors `self_attr_layers`.
    aliases: &'a HashMap<String, Vec<ScopedSelfAttrAlias>>,
    /// Every `self.<attr>` write of the file (`ShapeCtx::layer_at`).
    attr_writes: &'a [SelfAttrWrite<'a>],
    /// The dims of layer constructors written with their values: per place of
    /// the constructor and dim text (`ShapeCtx::layer_at`).
    layer_dims: std::cell::RefCell<HashMap<(usize, String), String>>,
    root: Node<'a>,
    hierarchy: &'a ClassHierarchy,
    /// Workspace/site-packages search roots, for resolving imported helper
    /// functions on disk (cross-file return-type tracing).
    search_roots: &'a [PathBuf],
    read_file: &'a dyn Fn(&PathBuf) -> Option<String>,
    max_depth: usize,
    cache: Option<&'a ResolutionCache>,
    scopes: &'a mut [FunctionShapeScope],
    vmap_targets: &'a mut HashMap<String, VmapInfo>,
    applications: &'a mut Vec<LayerApplication>,
    errors: &'a mut Vec<ShapeError>,
    /// Per-assignment shape records for inlay hints (issue #28).
    assignment_shapes: &'a mut Vec<AssignmentShape>,
    /// Monotonic counter for synthetic variable names used to bind inline
    /// expression shapes so that known-function / method-call helpers can
    /// look them up by name.
    synthetic_counter: usize,
    /// Side-channel map for synthetic bindings. Entries here are visible
    /// to `resolve_shape` lookups but are NOT persisted to
    /// `scopes[...].shapes`, preventing __synth_* keys from leaking
    /// into the LSP's inlay hints.
    synthetics: HashMap<usize, HashMap<String, Vec<String>>>,
    /// Source text of each synthetic binding, to show the user's own
    /// expression where an error message names a synthetic.
    synthetic_text: HashMap<String, String>,
    /// Real dimension behind an integer name unpacked from `x.shape`
    /// (`a, b, c = x.shape`), so `reshape(a * b, c)` shows `b*s`. Dropped
    /// when the name is rebound.
    scalar_dims: HashMap<(usize, String), String>,
    /// Names whose current binding is a homogeneous tuple carry (`init = (h,
    /// c)`, a scan carry, a seeded carry parameter): the one shape stands for
    /// every element. Any other name with a shape is an array only when
    /// `name_is_array` proves it. Dropped when the name is rebound.
    tuple_carries: HashSet<(usize, String)>,
    /// `function_definition` node for each entry of `scopes` (aligned by
    /// index; `None` for the module scope at index 0), precomputed once so
    /// lazy call-site parameter seeding (`specialize_callee_call`) can find
    /// a callee's own body/return statements without re-walking the whole
    /// tree per call site.
    scope_function_nodes: &'a [Option<Node<'a>>],
    /// Stack of scope indices currently being lazily specialized (lazy
    /// call-site parameter seeding) — a recursion/cycle guard. Re-entering a
    /// scope already on this stack (direct or mutual recursion) aborts that
    /// specialization with `None` rather than looping forever; depth is
    /// additionally capped at `MAX_SPECIALIZATION_DEPTH`.
    active_specializations: Vec<usize>,
    /// Scope indices that have already had a "first-call-wins" successful
    /// specialization (see `specialize_callee_call`). Only the first
    /// specialization of a given scope writes its computed local shapes,
    /// `assignment_shapes`, and `errors` back into the shared analysis;
    /// every later specialization of the same scope (a different call site,
    /// possibly with different argument shapes) runs in an ephemeral copy
    /// that is discarded afterward, so it can't corrupt the first call's
    /// hover/inlay info or leak errors that only apply to its own seeding.
    specialized_scopes: std::collections::HashSet<usize>,
    /// Snapshot of every scope's `shapes` map taken ONCE, before any
    /// assignment is processed and before any specialization ever runs —
    /// i.e. exactly the jaxtyping/scalar-type annotations `extract_
    /// jaxtyping_shapes` produced, call-site-independent by construction.
    ///
    /// This is the correctness-critical piece of lazy call-site parameter
    /// seeding: `specialize_callee_call` must always start a fresh
    /// specialization from THIS baseline, never from the live (mutable)
    /// `scopes[...].shapes` — the live map may already carry a PRIOR call
    /// site's seeded params/locals (first-call-wins write-back), and
    /// blindly cloning it as the new starting point would leak that other
    /// call site's argument shapes into this one (`x.entry(name).or_insert`
    /// is a no-op once `x` is already present). Also used to decide,
    /// call-site-independently, whether a callee is "fully annotated" (safe
    /// for `bind_user_function_args`'s body-tracing path) vs. needs seeding.
    original_shapes: &'a [HashMap<String, Vec<String>>],
    /// Lexical links between function scopes (closures reading variables of
    /// the enclosing function).
    links: &'a ScopeLinks,
    submodules: &'a Submodules,
    /// Names some `global` statement of the file mentions, found on first use.
    global_names: std::cell::OnceCell<HashSet<String>>,
    /// Class bodies by byte range, and where each class starts.
    classes: ScopeIndex,
    class_starts: Vec<usize>,
    /// Plain assignments at module level (class bodies excluded) seen so far,
    /// per name, so a function can read the name as a module global.
    module_writes: HashMap<String, ModuleWrites>,
    /// Names some class body binds. Class bodies share scope 0 with the
    /// module, so a global of one of these names is not told apart from it.
    class_body_names: std::cell::OnceCell<HashSet<String>>,
    /// Per scope and name: where the top-level statement of the scope starts,
    /// when no function around it binds the name (`module_global`).
    global_scopes: std::cell::RefCell<HashMap<(usize, String), Option<usize>>>,
    /// Per name: the end of each module-level assignment (`module_global`).
    module_values: std::cell::RefCell<HashMap<String, Option<Vec<usize>>>>,
    /// How the module scope binds each name, found on first use.
    module_index: std::cell::OnceCell<ModuleIndex>,
    /// The names the jaxtyping annotations of the file use as dims, found on
    /// first use. A constant of such a name stays a name (`integer_constant`).
    annotation_symbols: std::cell::OnceCell<HashSet<String>>,
    /// Names assigned an integer literal so far, to try them as constant dims.
    int_names: HashSet<String>,
    /// Per scope owner (a node id) and name: the integer the name stands for.
    int_consts: std::cell::RefCell<HashMap<(usize, String), Option<String>>>,
    /// Closure scopes currently being evaluated at a call made from inside
    /// their enclosing function, so a name they do not bind resolves to the
    /// enclosing function's live shapes: Python reads those at call time.
    /// Outside these scopes (the whole-file pass evaluates a closure body at
    /// its definition) enclosing variables stay unknown.
    call_time: Vec<usize>,
    /// Names a lambda or comprehension being evaluated binds to something
    /// that is not a known array, per scope: a lookup finds nothing for them,
    /// whatever an enclosing scope or the module holds under the same name.
    inner_masked: Vec<(usize, String)>,
    /// Per binder node (a lambda or comprehension) and name: bound to a Python
    /// number, or to an array, while the body is evaluated.
    inner_scalars: HashSet<(usize, String)>,
    inner_arrays: HashSet<(usize, String)>,
}

/// The module-level assignments of one name seen so far.
struct ModuleWrites {
    count: usize,
    /// The shape every write left the name with. `None` once a write left it
    /// unknown or two writes disagreed.
    shape: Option<Vec<String>>,
}

/// A class that defines a method, and the scope of the method.
type MethodScope = ((usize, usize), usize);

/// Lexical facts about function scopes, computed only for closures and the
/// functions that enclose them (every other entry stays empty).
#[derive(Default)]
struct ScopeLinks {
    scope_index: ScopeIndex,
    /// Enclosing function scope. `None` for module-level functions and for
    /// methods, whose class body sits between them and the function.
    parent: Vec<Option<usize>>,
    /// Names the scope binds itself: parameters, assignment targets, loop and
    /// `with` targets, imports, nested `def`/`class` names, `global` names.
    /// A `nonlocal` name is not local.
    locals: Vec<HashSet<String>>,
    /// Names a nested closure rebinds through `nonlocal`. A call can change
    /// them at any time, so lookups in this scope treat them as unknown.
    poisoned: Vec<HashSet<String>>,
    /// The scope mentions a name that an enclosing function binds.
    captures: Vec<bool>,
    /// Another function of the same name is defined in the same function or
    /// class body, or at module level.
    redefined: Vec<bool>,
    /// The class whose body holds the definition of each scope.
    class_of: Vec<Option<(usize, usize)>>,
    methods: HashMap<String, Vec<MethodScope>>,
}

struct ScopeLevel<'c> {
    shapes: &'c HashMap<String, Vec<String>>,
    synthetics: Option<&'c HashMap<String, Vec<String>>>,
    locals: &'c HashSet<String>,
    poisoned: &'c HashSet<String>,
}

/// Borrowed view of one scope's shapes plus its synthetic bindings, followed
/// by the enclosing scopes a closure evaluated at call time can see.
/// Passed to the apply helpers as `&dyn ShapeLookup` so no merged map is
/// ever cloned (#43).
struct ScopeShapes<'c> {
    here: ScopeLevel<'c>,
    outer: Vec<ScopeLevel<'c>>,
    /// Where a name no scope binds may be a module global of this scope.
    globals: (&'c ShapeCtx<'c>, usize),
}

impl ShapeLookup for ScopeShapes<'_> {
    fn shape(&self, name: &str) -> Option<&Vec<String>> {
        let ctx = self.globals.0;
        if !ctx.inner_masked.is_empty()
            && !self.here.shapes.contains_key(name)
            && ctx.inner_masked.iter().any(|(scope, n)| *scope == self.globals.1 && n == name)
        {
            return None;
        }
        for level in std::iter::once(&self.here).chain(&self.outer) {
            if level.poisoned.contains(name) {
                return None;
            }
            if let Some(shape) = level
                .shapes
                .get(name)
                .or_else(|| level.synthetics?.get(name))
            {
                return Some(shape);
            }
            if level.locals.contains(name) {
                return None;
            }
        }
        self.globals.0.module_global(name, self.globals.1)
    }
}

/// Canonicalize every dim of a shape through the `self.<attr> = <ident>`
/// alias map (no-op when the map is empty or no dim mentions `self.`).
/// `byte` is the call/assignment site used to pick the enclosing class's
/// alias binding when the same attr is aliased differently across classes.
fn normalize_shape(
    shape: Vec<String>,
    aliases: &HashMap<String, Vec<ScopedSelfAttrAlias>>,
    hierarchy: &ClassHierarchy,
    byte: usize,
) -> Vec<String> {
    if aliases.is_empty() {
        return shape;
    }
    shape
        .into_iter()
        .map(|d| normalize_dim(&d, aliases, hierarchy, byte))
        .collect()
}

/// Resolve `self.<attr>`'s aliased identifier through the class enclosing
/// `byte`, then its bases. Mirrors `ShapeCtx::self_attr_layer_at`.
fn resolve_alias_at<'a>(
    aliases: &'a HashMap<String, Vec<ScopedSelfAttrAlias>>,
    hierarchy: &ClassHierarchy,
    attr: &str,
    byte: usize,
) -> Option<&'a str> {
    hierarchy
        .resolve(aliases.get(attr)?, byte, |e| (e.class_start, e.class_end))
        .map(|e| e.value.as_str())
}

/// Replace each `self.<attr>` token in a dim expression with its aliased value
/// (e.g. `self.dt_rank + self.d_state` → `dt_rank + d_state` when both attrs
/// were assigned from same-named identifiers). Tokens whose attribute has no
/// alias (in the class enclosing `byte`) are left untouched.
fn normalize_dim(
    dim: &str,
    aliases: &HashMap<String, Vec<ScopedSelfAttrAlias>>,
    hierarchy: &ClassHierarchy,
    byte: usize,
) -> String {
    if !dim.contains("self.") {
        return dim.to_string();
    }
    let bytes = dim.as_bytes();
    let mut result = String::with_capacity(dim.len());
    let mut i = 0;
    while i < dim.len() {
        if dim[i..].starts_with("self.") {
            let attr_start = i + 5;
            let mut j = attr_start;
            while j < dim.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            if let Some(val) = resolve_alias_at(aliases, hierarchy, &dim[attr_start..j], byte) {
                result.push_str(val);
                i = j;
                continue;
            }
        }
        let ch = dim[i..].chars().next().unwrap();
        result.push(ch);
        i += ch.len_utf8();
    }
    result
}

impl<'a> ShapeCtx<'a> {
    /// Resolve `self.<attr>` to the layer bound in the class enclosing `byte`,
    /// else in its bases (MRO order). Unrelated classes never match, and a
    /// base the file does not define ends the search with None.
    fn self_attr_layer_at(&self, attr: &str, byte: usize) -> Option<LayerKind> {
        self.hierarchy
            .resolve(self.self_attr_layers.get(attr)?, byte, |e| {
                (e.class_start, e.class_end)
            })
            .map(|e| self.layer_at(&e.kind, e.byte))
    }

    /// The `self.<attr>` field of the class enclosing `byte`, else of its
    /// bases. Same lookup as `self_attr_layer_at`.
    fn submodule_at(&self, attr: &str, byte: usize) -> Option<&SubmoduleField> {
        self.hierarchy
            .resolve(self.submodules.fields.get(attr)?, byte, |e| {
                (e.class_start, e.class_end)
            })
    }

    /// Scope of the method that calling `module` runs: `__call__`, or
    /// `forward`, of the class or its nearest base that defines one. None when
    /// a class between them rebinds a field the method reads, since the
    /// method would then see the other binding.
    fn module_callee(&self, module: &Submodule) -> Option<usize> {
        let method = self
            .hierarchy
            .resolve(&self.submodules.callables, module.class.0, |m| {
                (m.class_start, m.class_end)
            })?;
        let defining = (method.class_start, method.class_end);
        let rebinds = |class: (usize, usize)| {
            self.submodules
                .fields
                .keys()
                .chain(self.self_attr_layers.keys())
                .chain(self.aliases.keys())
                .any(|attr| self.class_writes(attr, class) && self.class_writes(attr, defining))
        };
        let mro = self.hierarchy.mro(module.class);
        if mro.iter().take_while(|&&class| class != defining).any(|&c| rebinds(c)) {
            return None;
        }
        Some(method.scope)
    }

    /// The method that `receiver.<name>(...)` runs, when `receiver` is the
    /// `self` or `cls` of the enclosing method, or a `super()` call. `None`
    /// when no method of the file is sure to run.
    fn method_callee(
        &self,
        receiver: Node,
        is_super: bool,
        name: &str,
        call_byte: usize,
    ) -> Option<usize> {
        let word = if is_super { "super" } else { &self.text[receiver.byte_range()] };
        if !matches!(word, "self" | "cls" | "super") {
            return None;
        }
        let links = self.links;
        let mut method = links.scope_index.lookup(call_byte)?;
        while links.class_of[method].is_none() {
            if self.scopes[method].all_params.iter().any(|p| p == word) {
                return None;
            }
            method = links.parent[method]?;
        }
        let class = links.class_of[method]?;
        let first = self.scopes[method].all_params.first()?.as_str();
        let kind = method_kind(self.scope_function_nodes[method]?, self.text);
        let on_class = match (word, first, kind) {
            ("self" | "super", "self", MethodKind::Instance) => false,
            ("cls" | "super", "cls", MethodKind::Class) => true,
            _ => return None,
        };
        if is_super {
            let args = receiver.child_by_field_name("arguments")?;
            let mut cursor = args.walk();
            let explicit: Vec<&str> = args
                .named_children(&mut cursor)
                .filter(|arg| arg.kind() != "comment")
                .map(|arg| &self.text[arg.byte_range()])
                .collect();
            match explicit[..] {
                [] => {}
                [of, instance] if instance == first && self.hierarchy.class_named(of) == Some(class) => {}
                _ => return None,
            }
        }

        let methods = links.methods.get(name);
        let callee = self.hierarchy.resolve_method(class, is_super, |c| {
            if self.hierarchy.binds_attr(c, name) || self.class_writes(name, c) {
                return Some(None);
            }
            methods?.iter().find(|m| m.0 == c).map(|m| Some(m.1))
        })?;
        let scope = &self.scopes[callee];
        if links.redefined[callee]
            || (scope.start_byte <= call_byte && call_byte < scope.end_byte)
            || !self.receiver_holds(receiver, word)
            || (is_super && !self.receiver_holds(receiver, first))
        {
            return None;
        }
        let receiver_param = matches!(scope.all_params.first().map(String::as_str), Some("self" | "cls"));
        let kind = method_kind(self.scope_function_nodes[callee]?, self.text);
        match kind {
            MethodKind::Instance if !on_class && receiver_param => Some(callee),
            MethodKind::Class if receiver_param => Some(callee),
            MethodKind::Static if !receiver_param => Some(callee),
            _ => None,
        }
    }

    /// Whether no scope from `at` up to the method rebinds `word`, the `self`
    /// or `cls` of the method. `super` must have no binding at all.
    fn receiver_holds(&self, at: Node, word: &str) -> bool {
        if word == "super" {
            return matches!(nearest_binding(at, word, self).1, NameBinding::Unbound);
        }
        let mut up = at.parent();
        while let Some(node) = up {
            match node.kind() {
                "lambda" | "list_comprehension" | "set_comprehension"
                | "dictionary_comprehension" | "generator_expression"
                    if inner_scope_binds(node, word, self.text) =>
                {
                    return false;
                }
                "function_definition" => {
                    let (Some(scope), Some(body)) =
                        (function_scope(node, self), node.child_by_field_name("body"))
                    else {
                        return false;
                    };
                    let (mut values, mut bound) = (Vec::new(), false);
                    if self.links.poisoned[scope].contains(word)
                        || !collect_name_values(body, word, self.text, &mut values, &mut bound)
                        || bound
                    {
                        return false;
                    }
                    if self.links.class_of[scope].is_some() {
                        return true;
                    }
                }
                _ => {}
            }
            up = node.parent();
        }
        false
    }

    /// Whether `class` itself writes `self.<attr>` as a layer, alias, or field.
    fn class_writes(&self, attr: &str, class: (usize, usize)) -> bool {
        let at = |start: usize, end: usize| (start, end) == class;
        self.submodules
            .fields
            .get(attr)
            .is_some_and(|v| v.iter().any(|e| at(e.class_start, e.class_end)))
            || self
                .self_attr_layers
                .get(attr)
                .is_some_and(|v| v.iter().any(|e| at(e.class_start, e.class_end)))
            || self
                .aliases
                .get(attr)
                .is_some_and(|v| v.iter().any(|e| at(e.class_start, e.class_end)))
    }

    /// Looks up a name that means something only inside one instance of
    /// `module`: `None` for any other name, else the value its constructor
    /// call gives it (`Some(None)`: no usable value). A value is an integer or
    /// a plain name.
    ///
    /// The names are the parameters of `__init__` (or, without one, the
    /// dataclass-style fields, in dataclass order) and every attribute the
    /// class writes. Layer dims lose their `self.` prefix at classification,
    /// so `self.features` can appear as `features`. An attribute holds its
    /// value only through `self.<attr> = <parameter>`.
    fn callee_names<'s>(
        &'s self,
        module: &'s Submodule,
    ) -> impl Fn(&str) -> Option<Option<String>> + 's {
        let at = |m: &ClassMethod| (m.class_start, m.class_end);
        let init = self.hierarchy.resolve(&self.submodules.inits, module.class.0, at);
        let params: Vec<String> = match init {
            Some(m) => without_receiver(&self.scopes[m.scope].all_params).to_vec(),
            None => self.dataclass_fields(module.class),
        };
        // Without `__init__` the positional order is the dataclass order,
        // which an undefined base can change.
        let mut positional = init.is_some() || self.hierarchy.mro_is_closed(module.class);
        let mut given: HashMap<String, String> = HashMap::new();
        let mut position = 0;
        for arg in &module.ctor_args {
            let (name, value) = match arg {
                CallArgument::Positional { value } if value.starts_with('*') => {
                    positional = false;
                    continue;
                }
                CallArgument::Positional { value } => {
                    position += 1;
                    match params.get(position - 1) {
                        Some(name) if positional => (name.as_str(), value),
                        _ => continue,
                    }
                }
                CallArgument::Keyword { name, value } => (name.as_str(), value),
            };
            let value = normalize_dim(value, self.aliases, self.hierarchy, module.byte);
            if let Some(value) = self.argument_value(&value, module.byte)
                && !value.is_empty()
                && value.chars().all(|c| c.is_alphanumeric() || c == '_')
            {
                given.insert(name.to_string(), value);
            }
        }
        move |name| {
            if self.submodule_at(name, module.class.0).is_some() {
                let param = resolve_alias_at(self.aliases, self.hierarchy, name, module.class.0);
                return Some(param.and_then(|param| given.get(param).cloned()));
            }
            params.iter().any(|p| p == name).then(|| given.get(name).cloned())
        }
    }

    /// Dataclass-style fields of `class` and its bases defined in the file,
    /// bases first.
    fn dataclass_fields(&self, class: (usize, usize)) -> Vec<String> {
        let mut fields: Vec<String> = Vec::new();
        for base in self.hierarchy.mro(class).iter().rev() {
            for name in self.hierarchy.fields(*base) {
                if !fields.contains(name) {
                    fields.push(name.clone());
                }
            }
        }
        fields
    }

    /// Insert a shape under a synthetic name and return that name.
    /// Used when an inline expression (e.g. a nested call) produces a shape
    /// but has no variable binding of its own. The helpers in
    /// `known_functions` look up arguments by name in `shapes`, so we need
    /// a name. `byte` is the originating expression's byte position, used to
    /// resolve `self.<attr>` aliases against the enclosing class.
    fn bind_synthetic(
        &mut self,
        shape: Vec<String>,
        scope_idx: usize,
        byte: usize,
        text: &str,
    ) -> String {
        let name = format!("__synth_{}", self.synthetic_counter);
        self.synthetic_counter += 1;
        self.synthetic_text.insert(name.clone(), text.to_string());
        let shape = normalize_shape(shape, self.aliases, self.hierarchy, byte);
        self.synthetics
            .entry(scope_idx)
            .or_default()
            .insert(name.clone(), shape);
        name
    }

    /// Bind a user-visible shape and, when `line` is `Some` (a non-annotated
    /// assignment), record it for per-reassignment inlay hints (issue #28).
    /// `byte` is the assignment's byte position, used to resolve
    /// `self.<attr>` aliases against the enclosing class.
    fn record_binding(
        &mut self,
        scope_idx: usize,
        name: &str,
        shape: Vec<String>,
        line: Option<u32>,
        byte: usize,
    ) {
        self.scalar_dims.remove(&(scope_idx, name.to_string()));
        if !self.tuple_carries.is_empty() {
            self.tuple_carries.remove(&(scope_idx, name.to_string()));
        }
        let shape = normalize_shape(shape, self.aliases, self.hierarchy, byte);
        if shape.iter().any(|dim| has_internal_name(dim)) {
            self.evict_binding(scope_idx, name);
            return;
        }
        if let Some(line) = line {
            self.assignment_shapes.push(AssignmentShape {
                line,
                name: name.to_string(),
                shape: shape.clone(),
            });
        }
        self.scopes[scope_idx].shapes.insert(name.to_string(), shape);
    }

    /// Drop a stale binding when a reassignment's RHS can't be shaped.
    /// Keeping the old shape would make every downstream use reason from
    /// stale data and report confident false diagnostics (issue #46).
    /// Whether an annotation of the file uses `name` as a dim.
    fn is_annotation_symbol(&self, name: &str) -> bool {
        self.annotation_symbols
            .get_or_init(|| {
                let declared = self.original_shapes.iter().flat_map(|s| s.values().flatten());
                let returned = self.scopes.iter().flat_map(|s| s.return_shape.iter().flatten());
                declared
                    .chain(returned)
                    .flat_map(|dim| dim_symbols(dim))
                    .map(str::to_string)
                    .collect()
            })
            .contains(name)
    }

    fn evict_binding(&mut self, scope_idx: usize, name: &str) {
        self.scalar_dims.remove(&(scope_idx, name.to_string()));
        if !self.tuple_carries.is_empty() {
            self.tuple_carries.remove(&(scope_idx, name.to_string()));
        }
        self.scopes[scope_idx].shapes.remove(name);
    }

    /// Whether `byte` is in a class body with no function between it and the
    /// class, given the function scope the byte is in.
    fn in_class_body(&self, byte: usize, scope: Option<usize>) -> bool {
        let Some(class) = self.classes.lookup(byte) else {
            return false;
        };
        match scope {
            Some(0) => true,
            Some(function) => self.class_starts[class] > self.scopes[function].start_byte,
            None => false,
        }
    }

    /// Fold the shape a module-level assignment of `name` just left into the
    /// record `module_global_shape` reads.
    fn note_module_write(&mut self, name: &str) {
        let now = self.scopes[0].shapes.get(name).cloned();
        let entry = self
            .module_writes
            .entry(name.to_string())
            .or_insert_with(|| ModuleWrites {
                count: 0,
                shape: now.clone(),
            });
        entry.count += 1;
        if entry.shape != now {
            entry.shape = None;
        }
    }

    /// The shape a function (scope `scope_idx`) sees for `name` when it reads
    /// a module global. Python reads the value at call time, so this holds
    /// only when no write can differ from the others: every module-level
    /// write is a plain assignment that stands before the top-level statement
    /// holding the function, all of them left the same shape, and the name
    /// still has it in scope 0. No function around the reader may bind the
    /// name, and no `global` statement or class body may write it.
    fn module_global(&self, name: &str, scope_idx: usize) -> Option<&Vec<String>> {
        let writes = self.module_writes.get(name)?;
        let shape = writes.shape.as_ref()?;
        if scope_idx == 0 || self.scopes[0].shapes.get(name) != Some(shape) {
            return None;
        }
        let function = self.scope_function_nodes.get(scope_idx).copied().flatten()?;
        let before = *self
            .global_scopes
            .borrow_mut()
            .entry((scope_idx, name.to_string()))
            .or_insert_with(|| self.global_unbound_before(function, scope_idx, name));
        let before = before?;
        let ends = self
            .module_values
            .borrow_mut()
            .entry(name.to_string())
            .or_insert_with(|| self.module_value_ends(function, name))
            .clone()?;
        (ends.len() == writes.count && ends.iter().all(|&end| end <= before)).then_some(shape)
    }

    /// Start of the top-level statement around `function` when no function
    /// around the reader binds `name`.
    fn global_unbound_before(&self, function: Node, scope_idx: usize, name: &str) -> Option<usize> {
        let mut owner = function;
        let mut idx = scope_idx;
        let mut top = function;
        loop {
            if !matches!(name_binding(owner, idx, name, self), NameBinding::Unbound) {
                return None;
            }
            loop {
                let parent = owner.parent()?;
                if parent.kind() == "module" {
                    return Some(top.start_byte());
                }
                top = parent;
                owner = parent;
                if owner.kind() == "function_definition" {
                    idx = function_scope(owner, self)?;
                    break;
                }
            }
        }
    }

    /// End byte of every module-level assignment of `name`, when they are all
    /// plain assignments and no class body binds the name.
    fn module_value_ends(&self, function: Node, name: &str) -> Option<Vec<usize>> {
        let mut root = function;
        while let Some(parent) = root.parent() {
            root = parent;
        }
        let NameBinding::Values(values) = name_binding(root, 0, name, self) else {
            return None;
        };
        (!self.class_body_binds(root, name)).then(|| values.iter().map(|v| v.end_byte()).collect())
    }

    fn class_body_binds(&self, root: Node, name: &str) -> bool {
        self.class_body_names
            .get_or_init(|| {
                let mut names = HashSet::new();
                collect_class_body_names(root, self.text, &mut names);
                names
            })
            .contains(name)
    }

    /// Look up a shape by name in the given scope, checking both real
    /// user-visible shapes and the synthetic side-channel.
    fn resolve_shape(&self, name: &str, scope_idx: usize) -> Option<Vec<String>> {
        self.scope_shapes(scope_idx).shape(name).cloned()
    }

    /// Zero-copy `ShapeLookup` over a scope's shapes plus its synthetic
    /// bindings, for passing to the apply helpers.
    fn scope_shapes(&self, scope_idx: usize) -> ScopeShapes<'_> {
        let mut outer = Vec::new();
        let mut cur = scope_idx;
        while self.call_time.contains(&cur)
            && let Some(parent) = self.links.parent[cur]
        {
            outer.push(self.scope_level(parent));
            cur = parent;
        }
        ScopeShapes {
            here: self.scope_level(scope_idx),
            outer,
            globals: (self, scope_idx),
        }
    }

    fn scope_level(&self, scope_idx: usize) -> ScopeLevel<'_> {
        ScopeLevel {
            shapes: &self.scopes[scope_idx].shapes,
            synthetics: self.synthetics.get(&scope_idx),
            locals: &self.links.locals[scope_idx],
            poisoned: &self.links.poisoned[scope_idx],
        }
    }

    /// Whether a call made in `call_scope` runs a closure of `parent` at a
    /// moment when `parent`'s live shapes are what the closure sees: the call
    /// is in `parent` itself, or in a closure being evaluated at its own call
    /// that sits (transitively) inside `parent`.
    fn call_sees_live(&self, call_scope: usize, parent: usize) -> bool {
        let mut cur = call_scope;
        while cur != parent {
            match self.links.parent[cur] {
                Some(up) if self.call_time.contains(&cur) => cur = up,
                _ => return false,
            }
        }
        true
    }
}

/// Recursively compute the shape of an expression node.
///
/// Returns `Some(shape)` if the shape could be determined, `None` if not
/// (silently skipped — unrecognised function, missing variable shape, etc.).
/// On error (dimension mismatch etc.) records a `ShapeError` and returns `None`.
fn shape_of_expression(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    match node.kind() {
        "call" => shape_of_call(node, ctx),
        "identifier" => shape_of_identifier(node, ctx),
        "attribute" => shape_of_attribute(node, ctx),
        "unary_operator" => shape_of_unary(node, ctx),
        "parenthesized_expression" => {
            // Transparent — recurse into the inner expression.
            let inner = node.named_child(0)?;
            shape_of_expression(inner, ctx)
        }
        "subscript" => shape_of_subscript(node, ctx),
        "binary_operator" => shape_of_binary_operator(node, ctx),
        "tuple" => shape_of_tuple(node, ctx),
        _ => None,
    }
}

/// A bare tuple literal (e.g. `(mean0, var0)`, whether inline in a call's
/// argument list or the RHS of a plain single-name assignment like
/// `init = (mean0, var0)`). Representable as a single `Vec<String>` shape
/// only when every element resolves to the *same* shape — the common case
/// for a homogeneous `scan` carry (e.g. per-feature EMA `(mean, var)`
/// stats, or an LSTM `(h, c)` pair with equal hidden sizes). Mirrors the
/// existing approximation for RNN `(h, c)` state in `tuple_rhs_shapes`'s
/// `Rnn`/`RnnCell` arms, where every nested name shares one shape.
/// Heterogeneous tuples aren't representable this way and stay `Ok(None)`.
fn shape_of_tuple(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    let mut elems = Vec::with_capacity(node.named_child_count());
    for i in 0..node.named_child_count() {
        let child = node.named_child(i as u32)?;
        elems.push(shape_of_expression(child, ctx)?);
    }
    let (first, rest) = elems.split_first()?;
    if rest.iter().all(|s| s == first) {
        Some(first.clone())
    } else {
        None
    }
}

/// Look up a bare identifier in the current scope's shapes.
/// Checks the user-visible scope.shapes first, then falls back to the
/// synthetic side-channel (for inline expression bindings).
fn shape_of_identifier(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    let name = node.utf8_text(ctx.text.as_bytes()).ok()?;
    let scope_idx = ctx.links.scope_index.lookup(node.start_byte())?;
    ctx.resolve_shape(name, scope_idx)
}

/// Whether a function, a lambda, or a comprehension around `at` binds `name`
/// itself: as a parameter or by an assignment, a loop, a `def`. An import of
/// the name inside the function binds the same module.
fn shadowed_locally(at: Node, name: &str, ctx: &ShapeCtx) -> bool {
    let mut up = at.parent();
    while let Some(owner) = up {
        up = owner.parent();
        match owner.kind() {
            "function_definition" => {
                let Some(idx) = function_scope(owner, ctx) else {
                    return true;
                };
                if ctx.scopes[idx].all_params.iter().any(|param| param == name)
                    || owner
                        .child_by_field_name("body")
                        .is_some_and(|body| modules::binders(body, ctx.text, false).contains_key(name))
                {
                    return true;
                }
            }
            "lambda" | "list_comprehension" | "set_comprehension" | "dictionary_comprehension"
            | "generator_expression"
                if inner_scope_binds(owner, name, ctx.text) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

fn function_scope(function: Node, ctx: &ShapeCtx) -> Option<usize> {
    ctx.scope_function_nodes
        .iter()
        .position(|f| f.is_some_and(|f| f.id() == function.id()))
}

/// Whether a lambda or a comprehension binds `name` itself: as a parameter,
/// or as the target of one of its `for` clauses.
fn inner_scope_binds(node: Node, name: &str, text: &str) -> bool {
    let mut cursor = node.walk();
    node.named_children(&mut cursor).any(|child| match child.kind() {
        "lambda_parameters" => mentions_name(child, text, &|other| other == name),
        "for_in_clause" => child
            .child_by_field_name("left")
            .is_some_and(|left| mentions_name(left, text, &|other| other == name)),
        _ => false,
    })
}

/// Names bound directly in a class body, for every class that is not inside
/// a function (the others do not write scope 0).
fn collect_class_body_names(node: Node, text: &str, out: &mut HashSet<String>) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "function_definition" => {}
            "class_definition" => {
                if let Some(body) = child.child_by_field_name("body") {
                    let mut bindings = Bindings::default();
                    collect_bindings(body, text, &mut bindings);
                    out.extend(bindings.bound);
                    collect_class_body_names(body, text, out);
                }
            }
            _ => collect_class_body_names(child, text, out),
        }
    }
}

/// A standalone `attribute` node (not the `function` field of a call).
///
/// Handles `self.<field>` (issue #31): class fields carrying jaxtyping
/// annotations are extracted into the module scope (scope 0) by
/// `extract_jaxtyping_shapes`, so we resolve `self.<field>` to that shape.
///
/// Other attribute chains like `x.reshape(3,4).sum(axis=1)` are handled in
/// `shape_of_call`, where the parse produces a `call` whose `function` is the
/// attribute. A non-`self` attribute (e.g. `obj.field`) has no shape rule,
/// except the `x.T` / `x.mT` torch tensor properties (see below).
fn shape_of_attribute(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    let obj = node.child_by_field_name("object")?;
    let field = node
        .child_by_field_name("attribute")?
        .utf8_text(ctx.text.as_bytes())
        .ok()?;

    if obj.kind() == "identifier" && obj.utf8_text(ctx.text.as_bytes()).ok()? == "self" {
        // ponytail: class fields land flat in module scope (scope 0); two
        // classes with a same-named field resolve last-wins. Add per-class
        // scoping if bitten.
        if ctx.module_writes.contains_key(field) {
            return None;
        }
        return ctx.resolve_shape(field, 0);
    }

    // `x.T` (torch): reverses every dim (deprecated for rank > 2 but still
    // defined that way). `x.mT`: swaps only the last two dims, requires
    // rank >= 2.
    if field == "T" {
        let mut shape = shape_of_expression(obj, ctx)?;
        if shape.is_empty() {
            return None;
        }
        shape.reverse();
        return Some(shape);
    }
    if field == "mT" {
        let mut shape = shape_of_expression(obj, ctx)?;
        if shape.len() < 2 {
            return None;
        }
        let len = shape.len();
        shape.swap(len - 1, len - 2);
        return Some(shape);
    }

    None
}

/// Unary operator: propagate the operand's shape unchanged.
/// Subsumes issue #32: `-x`, `+x`, `~x` all preserve shape.
fn shape_of_unary(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    // The unary_operator node has one named child: the operand.
    let operand = node.named_child(0)?;
    shape_of_expression(operand, ctx)
}

/// Subscript / indexing: `x[0]`, `x[i:j]`, `x[:, :d]`, `x[..., None]`.
///
/// Walks the index list against the value's shape, numpy-style:
/// - integer / scalar index → drops that axis (rank −1);
/// - slice → keeps the axis (dim narrowed where computable, else approximated
///   by the original axis — preserves rank);
/// - `None` / newaxis → inserts a size-1 axis;
/// - `...` (ellipsis) → expands to the un-indexed middle axes, unchanged;
/// - axes past the last index are kept unchanged (implicit full slices).
///
/// Advanced/boolean array indexing is approximated as a single axis-drop —
/// the common `x[i]` case is exact; fancy indexing is rare in model code.
fn shape_of_subscript(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    let value_node = node.child_by_field_name("value")?;
    let input_shape = shape_of_expression(value_node, ctx)?;
    let rank = input_shape.len();

    // Index nodes are every named child except the value.
    let mut indices = Vec::new();
    for i in 0..node.named_child_count() {
        let child = node.named_child(i as u32)?;
        if child.id() != value_node.id() {
            indices.push(child);
        }
    }

    // Axes consumed by indices that map onto an input axis (everything except
    // `None`/newaxis and the ellipsis itself). The ellipsis fills the rest.
    let consuming = indices
        .iter()
        .filter(|n| !matches!(n.kind(), "none" | "ellipsis"))
        .count();
    let ellipsis_axes = rank.saturating_sub(consuming);

    let mut out: Vec<String> = Vec::new();
    let mut axis = 0usize;

    for idx in &indices {
        match idx.kind() {
            "ellipsis" => {
                for _ in 0..ellipsis_axes {
                    out.push(input_shape.get(axis)?.clone());
                    axis += 1;
                }
            }
            "none" => out.push("1".to_string()),
            "slice" => {
                let orig = input_shape.get(axis)?;
                let text = idx.utf8_text(ctx.text.as_bytes()).ok()?;
                out.push(slice_dim(text, orig));
                axis += 1;
            }
            // integer / identifier / other scalar index → drop the axis.
            _ => {
                input_shape.get(axis)?; // bounds-check: over-indexing → None
                axis += 1;
            }
        }
    }

    // Trailing axes not addressed by any index stay as implicit full slices.
    while axis < rank {
        out.push(input_shape[axis].clone());
        axis += 1;
    }

    Some(out)
}

/// Resulting dim for a single sliced axis. Computes the length when it's
/// obvious (`:n` → `n`; numeric `a:b` → `b-a`), otherwise approximates by the
/// original axis so rank is preserved.
fn slice_dim(slice_text: &str, orig_dim: &str) -> String {
    let parts: Vec<&str> = slice_text.trim().split(':').collect();
    let start = parts.first().copied().unwrap_or("").trim();
    let stop = parts.get(1).copied().unwrap_or("").trim();
    let step = parts.get(2).copied().unwrap_or("").trim();

    // Full slice (`:` / `::`) → unchanged.
    if start.is_empty() && stop.is_empty() && step.is_empty() {
        return orig_dim.to_string();
    }
    // No step, known stop:
    if step.is_empty() && !stop.is_empty() {
        if start.is_empty() {
            return stop.to_string(); // `:n` → n
        }
        if let (Ok(a), Ok(b)) = (start.parse::<i64>(), stop.parse::<i64>()) {
            return (b - a).max(0).to_string(); // numeric `a:b`
        }
    }
    // Stepped / symbolic-bounded / start-only: approximate by the original axis.
    orig_dim.to_string()
}

/// Find the operator text between left and right children of a binary_operator node.
fn find_binary_op_text<'a>(
    node: Node<'a>,
    left_node: &Node<'a>,
    right_node: &Node<'a>,
    text: &str,
) -> Option<String> {
    for i in 0..node.child_count() {
        let child = node.child(i as u32)?;
        if !child.is_named()
            && child.start_byte() > left_node.end_byte()
            && child.end_byte() < right_node.start_byte()
            && let Ok(t) = child.utf8_text(text.as_bytes()) {
                return Some(t.trim().to_string());
            }
    }
    None
}

/// Binary operator: applies the matmul or elementwise shape rule to the
/// operand nodes directly.
/// A binary-op operand: a provable scalar (`2`, `math.sqrt(k)`, a name only
/// ever bound to scalars) or an array with a resolved shape. Scalars
/// broadcast trivially against any array. An operand that is neither stays
/// unknown: it could be an array that broadcasts.
enum BinopOperand {
    Scalar,
    Shaped(Vec<String>),
}

fn binop_operand(node: Node, scope_idx: usize, ctx: &mut ShapeCtx) -> Option<BinopOperand> {
    let shaped = match node.kind() {
        "integer" | "float" => return Some(BinopOperand::Scalar),
        "identifier" => {
            let name = node.utf8_text(ctx.text.as_bytes()).ok()?;
            ctx.resolve_shape(name, scope_idx)
        }
        // Nested expressions (calls, parenthesized binops, unary ops, …)
        // recurse through the main evaluator.
        _ => shape_of_expression(node, ctx),
    };
    match shaped {
        Some(shape) => Some(BinopOperand::Shaped(shape)),
        None => is_scalar_expr(node, ctx, &mut Vec::new()).then_some(BinopOperand::Scalar),
    }
}

/// How one scope binds one name.
enum NameBinding<'t> {
    /// The scope never binds the name, so a read looks further out.
    Unbound,
    /// Bound in a way that says nothing about the value (loop target, `with`,
    /// import, `global`, tuple unpacking, a non-scalar parameter, ...).
    Opaque,
    /// Bound only by plain assignments of these expressions, plus an
    /// annotated scalar parameter when `scalar_param`.
    Values(Vec<Node<'t>>),
}

/// Collects the expressions assigned to `name` in `node`'s own scope. False
/// when any other form binds it. Nested `def`, `class` and `lambda` bodies are
/// not entered; a nested `def` or `class` of that name binds it here.
fn collect_name_values<'t>(
    node: Node<'t>,
    name: &str,
    text: &str,
    values: &mut Vec<Node<'t>>,
    bound: &mut bool,
) -> bool {
    let mentions = |n: Node| mentions_name(n, text, &|other| other == name);
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "function_definition" | "class_definition" => {
                if child
                    .child_by_field_name("name")
                    .is_some_and(|n| n.utf8_text(text.as_bytes()) == Ok(name))
                {
                    return false;
                }
                continue;
            }
            "lambda" => continue,
            "import_from_statement"
                if child.named_children(&mut child.walk()).any(|c| c.kind() == "wildcard_import") =>
            {
                return false;
            }
            "global_statement" | "nonlocal_statement" | "import_statement"
            | "import_from_statement" | "future_import_statement" | "delete_statement"
            | "case_pattern" | "type_alias_statement" => {
                if mentions(child) {
                    return false;
                }
                continue;
            }
            "assignment" | "augmented_assignment" | "for_statement" => {
                if let Some(left) = child.child_by_field_name("left") {
                    let mut targets = HashSet::new();
                    insert_target_names(left, text, &mut targets);
                    if targets.contains(name) {
                        let plain = left.kind() == "identifier" && child.kind() != "for_statement";
                        let value = child.child_by_field_name("right");
                        match value {
                            Some(value) if plain => {
                                values.push(value);
                                *bound = true;
                            }
                            _ => return false,
                        }
                    }
                }
            }
            "named_expression" | "as_pattern" => {
                let target = child
                    .child_by_field_name("name")
                    .or_else(|| child.child_by_field_name("alias"));
                if target.is_some_and(mentions) {
                    return false;
                }
            }
            _ => {}
        }
        if !collect_name_values(child, name, text, values, bound) {
            return false;
        }
    }
    true
}

/// Binding of `name` in `owner`, a `function_definition` (scope `scope_idx`)
/// or the `module` (scope 0).
fn name_binding<'t>(
    owner: Node<'t>,
    scope_idx: usize,
    name: &str,
    ctx: &ShapeCtx,
) -> NameBinding<'t> {
    if scope_idx == 0 {
        return module_binding(owner, name, ctx);
    }
    let mut values = Vec::new();
    let mut bound = false;
    if ctx.links.poisoned[scope_idx].contains(name) {
        return NameBinding::Opaque;
    }
    if ctx.scopes[scope_idx].all_params.iter().any(|param| param == name) {
        // A scalar-typed parameter is seeded rank 0 (see `extract_jaxtyping_shapes`).
        if ctx.original_shapes[scope_idx].get(name).is_none_or(|shape| !shape.is_empty()) {
            return NameBinding::Opaque;
        }
        bound = true;
    }
    let Some(body) = owner.child_by_field_name("body") else {
        return NameBinding::Opaque;
    };
    if !collect_name_values(body, name, ctx.text, &mut values, &mut bound) {
        return NameBinding::Opaque;
    }
    if !bound {
        return NameBinding::Unbound;
    }
    NameBinding::Values(values)
}

/// How the module scope binds the names it binds: what `collect_name_values`
/// finds for one name, found for all of them in one walk, so that a file with
/// many globals is read once, not once for each.
#[derive(Default)]
struct ModuleIndex {
    /// `from m import *` may bind any name.
    wildcard: bool,
    /// Names bound in a way that says nothing about the value.
    opaque: HashSet<String>,
    /// Names bound only by plain assignments, and the expressions assigned.
    values: HashMap<String, Vec<NodeRef>>,
}

fn index_module(node: Node, text: &str, index: &mut ModuleIndex) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "function_definition" | "class_definition" => {
                if let Some(Ok(name)) = child
                    .child_by_field_name("name")
                    .map(|n| n.utf8_text(text.as_bytes()))
                {
                    index.opaque.insert(name.to_string());
                }
                continue;
            }
            "lambda" => continue,
            "import_from_statement"
                if child.named_children(&mut child.walk()).any(|c| c.kind() == "wildcard_import") =>
            {
                index.wildcard = true;
                continue;
            }
            "global_statement" | "nonlocal_statement" | "import_statement"
            | "import_from_statement" | "future_import_statement" | "delete_statement"
            | "case_pattern" | "type_alias_statement" => {
                insert_identifiers(child, text, &mut index.opaque);
                continue;
            }
            "assignment" | "augmented_assignment" | "for_statement" => {
                if let Some(left) = child.child_by_field_name("left") {
                    let mut targets = HashSet::new();
                    insert_target_names(left, text, &mut targets);
                    let plain = left.kind() == "identifier" && child.kind() != "for_statement";
                    let value = child.child_by_field_name("right").filter(|_| plain);
                    for target in targets {
                        match value {
                            Some(value) => index.values.entry(target).or_default().push(NodeRef::of(value)),
                            None => {
                                index.opaque.insert(target);
                            }
                        }
                    }
                }
            }
            "named_expression" | "as_pattern" => {
                if let Some(target) = child
                    .child_by_field_name("name")
                    .or_else(|| child.child_by_field_name("alias"))
                {
                    insert_identifiers(target, text, &mut index.opaque);
                }
            }
            _ => {}
        }
        index_module(child, text, index);
    }
}

fn module_binding<'t>(root: Node<'t>, name: &str, ctx: &ShapeCtx) -> NameBinding<'t> {
    let index = ctx.module_index.get_or_init(|| {
        let mut index = ModuleIndex::default();
        index_module(root, ctx.text, &mut index);
        index
    });
    if index.wildcard || index.opaque.contains(name) {
        return NameBinding::Opaque;
    }
    let Some(refs) = index.values.get(name) else {
        return NameBinding::Unbound;
    };
    let globals = ctx.global_names.get_or_init(|| {
        let mut names = HashSet::new();
        if ctx.text.contains("global") {
            collect_global_names(root, ctx.text, &mut names);
        }
        names
    });
    if globals.contains(name) {
        return NameBinding::Opaque;
    }
    match refs.iter().map(|r| r.find(root)).collect::<Option<Vec<_>>>() {
        Some(values) => NameBinding::Values(values),
        None => NameBinding::Opaque,
    }
}

/// The names the `global` statements under `node` mention. A statement stands
/// in a block or a clause of a compound one, never in an expression, so the
/// walk leaves the expressions out.
fn collect_global_names(node: Node, text: &str, out: &mut HashSet<String>) {
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "global_statement" => insert_identifiers(n, text, out),
            "module" | "block" | "function_definition" | "class_definition"
            | "decorated_definition" | "if_statement" | "elif_clause" | "else_clause"
            | "for_statement" | "while_statement" | "try_statement" | "except_clause"
            | "except_group_clause" | "finally_clause" | "with_statement"
            | "match_statement" | "case_clause" => {
                let mut cursor = n.walk();
                stack.extend(n.named_children(&mut cursor));
            }
            _ => {}
        }
    }
}

/// Whether the identifier at `at` is bound only to provable scalars in the
/// scope that owns it: every binding is a plain assignment of a scalar, or a
/// scalar-typed parameter. Flow-insensitive on purpose, so a reassignment in
/// a loop or branch cannot make an earlier read wrong. `visiting` breaks
/// cycles (`e = e * 2`), which cannot add a non-scalar value.
fn name_is_scalar(
    at: Node,
    name: &str,
    ctx: &ShapeCtx,
    visiting: &mut Vec<(usize, String)>,
) -> bool {
    match nearest_binding(at, name, ctx) {
        (_, NameBinding::Unbound) => false,
        (owner, NameBinding::Opaque) => ctx.inner_scalars.contains(&(owner.id(), name.to_string())),
        (owner, NameBinding::Values(values)) => {
            let key = (owner.id(), name.to_string());
            if visiting.contains(&key) {
                return true;
            }
            visiting.push(key);
            let scalar = values.iter().all(|v| is_scalar_expr(*v, ctx, visiting));
            visiting.pop();
            scalar
        }
    }
}

/// The innermost scope around `at` that binds `name`, with its binding.
/// `Opaque` when a scope on the way is not modelled, `Unbound` when no scope
/// binds the name (it is a builtin, an import, or undefined).
fn nearest_binding<'t>(at: Node<'t>, name: &str, ctx: &ShapeCtx) -> (Node<'t>, NameBinding<'t>) {
    let mut in_function = false;
    let mut up = at.parent();
    while let Some(owner) = up {
        up = owner.parent();
        let scope_idx = match owner.kind() {
            "function_definition" => {
                in_function = true;
                let found = ctx
                    .scope_function_nodes
                    .iter()
                    .position(|f| f.is_some_and(|f| f.id() == owner.id()));
                let Some(found) = found else {
                    return (owner, NameBinding::Opaque);
                };
                found
            }
            // A class body is a scope of its own, which this does not model.
            "class_definition" if !in_function => return (owner, NameBinding::Opaque),
            "module" => 0,
            // A lambda or a comprehension that binds the name hides the
            // scopes around it. `inner_scalars` and `inner_arrays` say what
            // it holds while its body is evaluated.
            "lambda" | "list_comprehension" | "set_comprehension" | "dictionary_comprehension"
            | "generator_expression"
                if inner_scope_binds(owner, name, ctx.text) =>
            {
                return (owner, NameBinding::Opaque);
            }
            _ => continue,
        };
        match name_binding(owner, scope_idx, name, ctx) {
            NameBinding::Unbound => {}
            binding => return (owner, binding),
        }
    }
    (at, NameBinding::Unbound)
}

/// Whether `node` is provably a scalar: a number, a `math` function or
/// constant, `len`/`float`/`int`, `x.shape[<integer>]`, a numpy or jax
/// elementwise function of scalars, a name only ever bound to scalars, or
/// arithmetic on those. Everything else, including every unresolved name or
/// call, is not provable: it could be an array that broadcasts.
fn is_scalar_expr(
    node: Node,
    ctx: &ShapeCtx,
    visiting: &mut Vec<(usize, String)>,
) -> bool {
    let text = |n: Node| n.utf8_text(ctx.text.as_bytes()).unwrap_or("");
    match node.kind() {
        "integer" | "float" | "true" | "false" => true,
        "parenthesized_expression" => node
            .named_child(0)
            .is_some_and(|inner| is_scalar_expr(inner, ctx, visiting)),
        "unary_operator" => node
            .child_by_field_name("argument")
            .is_some_and(|operand| is_scalar_expr(operand, ctx, visiting)),
        "binary_operator" => {
            node.child_by_field_name("operator").is_some_and(|op| op.kind() != "@")
                && ["left", "right"].iter().all(|side| {
                    node.child_by_field_name(side)
                        .is_some_and(|operand| is_scalar_expr(operand, ctx, visiting))
                })
        }
        "identifier" => name_is_scalar(node, text(node), ctx, visiting),
        "attribute" => {
            let resolved = resolve_call_target(text(node), ctx.imports.at(node.start_byte()));
            let parts: Vec<&str> = resolved.parts.iter().map(String::as_str).collect();
            matches!(
                parts.as_slice(),
                ["math" | "numpy" | "torch", "pi" | "e" | "inf" | "nan"]
                    | ["jax", "numpy", "pi" | "e" | "inf" | "nan"]
                    | ["math", "tau"]
            )
        }
        "subscript" => {
            let index = node.named_child(1).filter(|_| node.named_child_count() == 2);
            let is_integer = |n: Node| n.kind() == "integer";
            node.child_by_field_name("value").is_some_and(|value| {
                value.kind() == "attribute"
                    && value
                        .child_by_field_name("attribute")
                        .is_some_and(|attr| text(attr) == "shape")
            }) && index.is_some_and(|index| {
                is_integer(index)
                    || index.kind() == "unary_operator"
                        && index.child_by_field_name("argument").is_some_and(is_integer)
            })
        }
        "call" => {
            let Some(function) = node.child_by_field_name("function") else {
                return false;
            };
            let function_text = text(function);
            let resolved = resolve_call_target(function_text, ctx.imports.at(node.start_byte()));
            if function.kind() == "identifier" {
                return matches!(function_text, "len" | "float" | "int")
                    && resolved.parts == [function_text]
                    && matches!(nearest_binding(node, function_text, ctx).1, NameBinding::Unbound);
            }
            let Some((name, module)) = resolved.parts.split_last() else {
                return false;
            };
            // Python's `math` functions return a float for any argument.
            if module == ["math"] {
                return matches!(
                    name.as_str(),
                    "sqrt" | "exp" | "expm1" | "log" | "log2" | "log10" | "log1p" | "sin" | "cos"
                        | "tan" | "asin" | "acos" | "atan" | "atan2" | "sinh" | "cosh" | "tanh"
                        | "asinh" | "acosh" | "atanh" | "pow" | "fabs" | "hypot" | "degrees"
                        | "radians" | "erf" | "erfc" | "gamma" | "lgamma"
                );
            }
            // numpy / jax elementwise functions of scalars return a 0-d value.
            // torch's reject Python scalars, so they stay unknown, and so do
            // the axis-reducing `jax.nn` ones, which reject a 0-d input.
            module.first().map(String::as_str) != Some("torch")
                && is_shape_preserving_call(&resolved)
                && !matches!(name.as_str(), "softmax" | "log_softmax" | "standardize")
                && node.child_by_field_name("arguments").is_some_and(|arguments| {
                    arguments.named_child_count() > 0
                        && arguments.named_children(&mut arguments.walk()).all(|arg| {
                            !matches!(arg.kind(), "keyword_argument" | "list_splat" | "dictionary_splat")
                                && is_scalar_expr(arg, ctx, visiting)
                        })
                })
        }
        _ => false,
    }
}

fn binary_mismatch_axes(
    left: &[String],
    right: &[String],
    op: BinaryOp,
) -> Option<(usize, usize)> {
    if matches!(op, BinaryOp::MatMul) {
        if left.len() < 2 || right.len() < 2 {
            return None;
        }
        let left_batch_len = left.len() - 2;
        let right_batch_len = right.len() - 2;
        for k in (1..=left_batch_len.max(right_batch_len)).rev() {
            let left_axis = left_batch_len.checked_sub(k);
            let right_axis = right_batch_len.checked_sub(k);
            if let (Some(left_axis), Some(right_axis)) = (left_axis, right_axis) {
                let left_dim = &left[left_axis];
                let right_dim = &right[right_axis];
                if !dims_canonically_equal(left_dim, right_dim)
                    && left_dim != "1"
                    && right_dim != "1"
                {
                    return Some((left_axis, right_axis));
                }
            }
        }
        let left_axis = left.len() - 1;
        let right_axis = right.len() - 2;
        return (!dims_canonically_equal(&left[left_axis], &right[right_axis]))
            .then_some((left_axis, right_axis));
    }

    let rank = left.len().max(right.len());
    for i in 0..rank {
        let left_axis = (i + left.len()).checked_sub(rank);
        let right_axis = (i + right.len()).checked_sub(rank);
        if let (Some(left_axis), Some(right_axis)) = (left_axis, right_axis) {
            let left_dim = &left[left_axis];
            let right_dim = &right[right_axis];
            if !dims_canonically_equal(left_dim, right_dim)
                && left_dim != "1"
                && right_dim != "1"
            {
                return Some((left_axis, right_axis));
            }
        }
    }
    None
}

fn annotation_dimension_range(
    scope: &FunctionShapeScope,
    operand: Node,
    axis: usize,
    value: &str,
    text: &str,
) -> Option<tree_sitter::Range> {
    if operand.kind() != "identifier" {
        return None;
    }
    let binding = operand.utf8_text(text.as_bytes()).ok()?;
    scope
        .dimension_sites
        .iter()
        .rev()
        .find(|site| {
            site.binding.as_deref() == Some(binding)
                && site.axis == axis
                && site.range.end_byte <= operand.start_byte()
                && dims_canonically_equal(&site.value, value)
        })
        .map(|site| site.range)
}

fn transpose_fix(
    node: Node,
    right_node: Node,
    left_shape: &[String],
    right_shape: &[String],
    left_name: &str,
    right_name: &str,
) -> Option<ShapeFix> {
    if right_shape.len() != 2
        || !matches!(
            right_node.kind(),
            "identifier" | "attribute" | "call" | "subscript" | "parenthesized_expression"
        )
    {
        return None;
    }
    let mut transposed = right_shape.to_vec();
    transposed.reverse();
    matches!(
        apply_matmul_shape(left_shape, &transposed, left_name, right_name),
        Ok(Some(_))
    )
    .then_some(ShapeFix::AppendTranspose {
        expression_range: node.range(),
        operand_range: right_node.range(),
    })
}

fn shape_of_binary_operator(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    let left_node = node.child_by_field_name("left")?;
    let right_node = node.child_by_field_name("right")?;

    // Determine the operator text.
    let op_text = find_binary_op_text(node, &left_node, &right_node, ctx.text)?;

    let op = match op_text.as_str() {
        "@" => BinaryOp::MatMul,
        "+" => BinaryOp::Add,
        "-" => BinaryOp::Sub,
        "*" => BinaryOp::Mul,
        "/" => BinaryOp::Div,
        _ => return None,
    };

    let scope_idx = ctx.links.scope_index.lookup(node.start_byte())?;
    let left = binop_operand(left_node, scope_idx, ctx)?;
    let right = binop_operand(right_node, scope_idx, ctx)?;

    let (left_shape, right_shape) = match (left, right) {
        // Scalar literals broadcast: the result keeps the array side's shape.
        // (Scalar @ array is invalid Python; skip silently.)
        (BinopOperand::Scalar, BinopOperand::Shaped(s))
        | (BinopOperand::Shaped(s), BinopOperand::Scalar) => {
            return (op != BinaryOp::MatMul).then_some(s);
        }
        (BinopOperand::Scalar, BinopOperand::Scalar) => return None,
        (BinopOperand::Shaped(l), BinopOperand::Shaped(r)) => (l, r),
    };

    let left_name = left_node.utf8_text(ctx.text.as_bytes()).unwrap_or("?");
    let right_name = right_node.utf8_text(ctx.text.as_bytes()).unwrap_or("?");
    let result = match op {
        BinaryOp::MatMul => apply_matmul_shape(&left_shape, &right_shape, left_name, right_name),
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
            apply_elementwise_shape(&left_shape, &right_shape, op)
        }
    };

    match result {
        Ok(Some(shape)) => Some(shape),
        Ok(None) => None,
        Err(message) => {
            // We don't have a natural variable name here — use the full
            // expression text as the "variable" in the error.
            let var_text = node.utf8_text(ctx.text.as_bytes()).unwrap_or("?").to_string();
            // Both operand nodes are in hand here, so attach the right-hand
            // operand's own range as related information (convention: the
            // right operand is "the other one" relative to the left, which
            // the mismatch message already names first).
            let related_message = format!(
                "other operand `{}`: shape [{}]",
                right_name,
                right_shape.join(", ")
            );
            let (primary_range, related_range) = binary_mismatch_axes(
                &left_shape,
                &right_shape,
                op,
            )
            .and_then(|(left_axis, right_axis)| {
                let scope = &ctx.scopes[scope_idx];
                let left_range = annotation_dimension_range(
                    scope,
                    left_node,
                    left_axis,
                    &left_shape[left_axis],
                    ctx.text,
                )?;
                let right_range = annotation_dimension_range(
                    scope,
                    right_node,
                    right_axis,
                    &right_shape[right_axis],
                    ctx.text,
                )?;
                Some((left_range, right_range))
            })
            .unwrap_or((node.range(), right_node.range()));
            let mut error = ShapeError::mismatch(var_text, message, primary_range)
                .with_related(related_range, related_message);
            if matches!(op, BinaryOp::MatMul)
                && let Some(fix) = transpose_fix(
                    node,
                    right_node,
                    &left_shape,
                    &right_shape,
                    left_name,
                    right_name,
                )
            {
                error = error.with_fix(fix);
            }
            ctx.errors.push(error);
            None
        }
    }
}

/// Which array library the file uses, judged from its imports alone.
#[derive(PartialEq)]
enum FileBackend {
    Torch,
    NumpyLike,
    Unknown,
}

fn file_backend(import_map: &HashMap<String, ImportPath>) -> FileBackend {
    let (mut torch, mut other) = (false, false);
    for import in import_map.values() {
        match import.module.first().unwrap_or(&import.name).as_str() {
            "torch" => torch = true,
            "jax" | "numpy" | "flax" | "equinox" | "haiku" | "optax" => other = true,
            _ => {}
        }
    }
    match (torch, other) {
        (true, false) => FileBackend::Torch,
        (false, true) => FileBackend::NumpyLike,
        _ => FileBackend::Unknown,
    }
}

/// `x.transpose(i, j)` swaps two dims in torch but is a full axes permutation
/// in numpy/jax. `dim0`/`dim1` keywords only exist in torch. With two
/// positional dims and no known backend, the result is kept only when both
/// readings give the same shape.
fn apply_method_call_for_backend(
    known: &KnownFunction,
    receiver: &str,
    args: &[CallArgument],
    import_map: &HashMap<String, ImportPath>,
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    if matches!(known, KnownFunction::Max | KnownFunction::Min) {
        return apply_max_min_method_for_backend(known, receiver, args, import_map, shapes);
    }
    if !matches!(known, KnownFunction::Transpose) {
        return apply_method_call(known, receiver, args, shapes);
    }
    let torch_keywords = args.iter().any(|arg| {
        matches!(arg, CallArgument::Keyword { name, .. } if name == "dim0" || name == "dim1")
    });
    if torch_keywords {
        return apply_method_call(&KnownFunction::SwapAxes, receiver, args, shapes);
    }
    let positional_count = args
        .iter()
        .filter(|a| matches!(a, CallArgument::Positional { .. }))
        .count();
    if positional_count != 2 {
        return apply_method_call(known, receiver, args, shapes);
    }
    match file_backend(import_map) {
        FileBackend::Torch => apply_method_call(&KnownFunction::SwapAxes, receiver, args, shapes),
        FileBackend::NumpyLike => apply_method_call(known, receiver, args, shapes),
        FileBackend::Unknown => {
            let swapped = apply_method_call(&KnownFunction::SwapAxes, receiver, args, shapes);
            let permuted = apply_method_call(known, receiver, args, shapes);
            match (swapped, permuted) {
                (Ok(Some(a)), Ok(Some(b))) if a == b => Ok(Some(a)),
                _ => Ok(None),
            }
        }
    }
}

/// `x.max(y)` broadcasts and `x.max(dim)` returns a `(values, indices)`
/// namedtuple in torch, while numpy/jax `x.max(axis)` reduces to an array.
/// With no known backend, the result is kept only when both readings agree.
fn apply_max_min_method_for_backend(
    known: &KnownFunction,
    receiver: &str,
    args: &[CallArgument],
    import_map: &HashMap<String, ImportPath>,
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    match file_backend(import_map) {
        FileBackend::Torch => apply_method_call(&KnownFunction::TorchMaxMin, receiver, args, shapes),
        FileBackend::NumpyLike => apply_method_call(known, receiver, args, shapes),
        FileBackend::Unknown => {
            let torch = apply_method_call(&KnownFunction::TorchMaxMin, receiver, args, shapes);
            let numpy = apply_method_call(known, receiver, args, shapes);
            match (torch, numpy) {
                (Ok(Some(a)), Ok(Some(b))) if a == b => Ok(Some(a)),
                _ => Ok(None),
            }
        }
    }
}

/// Resolve a `call` node's shape.
///
/// This is the main dispatch point that handles:
/// 1. Chained method calls (e.g. `x.reshape(3,4).sum(axis=1)`)
/// 2. Free function calls (e.g. `jnp.exp(...)`)
/// 3. Layer applications
/// 4. vmap bindings and calls
/// 5. User-defined function propagation
/// 6. Known-function resolution
fn shape_of_call(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    let call_byte = node.start_byte();
    let scope_idx = ctx.links.scope_index.lookup(call_byte)?;

    let func_node = node.child_by_field_name("function")?;
    let args_node = node.child_by_field_name("arguments")?;

    if let Some(lambda) = called_lambda(func_node) {
        return shape_of_direct_lambda(lambda, args_node, scope_idx, ctx);
    }
    if func_node.kind() == "identifier"
        && let Some(shape) = shape_of_sum(node, scope_idx, ctx)
    {
        return Some(shape);
    }
    if func_node.kind() == "identifier"
        && let Some(result) = shape_of_loop_module_call(node, func_node, args_node, scope_idx, ctx)
    {
        return result;
    }
    if func_node.kind() == "subscript"
        && let Some(result) = shape_of_indexed_module_call(node, func_node, args_node, scope_idx, ctx)
    {
        return result;
    }

    // ── Call-of-a-call ──
    // `jax.vmap(self.layer)(x)` (inline vmap, issue #35) or a layer
    // constructed and applied in one expression like `nn.Dense(64)(x)`
    // (the dominant flax style).
    if func_node.kind() == "call" {
        if let Some(shape) = shape_of_inline_vmap(func_node, args_node, scope_idx, ctx) {
            return Some(shape);
        }
        return shape_of_inline_layer(func_node, args_node, scope_idx, ctx)
            .or_else(|| shape_of_inline_module(func_node, args_node, scope_idx, ctx));
    }

    // ── Chained method calls ──
    // In tree-sitter Python, `x.reshape(3,4).sum(axis=1)` parses as a
    // single `call` node whose `function` is an `attribute` like
    // `x.reshape(3,4).sum`.  The attribute's `object` is a `call`
    // (`x.reshape(3,4)`) and its `attribute` is `sum`.
    //
    // We handle this by resolving the object (receiver) recursively,
    // then treating the outer call as a method call on that receiver.

    if func_node.kind() == "attribute" {
        let attr_node = func_node;
        let obj_node = attr_node.child_by_field_name("object")?;
        let method_name_node = attr_node.child_by_field_name("attribute")?;
        let method_name = method_name_node
            .utf8_text(ctx.text.as_bytes())
            .ok()?
            .to_string();

        // Direct self-attribute layer call: `self.conv1(x)` / `self.fc(h)`,
        // the dominant equinox/torch forward-method style. `self.<attr>`
        // resolves to a layer built in `__init__`; apply it directly (the
        // layer rule already handles its own input rank).
        if obj_node.kind() == "identifier"
            && obj_node.utf8_text(ctx.text.as_bytes()).ok()? == "self"
            && let Some(layer) = ctx
                .self_attr_layer_at(&method_name, attr_node.start_byte())
        {
            let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
            let args = resolve_call_args(raw_args, args_node, scope_idx, ctx)?;

            // MultiheadAttention's real return is an `(output, weights)`
            // tuple; `apply_layer_application` can't express that and
            // always returns `Ok(None)` for it (tuple LHS is handled
            // separately in `tuple_rhs_shapes`). A single-assignment call
            // site (`out = self.attn(q, k, v)`) still wants *something* —
            // bind the primary output, which has the query's shape (same
            // rule as `tuple_rhs_shapes`'s `attn_out`).
            if matches!(layer, LayerKind::MultiheadAttention { .. }) {
                let CallArgument::Positional { value: query } = args.first()?.clone() else {
                    return None;
                };
                return ctx.resolve_shape(&query, scope_idx);
            }

            let CallArgument::Positional { value: input } = args.first()?.clone() else {
                return None;
            };
            let application = LayerApplication {
                variable: String::new(),
                layer: format!("self.{method_name}"),
                input,
                extra_inputs: extra_layer_inputs(&args, &layer),
                kind: layer,
                range: args_node.range(),
            };
            return match apply_layer_application(&application, &ctx.scope_shapes(scope_idx)) {
                Ok(Some(output)) => Some(output),
                Ok(None) => None,
                Err(message) => {
                    ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
                    None
                }
            };
        }

        // Direct call of a user module held in a field: `self.blk(x)`. A field
        // that is written but not as one known module never falls back to a
        // function of the same name.
        if obj_node.kind() == "identifier"
            && obj_node.utf8_text(ctx.text.as_bytes()).ok()? == "self"
            && let Some(field) = ctx
                .submodule_at(&method_name, attr_node.start_byte())
                .map(|field| field.module.clone())
        {
            let field = field?;
            if let Field::One(Module::Imported(module)) = &field {
                // The arguments are evaluated once the callee is a class.
                imported_class(module, ctx)?;
            }
            let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
            let args = resolve_call_args(raw_args, args_node, scope_idx, ctx)?;
            let display = format!("self.{method_name}");
            let range = args_node.range();
            return match field {
                Field::One(module) => {
                    apply_module(&module, &display, &args, call_byte, scope_idx, range, ctx)
                }
                Field::Many(sequence) => {
                    apply_sequence_call(&sequence, &display, &args, call_byte, scope_idx, range, ctx)
                }
            };
        }

        // A method of the class of the call: `self.m(x)`, `cls.m(x)`,
        // `super().m(x)`. A `super()` receiver is never an array.
        let is_super = is_super_call(obj_node, ctx.text);
        if method_name != "__init__"
            && let Some(callee_idx) = ctx.method_callee(obj_node, is_super, &method_name, call_byte)
        {
            let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
            let args = resolve_call_args(raw_args, args_node, scope_idx, ctx)?;
            let display = ctx.text[func_node.byte_range()].to_string();
            let call = UserCall { name: &method_name, display, args: &args, args_node, call_byte };
            return apply_user_function(callee_idx, call, scope_idx, ctx).flatten();
        }
        if is_super {
            return None;
        }

        // Distinguish between:
        //   (a) Chained method call: x.reshape(3,4).sum(axis=1)
        //       where obj_node is itself a `call` node.
        //   (b) Qualified free function: jnp.reshape(x, (5,3))
        //       where obj_node is an identifier (jnp).
        //   (c) Simple method call: x.reshape(3, 4)
        //       where obj_node is an identifier (x).
        //
        // For (a), we resolve the receiver recursively.
        // For (b), we fall through to free-function resolution.
        // For (c), we dispatch to classify_method_call.

        // (a) Method call on a sub-expression we can shape:
        //       - a chained call: `x.reshape(3,4).sum(axis=1)` (obj is a call)
        //       - a self attribute: `self.A_log.astype(...)` (obj is an
        //         attribute, issue #31)
        //     Resolve the receiver recursively; if it has a shape, treat the
        //     outer call as a method call on the result.
        if obj_node.kind() == "call" || obj_node.kind() == "attribute" {
            if let Some(receiver_shape) = shape_of_expression(obj_node, ctx) {
                let receiver_name = ctx.bind_synthetic(
                    receiver_shape,
                    scope_idx,
                    obj_node.start_byte(),
                    ctx.text.get(obj_node.byte_range()).unwrap_or_default(),
                );
                let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
                let args = resolve_call_args(raw_args, args_node, scope_idx, ctx)?;

                if let Some(known) = classify_method_call(&method_name) {
                    let args = if matches!(known, KnownFunction::Clamp) {
                        without_scalar_bounds(args, args_node, false, ctx)
                    } else {
                        args
                    };
                    let result = apply_method_call_for_backend(&known, &receiver_name, &args, ctx.imports.file(), &ctx.scope_shapes(scope_idx));
                    return match result {
                        Ok(Some(shape)) => Some(shape),
                        Ok(None) => None,
                        Err(message) => {
                            ctx.errors.push(ShapeError::mismatch(method_name, message, args_node.range()));
                            None
                        }
                    };
                }
                return None;
            }
            // Receiver has no shape. For a `call` receiver there's nothing
            // else to try. For an `attribute` (e.g. `jax.nn` in
            // `jax.nn.softplus(x)`), fall through to free-function resolution.
            if obj_node.kind() == "call" {
                return None;
            }
        }

        // (c) Simple method call: x.reshape(3, 4)
        //     where obj_node is an identifier that is NOT an import alias.
        if obj_node.kind() == "identifier" {
            let receiver_name = obj_node
                .utf8_text(ctx.text.as_bytes())
                .ok()?
                .to_string();

            // If the receiver is an import alias (like jnp), this is a
            // qualified free function — fall through.
            if !is_imported(ctx.imports.at(obj_node.start_byte()), &receiver_name) {
                let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
                let args = resolve_call_args(raw_args, args_node, scope_idx, ctx)?;

                if let Some(known) = classify_method_call(&method_name) {
                    let args = if matches!(known, KnownFunction::Clamp) {
                        without_scalar_bounds(args, args_node, false, ctx)
                    } else {
                        args
                    };
                    let result = apply_method_call_for_backend(
                        &known,
                        &receiver_name,
                        &args,
                        ctx.imports.file(),
                        &ctx.scope_shapes(scope_idx),
                    );
                    return match result {
                        Ok(Some(shape)) => Some(shape),
                        Ok(None) => None,
                        Err(message) => {
                            ctx.errors.push(ShapeError::mismatch(receiver_name, message, args_node.range()));
                            None
                        }
                    };
                }

                // Check if receiver is a layer — layers are called like layer(x)
                // but they use identifier func_node, not attribute. Skip here.
            }
        }

        // For (b) and remaining cases: fall through to free-function
        // resolution. The `target` string includes the full qualified name.
    }

    // ── Free function call ──
    // func_node is an identifier (e.g. `my_func`) or an attribute chain
    // (e.g. `jnp.exp`, `jax.nn.softplus`).

    let target = func_node
        .utf8_text(ctx.text.as_bytes())
        .ok()?
        .to_string();

    // Resolve arguments — but first, recursively evaluate any inline
    // expression arguments (nested calls, unary ops, etc.) so they have
    // shapes in the scope.
    let imports = ctx.imports.at(func_node.start_byte());
    let resolved = resolve_call_target(&target, imports);
    let known = classify_known_function(&resolved);
    let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
    let args = if matches!(known, Some(KnownFunction::LaxPolynomial)) {
        let mut args = raw_args;
        if let Some((i, input)) = args.iter_mut().enumerate().find(|(_, arg)| {
            matches!(arg, CallArgument::Positional { .. })
                || matches!(arg, CallArgument::Keyword { name, .. } if name == "x")
        }) {
            *input = resolve_call_arg(input.clone(), args_node.named_child(i as u32)?, scope_idx, ctx)?;
        }
        args
    } else if matches!(
        known.as_ref().and_then(sequence_kind),
        Some(SequenceArg::Elements | SequenceArg::Operands)
    ) {
        resolve_call_args_keeping_sequences(raw_args, args_node, scope_idx, ctx)?
    } else {
        resolve_call_args(raw_args, args_node, scope_idx, ctx)?
    };

    // 1. Layer check
    if let Some(record) = find_scoped_layer(ctx.layer_records, ctx.scopes, &ctx.links.scope_index, call_byte, &target) {
        let kind = ctx.layer_at(&record.kind, record.byte_position);
        let Some(CallArgument::Positional { value: input }) = args.first().cloned() else {
            return None;
        };
        // Use the LHS variable name if available, or a synthetic one.
        // For now we don't have the LHS name here; the caller
        // (propagate_calls) handles the binding.
        let application = LayerApplication {
            variable: String::new(), // filled by caller
            layer: target.clone(),
            input,
            extra_inputs: extra_layer_inputs(&args, &kind),
            kind,
            range: args_node.range(),
        };
        let result = apply_layer_application(&application, &ctx.scope_shapes(scope_idx));
        ctx.applications.push(application.clone());
        match result {
            Ok(Some(output)) => return Some(output),
            Ok(None) => return None,
            Err(message) => {
                // We'll record the error with a variable name from the caller.
                // For now use the target as placeholder — caller replaces it.
                ctx.errors.push(ShapeError::mismatch(target.clone(), message, application.range));
                return None;
            }
        }
    }

    // 2. vmap-target check
    if let Some(info) = ctx.vmap_targets.get(&target).cloned() {
        let result = apply_vmap_call(
            &info,
            &args,
            &ctx.scope_shapes(scope_idx),
            ctx.links,
            ctx.scopes,
        );
        return match result {
            Ok(Some(shape)) => Some(shape),
            Ok(None) => None,
            Err(message) => {
                ctx.errors.push(ShapeError::mismatch(target, message, args_node.range()));
                None
            }
        };
    }

    // 2b. `jax.lax.map(f, xs)` — maps `f` over the leading axis of `xs`,
    // exactly like `vmap(f)(xs)` with `in_axes=out_axes=0`. Handled here
    // (rather than through `apply_known_function`) because resolving `f`
    // needs the callee's `FunctionShapeScope`/self-attr-layer machinery.
    {
        if matches!(known.as_ref(), Some(KnownFunction::LaxMap)) {
            let mut positionals = args.iter().filter_map(|a| match a {
                CallArgument::Positional { value } => Some(value.clone()),
                _ => None,
            });
            if let Some(lambda) = lambda_argument(args_node, 0)
                && matches!(args.first(), Some(CallArgument::Positional { .. }))
            {
                return map_lambda(lambda, args_node, &args[1..], scope_idx, ctx);
            }
            let callable = positionals.next()?;
            let rest: Vec<CallArgument> = args.iter().skip(1).cloned().collect();
            if let Some(attr) = callable.strip_prefix("self.")
                && let Some(layer) = ctx.self_attr_layer_at(attr, call_byte)
            {
                return apply_inline_vmap_layer(
                    (&callable, &layer),
                    &rest,
                    0,
                    0,
                    scope_idx,
                    args_node.range(),
                    ctx,
                );
            }
            if let Some(attr) = callable.strip_prefix("self.")
                && let Some(field) = ctx
                    .submodule_at(attr, call_byte)
                    .map(|field| field.module.clone())
            {
                let Some(Field::One(Module::Local(module))) = field else {
                    return None;
                };
                return apply_vmapped_submodule(&module, &callable, &rest, &[(0, 0)], scope_idx, args_node, ctx);
            }
            if !callable.contains('.') {
                let info = VmapInfo {
                    wrapped: callable,
                    in_axes: 0,
                    out_axes: 0,
                    site: call_byte,
                };
                return match apply_vmap_call(&info, &rest, &ctx.scope_shapes(scope_idx), ctx.links, ctx.scopes) {
                    Ok(shape) => shape,
                    Err(message) => {
                        ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
                        None
                    }
                };
            }
            return None;
        }
    }

    // 3. User-defined function propagation.
    if !target.contains('.')
        && let Some(callee_idx) = find_callee_scope(&target, call_byte, ctx.links, ctx.scopes)
    {
        let call = UserCall {
            name: &target,
            display: target.clone(),
            args: &args,
            args_node,
            call_byte,
        };
        if let Some(result) = apply_user_function(callee_idx, call, scope_idx, ctx) {
            return result;
        }
    }

    // 4. vmap *recording* (e.g. `vf = jax.vmap(f)`)
    if matches!(known.as_ref(), Some(KnownFunction::Vmap))
        && let Some(_info) = parse_vmap_call(&args, call_byte) {
            // propagate_calls records the vmap binding, so there is no shape here.
            return None;
        }

    // 5. Known-function check
    if let Some(known) = known {
        if matches!(known, KnownFunction::BroadcastArrays)
            || (matches!(known, KnownFunction::AtLeast1D | KnownFunction::AtLeast2D | KnownFunction::AtLeast3D)
                && args.len() > 1)
        {
            return None;
        }
        // Functions that consume a sequence must see every element: the
        // homogeneous-tuple approximation used for scan carries would
        // collapse `(x, x)` into one operand. Share this rule across backends.
        let args = match sequence_kind(&known) {
            Some(SequenceArg::Elements) => resolve_stack_args(args_node, scope_idx, ctx)?,
            Some(SequenceArg::Operands) if resolved.parts.first().map(String::as_str) == Some("torch") => resolve_einsum_args(args, args_node, scope_idx, ctx)?,
            Some(SequenceArg::Opaque) if first_arg_is_tuple(args_node) => return None,
            _ => args,
        };
        let args = if matches!(known, KnownFunction::Clamp) {
            without_scalar_bounds(args, args_node, true, ctx)
        } else {
            args
        };
        let result = apply_known_function(&known, &args, &ctx.scope_shapes(scope_idx));
        return match result {
            Ok(Some(shape)) => Some(shape),
            Ok(None) => None,
            Err(message) => {
                // `jax.lax.dot_general` is classified as `KnownFunction::Matmul`
                // but is only *approximated* as matmul (see `llm.txt`); a
                // mismatch here may be a false positive from that
                // approximation rather than a real shape bug. Real
                // `matmul`/`dot` calls (also `KnownFunction::Matmul`) don't
                // resolve to a `dot_general` target, so they stay `Mismatch`.
                let error = if resolved.parts.last().map(String::as_str) == Some("dot_general") {
                    ShapeError::approximation(target, message, args_node.range())
                } else {
                    ShapeError::mismatch(target, message, args_node.range())
                };
                ctx.errors.push(error);
                None
            }
        };
    }

    // 6. Elementwise / activation fallback (not in classify_known_function)
    //
    // Functions like jnp.exp, jnp.sin, jax.nn.relu, jax.nn.softplus, etc.
    // are shape-preserving (they return the input array's shape unchanged).
    // They're not in `classify_known_function` to avoid bloating the enum,
    // but we handle them here with a simple shape-preserving rule.
    if is_shape_preserving_call(&resolved) {
        if is_binary_elementwise_call(&resolved) {
            return binary_call_shape(node, &args, args_node, scope_idx, ctx);
        }
        let Some(CallArgument::Positional { value }) = args.first() else {
            return None;
        };
        return ctx
            .resolve_shape(value, scope_idx)
            .or_else(|| is_scalar_expr(node, ctx, &mut Vec::new()).then(Vec::new));
    }

    // 7. Cross-file user-defined function propagation. Same-file helpers are
    // handled in step 3 above; this extends the same bind-and-substitute
    // logic to helpers imported from another file on disk, e.g.
    // `from mylib.helpers import project` then `y = project(x)`. Tried last
    // (after known-function/elementwise) so the common case — calls into
    // jax/numpy/torch, which never resolve on disk — doesn't pay for a
    // filesystem walk on every call.
    if let Some(result) = apply_imported_user_function(
        &target,
        &args,
        &ctx.scope_shapes(scope_idx),
        imports,
        ctx.search_roots,
        ctx.read_file,
        ctx.max_depth,
        ctx.cache,
    ) {
        return match result {
            Ok(Some(shape)) => Some(shape),
            Ok(None) => None,
            Err(message) => {
                ctx.errors
                    .push(ShapeError::mismatch(target, message, args_node.range()));
                None
            }
        };
    }

    None
}

/// `jnp.maximum(a, b)` and the other two-operand elementwise functions
/// broadcast their operands, and a Python scalar broadcasts against anything.
/// A mismatch, a third positional argument, or an operand passed by keyword
/// stays unknown.
fn binary_call_shape(
    node: Node,
    args: &[CallArgument],
    args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let mut operands = args.iter().take_while(|arg| matches!(arg, CallArgument::Positional { .. }));
    let (Some(CallArgument::Positional { value: left }), Some(CallArgument::Positional { value: right }), None) =
        (operands.next(), operands.next(), operands.next())
    else {
        return None;
    };
    let operand = |value: &String, index: u32| {
        match ctx.resolve_shape(value, scope_idx) {
            Some(shape) => Some(Some(shape)),
            None => is_scalar_expr(args_node.named_child(index)?, ctx, &mut Vec::new()).then_some(None),
        }
    };
    match (operand(left, 0)?, operand(right, 1)?) {
        (Some(left), Some(right)) => apply_elementwise_shape(&left, &right, BinaryOp::Add).ok()?,
        (Some(shape), None) | (None, Some(shape)) => Some(shape),
        (None, None) => is_scalar_expr(node, ctx, &mut Vec::new()).then(Vec::new),
    }
}

/// Removes the `clip` / `clamp` bounds that are `None` or provable scalars,
/// which never change the input's shape. `skip_input` keeps the first
/// positional argument, the array being clipped, in a free-function call.
fn without_scalar_bounds(
    args: Vec<CallArgument>,
    args_node: Node,
    skip_input: bool,
    ctx: &ShapeCtx,
) -> Vec<CallArgument> {
    let mut seen_positional = false;
    args.into_iter()
        .enumerate()
        .filter(|(i, arg)| {
            let value_node = match arg {
                CallArgument::Positional { .. } => {
                    let is_input = skip_input && !seen_positional;
                    seen_positional = true;
                    if is_input {
                        return true;
                    }
                    args_node.named_child(*i as u32)
                }
                CallArgument::Keyword { name, .. }
                    if matches!(name.as_str(), "min" | "max" | "a_min" | "a_max") =>
                {
                    args_node
                        .named_child(*i as u32)
                        .and_then(|n| n.child_by_field_name("value"))
                }
                CallArgument::Keyword { .. } => return true,
            };
            !value_node.is_some_and(|n| {
                n.kind() == "none" || is_scalar_expr(n, ctx, &mut Vec::new())
            })
        })
        .map(|(_, arg)| arg)
        .collect()
}

fn has_internal_name(text: &str) -> bool {
    text.contains("__synth_") || text.contains("__ellipsis")
}

/// Replace each `__synth_N` in `text` with the source text it stands for.
/// `None` when an internal name has no source text.
fn restore_synthetic_text(text: &str, sources: &HashMap<String, String>) -> Option<String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("__synth_") {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        let digits = tail["__synth_".len()..]
            .bytes()
            .take_while(u8::is_ascii_digit)
            .count();
        let end = "__synth_".len() + digits;
        out.push_str(sources.get(&tail[..end])?);
        rest = &tail[end..];
    }
    out.push_str(rest);
    (!out.contains("__ellipsis")).then_some(out)
}

/// Remove internal names from the messages of `errors`. An error whose
/// message cannot be restored is dropped: a garbled message is worse than
/// none.
fn scrub_internal_names(errors: &mut Vec<ShapeError>, sources: &HashMap<String, String>) {
    errors.retain_mut(|error| {
        let restore = |text: &mut String| -> bool {
            if !has_internal_name(text) {
                return true;
            }
            restore_synthetic_text(text, sources).map(|t| *text = t).is_some()
        };
        restore(&mut error.variable)
            && restore(&mut error.message)
            && error.related.as_mut().is_none_or(|(_, m)| restore(m))
    });
}

/// Put the source text of an inline expression where an application names it
/// by a synthetic name, so the name does not depend on the order of the pass.
fn scrub_applications(applications: &mut [LayerApplication], sources: &HashMap<String, String>) {
    for application in applications {
        for input in std::iter::once(&mut application.input).chain(&mut application.extra_inputs) {
            if has_internal_name(input)
                && let Some(restored) = restore_synthetic_text(input, sources)
            {
                *input = restored;
            }
        }
    }
}

/// The dimension an integer expression stands for when it is built from
/// integer literals, names unpacked from `x.shape`, other plain integer
/// names, and `*`. `None` for anything else.
fn scalar_dim_expr(node: Node, scope_idx: usize, ctx: &ShapeCtx) -> Option<String> {
    let text = ctx.text.get(node.byte_range())?;
    match node.kind() {
        "integer" => Some(text.to_string()),
        "unary_operator" => {
            let literal: String = text.split_whitespace().collect();
            (literal.starts_with('-') && literal[1..].bytes().all(|b| b.is_ascii_digit()))
                .then_some(literal)
        }
        "identifier" => ctx
            .scalar_dims
            .get(&(scope_idx, text.to_string()))
            .cloned()
            .or_else(|| integer_constant(node, text, ctx)),
        "parenthesized_expression" => scalar_dim_expr(node.named_child(0)?, scope_idx, ctx),
        "binary_operator" => {
            let op = node.child_by_field_name("operator")?;
            if ctx.text.get(op.byte_range())? != "*" {
                return None;
            }
            let left = scalar_dim_expr(node.child_by_field_name("left")?, scope_idx, ctx)?;
            let right = scalar_dim_expr(node.child_by_field_name("right")?, scope_idx, ctx)?;
            Some(multiply_dim(&left, &right))
        }
        _ => None,
    }
}

/// The integer a name stands for when every binding of it, in the scope that
/// owns it, is a plain assignment of the same decimal literal. Whatever the
/// call time, the name then holds that integer.
fn integer_constant(at: Node, name: &str, ctx: &ShapeCtx) -> Option<String> {
    // An annotation that uses the name as a dim means a size that is not the
    // value of this name: `x: Float[Array, "D 3"]` next to `D = 64`.
    if !ctx.int_names.contains(name) || ctx.is_annotation_symbol(name) {
        return None;
    }
    literal_constant(at, name, ctx)
}

fn literal_constant(at: Node, name: &str, ctx: &ShapeCtx) -> Option<String> {
    let mut up = at.parent();
    let mut owner = None;
    while let Some(node) = up {
        match node.kind() {
            "lambda" | "list_comprehension" | "set_comprehension" | "dictionary_comprehension"
            | "generator_expression" => {
                if inner_scope_binds(node, name, ctx.text) {
                    return None;
                }
            }
            "function_definition" | "class_definition" | "module" => {
                owner = Some(node.id());
                break;
            }
            _ => {}
        }
        up = node.parent();
    }
    let key = (owner?, name.to_string());
    if let Some(known) = ctx.int_consts.borrow().get(&key) {
        return known.clone();
    }
    let found = integer_literal_of(at, name, ctx);
    ctx.int_consts.borrow_mut().insert(key, found.clone());
    found
}

fn integer_literal_of(at: Node, name: &str, ctx: &ShapeCtx) -> Option<String> {
    let (owner, NameBinding::Values(values)) = nearest_binding(at, name, ctx) else {
        return None;
    };
    if owner.kind() == "function_definition"
        && function_scope(owner, ctx)
            .is_none_or(|idx| ctx.scopes[idx].all_params.iter().any(|p| p == name))
    {
        return None;
    }
    let mut literal: Option<&str> = None;
    for value in values {
        let plain = value.parent().is_some_and(|p| {
            p.kind() == "assignment"
                && p.child_by_field_name("type")
                    .is_none_or(|t| ctx.text.get(t.byte_range()) == Some("int"))
        });
        let text = ctx.text.get(value.byte_range())?;
        if !plain || value.kind() != "integer" || !text.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        if *literal.get_or_insert(text) != text {
            return None;
        }
    }
    literal.map(str::to_string)
}

/// A call argument that is an integer expression, or a tuple or list of
/// them, written with real dimensions. `None` when it is not one, or when
/// writing it with real dimensions changes nothing.
fn dim_list_text(node: Node, scope_idx: usize, ctx: &ShapeCtx) -> Option<String> {
    if ctx.scalar_dims.is_empty() && ctx.int_names.is_empty() {
        return None;
    }
    let source = ctx.text.get(node.byte_range())?;
    if !matches!(node.kind(), "tuple" | "list") {
        return scalar_dim_expr(node, scope_idx, ctx).filter(|dim| dim != source);
    }
    let mut cursor = node.walk();
    let elements: Vec<Node> = node.named_children(&mut cursor).collect();
    let resolved: Vec<Option<String>> = elements
        .iter()
        .map(|element| scalar_dim_expr(*element, scope_idx, ctx))
        .collect();
    let written = |element: &Node| ctx.text.get(element.byte_range());
    if !elements
        .iter()
        .zip(&resolved)
        .any(|(element, dim)| dim.as_deref().is_some_and(|d| Some(d) != written(element)))
    {
        return None;
    }
    let dims = elements
        .iter()
        .zip(resolved)
        .map(|(element, dim)| dim.or_else(|| written(element).map(str::to_string)))
        .collect::<Option<Vec<_>>>()?;
    let text = format!("({}{})", dims.join(", "), if dims.len() == 1 { "," } else { "" });
    let squeeze = |t: &str| t.split_whitespace().collect::<String>();
    (squeeze(&text) != squeeze(source)).then_some(text)
}

/// Resolve one argument from its matching AST node. Expressions without a
/// known shape pass through unchanged.
fn resolve_call_arg(
    arg: CallArgument,
    arg_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<CallArgument> {
    let value = match &arg {
        CallArgument::Positional { value } | CallArgument::Keyword { value, .. } => value,
    };
    let value_node = match &arg {
        CallArgument::Positional { .. } => arg_node,
        CallArgument::Keyword { .. } => arg_node.child_by_field_name("value")?,
    };
    if let Some(dims) = dim_list_text(value_node, scope_idx, ctx) {
        return Some(match arg {
            CallArgument::Positional { .. } => CallArgument::Positional { value: dims },
            CallArgument::Keyword { name, .. } => CallArgument::Keyword { name, value: dims },
        });
    }
    if value.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return Some(arg);
    }
    let Some(shape) = shape_of_expression(value_node, ctx) else {
        return Some(arg);
    };
    let value = ctx.bind_synthetic(
        shape,
        scope_idx,
        value_node.start_byte(),
        ctx.text.get(value_node.byte_range()).unwrap_or_default(),
    );
    Some(match arg {
        CallArgument::Positional { .. } => CallArgument::Positional { value },
        CallArgument::Keyword { name, .. } => CallArgument::Keyword { name, value },
    })
}

fn resolve_call_args(
    raw_args: Vec<CallArgument>,
    args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<CallArgument>> {
    raw_args
        .into_iter()
        .enumerate()
        .map(|(i, arg)| resolve_call_arg(arg, args_node.named_child(i as u32)?, scope_idx, ctx))
        .collect()
}

/// Tuple and list arguments are evaluated element by element by the
/// sequence resolvers, so evaluating them here would report errors twice.
fn resolve_call_args_keeping_sequences(
    raw_args: Vec<CallArgument>,
    args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<CallArgument>> {
    raw_args
        .into_iter()
        .enumerate()
        .map(|(i, arg)| {
            let arg_node = args_node.named_child(i as u32)?;
            let value_node = match &arg {
                CallArgument::Positional { .. } => arg_node,
                CallArgument::Keyword { .. } => arg_node.child_by_field_name("value")?,
            };
            if matches!(value_node.kind(), "tuple" | "list") {
                return Some(arg);
            }
            resolve_call_arg(arg, arg_node, scope_idx, ctx)
        })
        .collect()
}

enum SequenceArg {
    /// A tuple or list of arrays in the first argument.
    Elements,
    /// A tuple or list of operands after the equation (torch.einsum).
    Operands,
    /// A sequence the rule cannot read from a tuple, so a tuple stays unknown.
    Opaque,
}

fn sequence_kind(known: &KnownFunction) -> Option<SequenceArg> {
    match known {
        KnownFunction::Stack
        | KnownFunction::Concatenate
        | KnownFunction::Vstack
        | KnownFunction::Hstack
        | KnownFunction::Dstack
        | KnownFunction::ColumnStack => Some(SequenceArg::Elements),
        KnownFunction::Einsum => Some(SequenceArg::Operands),
        KnownFunction::Block
        | KnownFunction::EinopsRearrange
        | KnownFunction::EinopsReduce
        | KnownFunction::EinopsRepeat => Some(SequenceArg::Opaque),
        _ => None,
    }
}

fn first_arg_is_tuple(args_node: Node) -> bool {
    find_positional_arg_node(args_node, 0).is_some_and(|node| node.kind() == "tuple")
}

fn resolve_einsum_args(
    args: Vec<CallArgument>,
    args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<CallArgument>> {
    let Some(sequence) = find_positional_arg_node(args_node, 1) else {
        return Some(args);
    };
    if !matches!(sequence.kind(), "tuple" | "list") {
        return Some(args);
    }
    let mut spliced = Vec::new();
    let mut positional = 0;
    for arg in args {
        if !matches!(arg, CallArgument::Positional { .. }) {
            spliced.push(arg);
            continue;
        }
        positional += 1;
        if positional != 2 {
            spliced.push(arg);
            continue;
        }
        let mut cursor = sequence.walk();
        for element in sequence.named_children(&mut cursor) {
            if element.kind() == "comment" {
                continue;
            }
            let shape = match binop_operand(element, scope_idx, ctx)? {
                BinopOperand::Scalar => Vec::new(),
                BinopOperand::Shaped(shape) => shape,
            };
            let value = ctx.bind_synthetic(
                shape,
                scope_idx,
                element.start_byte(),
                ctx.text.get(element.byte_range()).unwrap_or_default(),
            );
            spliced.push(CallArgument::Positional { value });
        }
    }
    Some(spliced)
}

fn resolve_stack_args(
    args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<CallArgument>> {
    let args = extract_call_arguments(args_node, ctx.text).ok()?;
    let sequence = find_positional_arg_node(args_node, 0).or_else(|| {
        ["arrays", "tensors", "arys", "operands"].iter().find_map(|name| {
            find_keyword_arg_node(args_node, name, ctx.text.as_bytes())?
                .child_by_field_name("value")
        })
    })?;
    if sequence.kind() == "list_comprehension" {
        let names = comprehension_names(sequence, scope_idx, ctx)?;
        return Some(with_sequence(args, format!("[{}]", names.join(", "))));
    }
    if !matches!(sequence.kind(), "tuple" | "list") {
        // A sequence variable is not represented by the tensor-shape map.
        return None;
    }
    let mut names = Vec::new();
    let mut cursor = sequence.walk();
    for element in sequence.named_children(&mut cursor) {
        if element.kind() == "comment" {
            continue;
        }
        let shape = match binop_operand(element, scope_idx, ctx)? {
            BinopOperand::Scalar => Vec::new(),
            BinopOperand::Shaped(shape) => shape,
        };
        names.push(ctx.bind_synthetic(
            shape,
            scope_idx,
            element.start_byte(),
            ctx.text.get(element.byte_range()).unwrap_or_default(),
        ));
    }
    Some(with_sequence(args, format!("[{}]", names.join(", "))))
}

/// `args` with the sequence argument of `stack` and its family replaced.
fn with_sequence(mut args: Vec<CallArgument>, value: String) -> Vec<CallArgument> {
    for arg in &mut args {
        match arg {
            CallArgument::Positional { value: original } => {
                *original = value;
                break;
            }
            CallArgument::Keyword { name, value: original }
                if ["arrays", "tensors", "arys", "operands"].contains(&name.as_str()) =>
            {
                *original = value;
                break;
            }
            _ => {}
        }
    }
    args
}

/// Find the Nth positional argument node in an argument_list.
///
/// Invariant: Python requires positional args to precede keyword args,
/// so `positional_index` matches the raw iteration index for positional
/// children. This function skips keyword_argument nodes to count only
/// positionals, making the index correct regardless of arg ordering.
fn find_positional_arg_node<'a>(args_node: Node<'a>, positional_index: usize) -> Option<Node<'a>> {
    let mut count = 0;
    for i in 0..args_node.named_child_count() {
        let child = args_node.named_child(i as u32)?;
        if child.kind() != "keyword_argument" {
            if count == positional_index {
                return Some(child);
            }
            count += 1;
        }
    }
    None
}

/// Find a keyword argument node by name in an argument_list.
fn find_keyword_arg_node<'a>(args_node: Node<'a>, name: &str, source_bytes: &[u8]) -> Option<Node<'a>> {
    for i in 0..args_node.named_child_count() {
        let child = args_node.named_child(i as u32)?;
        if child.kind() == "keyword_argument" {
            let name_node = child.child_by_field_name("name")?;
            let name_text = name_node.utf8_text(source_bytes).ok()?;
            if name_text == name {
                return Some(child);
            }
        }
    }
    None
}

// ── End recursive shape evaluator ────────────────────────────────────────

// ── vmap support ──────────────────────────────────────────────────────────
//
// v1 limitations (documented in code comments and PR description):
// - Only scalar-integer `in_axes` / `out_axes` are supported. Tuple,
//   PyTree, or `None` values cause `parse_vmap_call` to return None and
//   the call is silently skipped.
// - Nested vmap (`vmap(vmap(f))`) is not supported: the first positional
//   arg must be a bare identifier, so a `vmap(...)` expression won't parse
//   as a function name.
// - Cross-file wrapped functions are not supported: if `f` is imported
//   from another module, scope lookup will fail and we return Ok(None).
// - `vmap_targets` is global to the analysis pass, not per-scope. Nested-
//   scope shadowing of a vmap name is out of scope for v1.
// - `axis_size` keyword is not supported.

/// Metadata recorded when `jax.vmap(f, ...)` or `equinox.filter_vmap(f, ...)`
/// is encountered during the walk.
#[derive(Debug, Clone)]
struct VmapInfo {
    /// Name of the function being vmapped (bare identifier, e.g. "f").
    wrapped: String,
    /// `in_axes` value — default 0. Only scalar integers are honoured in v1.
    in_axes: isize,
    /// `out_axes` value — default 0. Only scalar integers are honoured in v1.
    out_axes: isize,
    /// Where the `vmap` call is: the name means what it means there.
    site: usize,
}

/// Parse a vmap/filter_vmap call's arguments into a `VmapInfo`.
///
/// Returns `None` if:
/// - The first positional arg is not a bare identifier (contains `.` or is
///   not simple).
/// - `in_axes` or `out_axes` is present but not a literal integer (tuple,
///   variable, `None`, etc.).
fn parse_vmap_call(args: &[CallArgument], site: usize) -> Option<VmapInfo> {
    // First positional arg = wrapped function name.
    let first_positional = args.iter().find_map(|arg| match arg {
        CallArgument::Positional { value } => Some(value.clone()),
        _ => None,
    })?;

    // Must be a bare identifier — no dots (no qualified names like
    // `module.func`), and no expressions like `lambda x: x`.
    if first_positional.contains('.') {
        return None;
    }

    // Read in_axes (default 0). Only scalar literal ints are honoured.
    let in_axes = parse_int_keyword(args, "in_axes", 0)?;
    // Read out_axes (default 0). Same rule.
    let out_axes = parse_int_keyword(args, "out_axes", 0)?;

    Some(VmapInfo {
        wrapped: first_positional,
        in_axes,
        out_axes,
        site,
    })
}

/// Extract a keyword argument as `isize`, falling back to `default`.
/// Returns `None` if the keyword is present but its value is not a
/// parseable integer (e.g. tuple, variable, None).
fn parse_int_keyword(args: &[CallArgument], name: &str, default: isize) -> Option<isize> {
    for arg in args {
        if let CallArgument::Keyword {
            name: kw_name,
            value,
        } = arg
            && kw_name == name
        {
            return value.parse::<isize>().ok();
        }
    }
    Some(default) // keyword not present → use default
}

/// Peel the batch dim from a shape at position `axis`.
/// Returns `Ok((peeled_shape, batch_dim))` or `Err(msg)`.
fn peel_batch_dim(shape: &[String], axis: isize) -> Result<(Vec<String>, String), String> {
    let len = shape.len() as isize;
    if len == 0 {
        return Err("vmap cannot map over a scalar (rank-0) input".to_string());
    }
    let axis = if axis < 0 { axis + len } else { axis };
    if axis < 0 || axis >= len {
        return Err(format!(
            "vmap in_axes {} out of bounds for input rank {}",
            axis, len
        ));
    }
    let axis = axis as usize;
    let batch_dim = shape[axis].clone();
    let mut peeled = shape.to_vec();
    peeled.remove(axis);
    Ok((peeled, batch_dim))
}

/// Insert a batch dim at position `axis` in the shape.
/// Negative axes count from the right *relative to the shape after
/// insertion* (numpy `np.insert` semantics). For v1 we clamp the
/// effective axis to `0..=len`.
///
/// The `len + axis + 1` formula for negative axes follows np.insert
/// convention: axis = -1 means "insert before the last element of
/// the result after insertion", which for a pre-insertion length of
/// `len` means position `len` (i.e. append at the end).
///
/// Example: shape = ["m"] (post-peel rank 1), axis = -1
///   → effective axis = 1 + (-1) + 1 = 1 → insert at position 1 (end of result)
///   → result = ["m", "B"]
///
/// Example: shape = ["m"] (post-peel rank 1), axis = 1
///   → effective axis = 1 (clamped to len=1) → insert at position 1 (end)
///   → result = ["m", "B"]
///
/// Example: shape = ["m", "k"] (post-peel rank 2), axis = -1
///   → effective axis = 2 + (-1) + 1 = 2 → insert at position 2 (end)
///   → result = ["m", "k", "B"]
fn prepend_batch_dim(mut shape: Vec<String>, axis: isize, dim: String) -> Vec<String> {
    let len = shape.len() as isize;
    // Normalise negative axis: -1 means "insert before the last element"
    // which means position len for len elements, or position len-1+1=len.
    // np.insert convention: axis = -1 → insert at the end.
    let axis = if axis < 0 {
        (len + axis + 1).max(0) as usize
    } else {
        (axis as usize).min(shape.len())
    };
    shape.insert(axis, dim);
    shape
}

/// Collect every `function_definition` node in `node`'s subtree, in the
/// same pre-order (push-before-recurse) traversal `extract_jaxtyping_shapes`
/// uses to push scopes — so index `i` here corresponds exactly to
/// `scopes[i + 1]` (the module scope at index 0 has no function node).
fn collect_function_definition_nodes<'t>(
    node: Node<'t>,
    out: &mut Vec<Node<'t>>,
    classes: &mut Vec<(usize, usize)>,
) {
    if node.kind() == "function_definition" {
        out.push(node);
    } else if node.kind() == "class_definition" {
        classes.push((node.start_byte(), node.end_byte()));
    }
    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i as u32) {
            collect_function_definition_nodes(child, out, classes);
        }
    }
}

/// Names a function body binds in its own scope. Nested `def`/`class` bodies
/// and lambdas are not entered; their names are bound here. Comprehension
/// targets are not bound (Python 3 scopes them to the comprehension), but a
/// walrus inside one is.
#[derive(Default)]
struct Bindings {
    bound: HashSet<String>,
    nonlocal: HashSet<String>,
    global: HashSet<String>,
}

fn insert_identifiers(node: Node, text: &str, out: &mut HashSet<String>) {
    if node.kind() == "identifier" {
        if let Ok(name) = node.utf8_text(text.as_bytes()) {
            out.insert(name.to_string());
        }
        return;
    }
    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i as u32) {
            insert_identifiers(child, text, out);
        }
    }
}

fn insert_target_names(target: Node, text: &str, out: &mut HashSet<String>) {
    match target.kind() {
        "identifier" => insert_identifiers(target, text, out),
        "pattern_list" | "tuple_pattern" | "list_pattern" | "list_splat_pattern"
        | "parenthesized_expression" | "expression_list" | "tuple" | "list" => {
            for i in 0..target.named_child_count() {
                if let Some(child) = target.named_child(i as u32) {
                    insert_target_names(child, text, out);
                }
            }
        }
        _ => {}
    }
}

fn collect_bindings(node: Node, text: &str, out: &mut Bindings) {
    for i in 0..node.named_child_count() {
        let Some(child) = node.named_child(i as u32) else {
            continue;
        };
        match child.kind() {
            "function_definition" | "class_definition" => {
                if let Some(name) = child.child_by_field_name("name") {
                    insert_identifiers(name, text, &mut out.bound);
                }
                continue;
            }
            "lambda" => continue,
            "global_statement" => {
                insert_identifiers(child, text, &mut out.global);
                continue;
            }
            "nonlocal_statement" => {
                insert_identifiers(child, text, &mut out.nonlocal);
                continue;
            }
            "import_statement" | "import_from_statement" | "future_import_statement"
            | "delete_statement" | "case_pattern" => {
                insert_identifiers(child, text, &mut out.bound);
                continue;
            }
            "assignment" | "augmented_assignment" | "for_statement" => {
                if let Some(left) = child.child_by_field_name("left") {
                    insert_target_names(left, text, &mut out.bound);
                }
            }
            "named_expression" => {
                if let Some(name) = child.child_by_field_name("name") {
                    insert_identifiers(name, text, &mut out.bound);
                }
            }
            "as_pattern" => {
                if let Some(alias) = child.child_by_field_name("alias") {
                    insert_identifiers(alias, text, &mut out.bound);
                }
            }
            _ => {}
        }
        collect_bindings(child, text, out);
    }
}

fn mentions_name(node: Node, text: &str, wanted: &dyn Fn(&str) -> bool) -> bool {
    if node.kind() == "identifier" {
        return node.utf8_text(text.as_bytes()).is_ok_and(wanted);
    }
    (0..node.named_child_count())
        .filter_map(|i| node.named_child(i as u32))
        .any(|child| mentions_name(child, text, wanted))
}

/// Link every nested function to the function that encloses it. `function_
/// nodes` is aligned with `scopes` (index 0, the module, has no node) and in
/// document order, so one stack of open ranges finds each parent.
fn build_scope_links(
    scopes: &[FunctionShapeScope],
    function_nodes: &[Option<Node>],
    class_ranges: &[(usize, usize)],
    text: &str,
) -> ScopeLinks {
    let n = scopes.len();
    let class_index = ScopeIndex::from_ranges(class_ranges.to_vec());
    let mut links = ScopeLinks {
        scope_index: ScopeIndex::new(scopes),
        parent: vec![None; n],
        locals: vec![HashSet::new(); n],
        poisoned: vec![HashSet::new(); n],
        captures: vec![false; n],
        redefined: vec![false; n],
        class_of: vec![None; n],
        methods: HashMap::new(),
    };

    let mut open: Vec<usize> = Vec::new();
    let mut has_closure = false;
    for i in 1..n {
        let Some(node) = function_nodes.get(i).copied().flatten() else {
            continue;
        };
        while open
            .last()
            .is_some_and(|&o| scopes[o].end_byte <= scopes[i].start_byte)
        {
            open.pop();
        }
        if let Some(&p) = open.last()
            && let Some(parent_node) = function_nodes.get(p).copied().flatten()
        {
            let mut up = node.parent();
            let mut through_class = false;
            while let Some(ancestor) = up
                && ancestor.id() != parent_node.id()
            {
                through_class |= ancestor.kind() == "class_definition";
                up = ancestor.parent();
            }
            if !through_class {
                links.parent[i] = Some(p);
                has_closure = true;
            }
        }
        links.class_of[i] = class_index
            .lookup(scopes[i].start_byte)
            .map(|class| class_ranges[class])
            .filter(|class| open.last().is_none_or(|&p| class.0 > scopes[p].start_byte));
        open.push(i);
    }
    // The enclosing function, the enclosing class, and the name.
    type Namespace<'a> = (Option<usize>, Option<(usize, usize)>, &'a str);
    let mut defined: HashMap<Namespace, Vec<usize>> = HashMap::new();
    for (i, scope) in scopes.iter().enumerate().skip(1) {
        if let Some(name) = scope.function_name.as_deref() {
            defined.entry((links.parent[i], links.class_of[i], name)).or_default().push(i);
            if let Some(class) = links.class_of[i] {
                links.methods.entry(name.to_string()).or_default().push((class, i));
            }
        }
    }
    for twins in defined.values().filter(|twins| twins.len() > 1) {
        for &i in twins {
            links.redefined[i] = true;
        }
    }
    if !has_closure {
        return links;
    }

    let mut needed = vec![false; n];
    for i in 1..n {
        let mut cur = links.parent[i].map(|_| i);
        while let Some(c) = cur
            && !needed[c]
        {
            needed[c] = true;
            cur = links.parent[c];
        }
    }
    let mut bindings: Vec<Bindings> = (0..n).map(|_| Bindings::default()).collect();
    for i in 1..n {
        if !needed[i] {
            continue;
        }
        let Some(body) = function_nodes[i].and_then(|f| f.child_by_field_name("body")) else {
            continue;
        };
        collect_bindings(body, text, &mut bindings[i]);
        let b = &bindings[i];
        links.locals[i] = scopes[i]
            .all_params
            .iter()
            .chain(&b.bound)
            .chain(&b.global)
            .filter(|name| !b.nonlocal.contains(*name))
            .cloned()
            .collect();
    }

    for i in 1..n {
        if links.parent[i].is_none() {
            continue;
        }
        let rebound: Vec<String> = bindings[i]
            .nonlocal
            .intersection(&bindings[i].bound)
            .cloned()
            .collect();
        for name in rebound {
            let mut owner = links.parent[i];
            while let Some(o) = owner {
                if links.locals[o].contains(&name) {
                    links.poisoned[o].insert(name);
                    break;
                }
                owner = links.parent[o];
            }
        }
        let Some(body) = function_nodes[i].and_then(|f| f.child_by_field_name("body")) else {
            continue;
        };
        let enclosing_binds = |name: &str| {
            if links.locals[i].contains(name) {
                return false;
            }
            let mut up = links.parent[i];
            while let Some(a) = up {
                if links.locals[a].contains(name) {
                    return true;
                }
                up = links.parent[a];
            }
            false
        };
        links.captures[i] = mentions_name(body, text, &enclosing_binds);
    }
    links
}

/// A module of this file built as `Class(...)`: the class, and the constructor
/// arguments that give its `self.<attr>` fields and parameters their values.
#[derive(Clone)]
struct Submodule {
    class: (usize, usize),
    ctor_args: Vec<CallArgument>,
    /// Start of the constructor call, to resolve `self.<attr>` in its arguments.
    byte: usize,
}

/// A `self.<attr>` field of one class, mirroring `ScopedSelfAttrLayer`.
struct SubmoduleField {
    class_start: usize,
    class_end: usize,
    /// `None` when the class writes the attribute with anything but one
    /// constructor call of one module class or one sequence of them, so the
    /// call stays unknown.
    module: Option<Field>,
}

struct ClassMethod {
    class_start: usize,
    class_end: usize,
    scope: usize,
}

/// User modules held in `self.<attr>` fields, plus the methods that decide
/// what calling one does.
#[derive(Default)]
struct Submodules {
    fields: HashMap<String, Vec<SubmoduleField>>,
    /// `__call__` of every class, then `forward` of every class: within one
    /// class the first entry wins, so `__call__` beats `forward`.
    callables: Vec<ClassMethod>,
    inits: Vec<ClassMethod>,
    /// The names that a `for` statement binds over a field that holds a
    /// sequence of modules.
    loop_variables: HashSet<String>,
}

/// The class a method is defined directly in.
fn defining_class(method: Node) -> Option<Node> {
    let mut parent = method.parent()?;
    if parent.kind() == "decorated_definition" {
        parent = parent.parent()?;
    }
    let class = parent.parent()?;
    (parent.kind() == "block" && class.kind() == "class_definition").then_some(class)
}

/// `Class(...)` where `Class` is defined once in this file and not shadowed by
/// an import.
fn constructed_module(
    call: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
    hierarchy: &ClassHierarchy,
) -> Option<Submodule> {
    if call.kind() != "call" {
        return None;
    }
    let name = call.child_by_field_name("function")?;
    if name.kind() != "identifier" {
        return None;
    }
    let name = name.utf8_text(text.as_bytes()).ok()?;
    if is_imported(import_map, name) {
        return None;
    }
    Some(Submodule {
        class: hierarchy.class_named(name)?,
        ctor_args: extract_call_arguments(call.child_by_field_name("arguments")?, text).ok()?,
        byte: call.start_byte(),
    })
}

fn extract_submodules(
    root: Node,
    text: &str,
    imports: &ImportScopes,
    hierarchy: &ClassHierarchy,
    writes: &[SelfAttrWrite],
    scopes: &[FunctionShapeScope],
    function_nodes: &[Option<Node>],
) -> Submodules {
    let mut out = Submodules::default();
    let mut forwards = Vec::new();
    for (i, scope) in scopes.iter().enumerate() {
        let list = match scope.function_name.as_deref() {
            Some("__call__") => &mut out.callables,
            Some("forward") => &mut forwards,
            Some("__init__") => &mut out.inits,
            _ => continue,
        };
        if let Some(class) = function_nodes.get(i).copied().flatten().and_then(defining_class) {
            list.push(ClassMethod {
                class_start: class.start_byte(),
                class_end: class.end_byte(),
                scope: i,
            });
        }
    }
    out.callables.append(&mut forwards);

    for write in writes.iter().filter(|w| w.class != (0, usize::MAX)) {
        let module = write
            .value
            .and_then(|value| field_of(value, text, imports.at(value.start_byte()), hierarchy));
        let fields = out.fields.entry(write.attr.clone()).or_default();
        match fields
            .iter_mut()
            .find(|f| (f.class_start, f.class_end) == write.class)
        {
            Some(field) => {
                let same = matches!((&field.module, &module), (Some(a), Some(b)) if a.same(b));
                if !same {
                    field.module = None;
                }
            }
            None => fields.push(SubmoduleField {
                class_start: write.class.0,
                class_end: write.class.1,
                module,
            }),
        }
    }

    // A sequence that some code changes in place is not the sequence that the
    // constructor built.
    let sequences: HashSet<&str> = out
        .fields
        .iter()
        .filter(|(_, fields)| {
            fields
                .iter()
                .any(|f| f.module.as_ref().is_some_and(Field::is_sequence))
        })
        .map(|(attr, _)| attr.as_str())
        .collect();
    if !sequences.is_empty() {
        for (attr, class) in changed_in_place(root, text, &sequences) {
            for field in out.fields.get_mut(&attr).into_iter().flatten() {
                let own = (field.class_start, field.class_end);
                let reached = class.is_none_or(|class| {
                    class == own
                        || hierarchy.mro(class).contains(&own)
                        || hierarchy.mro(own).contains(&class)
                });
                if reached {
                    field.module = None;
                }
            }
        }
        let sequences: HashSet<&str> = out
            .fields
            .iter()
            .filter(|(_, fields)| {
                fields
                    .iter()
                    .any(|f| f.module.as_ref().is_some_and(Field::is_sequence))
            })
            .map(|(attr, _)| attr.as_str())
            .collect();
        out.loop_variables = loop_variables(root, text, &sequences);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn propagate_calls(
    node: Node,
    text: &str,
    imports: &ImportScopes,
    layer_records: &[LayerAssignment],
    self_attr_layers: &HashMap<String, Vec<ScopedSelfAttrLayer>>,
    aliases: &HashMap<String, Vec<ScopedSelfAttrAlias>>,
    self_attr_writes: &[SelfAttrWrite],
    hierarchy: &ClassHierarchy,
    search_roots: &[PathBuf],
    read_file: &dyn Fn(&PathBuf) -> Option<String>,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
    scopes: &mut [FunctionShapeScope],
) -> Result<PropagateOutput, String> {
    let (output, ()) = with_propagation(
        node,
        text,
        imports,
        layer_records,
        self_attr_layers,
        aliases,
        self_attr_writes,
        hierarchy,
        search_roots,
        read_file,
        max_depth,
        cache,
        scopes,
        |ctx| {
            // Collect all assignments (identifier and tuple-pattern LHS) in
            // source order.
            let assignments = collect_assignment_items(node, text)?;
            process_assignment_items(node, assignments, None, ctx);

            // Also handle binary-operator expressions in return/yield/assert
            // statements. The recursive evaluator handles the expression
            // itself, but we need to walk these statement types to catch
            // shape errors (e.g., matmul dimension mismatches) that don't
            // appear in assignments.
            let value_statements = collect_value_statements(node, text)?;
            process_value_statements(value_statements, ctx);

            scrub_internal_names(ctx.errors, &ctx.synthetic_text);
            scrub_applications(ctx.applications, &ctx.synthetic_text);
            Ok(())
        },
    )?;
    Ok(output)
}

/// `propagate_calls` over the units the plan marks dirty. The records of the
/// others stand in for them.
#[allow(clippy::too_many_arguments)]
fn propagate_planned(
    node: Node,
    text: &str,
    imports: &ImportScopes,
    layer_records: &[LayerAssignment],
    self_attr_layers: &HashMap<String, Vec<ScopedSelfAttrLayer>>,
    aliases: &HashMap<String, Vec<ScopedSelfAttrAlias>>,
    self_attr_writes: &[SelfAttrWrite],
    hierarchy: &ClassHierarchy,
    search_roots: &[PathBuf],
    read_file: &dyn Fn(&PathBuf) -> Option<String>,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
    scopes: &mut [FunctionShapeScope],
    plan: PropPlan,
    next: &mut UnitMemo,
) -> Result<PropagateOutput, String> {
    let nodes = unit_nodes(node);
    let runs = if plan.dirty.iter().any(|&dirty| dirty) {
        with_propagation(
            node,
            text,
            imports,
            layer_records,
            self_attr_layers,
            aliases,
            self_attr_writes,
            hierarchy,
            search_roots,
            read_file,
            max_depth,
            cache,
            scopes,
            |ctx| run_units(&nodes, &plan.dirty, ctx),
        )?
        .1
    } else {
        nodes.iter().map(|_| None).collect()
    };
    plan.finish(runs, scopes, next)
}

/// How much output a pass had made at one point.
#[derive(Clone, Copy)]
struct Mark {
    errors: usize,
    applications: usize,
    assignments: usize,
}

impl Mark {
    fn of(ctx: &ShapeCtx) -> Self {
        Self {
            errors: ctx.errors.len(),
            applications: ctx.applications.len(),
            assignments: ctx.assignment_shapes.len(),
        }
    }
}

/// Run the pass over the dirty units in source order: all their assignments,
/// then all their value statements. The output of each unit and phase is cut
/// from the output of the pass, and its internal names are put back.
fn run_units(
    nodes: &[Node],
    dirty: &[bool],
    ctx: &mut ShapeCtx,
) -> Result<Vec<Option<UnitRun>>, String> {
    let cache = ctx.cache;
    let record = |f: &mut dyn FnMut()| match cache {
        Some(cache) => cache.record_deps(f).1,
        None => {
            f();
            None
        }
    };
    let mut cuts: Vec<[Option<(Mark, Mark)>; 2]> = vec![[None, None]; nodes.len()];
    let mut deps: Vec<[Option<crate::resolution::Deps>; 2]> = vec![[None, None]; nodes.len()];
    for (phase, step) in [0usize, 1].into_iter().enumerate() {
        for (i, node) in nodes.iter().enumerate().filter(|(i, _)| dirty[*i]) {
            let from = Mark::of(ctx);
            let mut failed = false;
            deps[i][phase] = record(&mut || {
                if step == 0 {
                    let mut items = Vec::new();
                    failed |= collect_items_of_child(*node, ctx.text, &mut items).is_err();
                    items.sort_by_key(|(_, rhs, _)| rhs.start_byte());
                    replay_items(collect_flow(*node, ctx.text, ctx.imports, false), items, None, ctx);
                } else {
                    let mut statements = Vec::new();
                    failed |=
                        collect_value_statements_of_child(*node, ctx.text, &mut statements).is_err();
                    statements.sort_by_key(|n| n.start_byte());
                    process_value_statements(statements, ctx);
                }
            });
            if failed {
                return Err("a unit has no statements to read".to_string());
            }
            cuts[i][phase] = Some((from, Mark::of(ctx)));
        }
    }
    let mut errors = std::mem::take(ctx.errors).into_iter();
    let mut applications = std::mem::take(ctx.applications).into_iter();
    let mut assignments = std::mem::take(ctx.assignment_shapes).into_iter();
    let mut runs: Vec<Option<UnitRun>> = nodes.iter().map(|_| None).collect();
    let mut phases: Vec<[Segment; 2]> = nodes.iter().map(|_| Default::default()).collect();
    for phase in 0..2 {
        for i in (0..nodes.len()).filter(|&i| dirty[i]) {
            let (from, to) = cuts[i][phase].ok_or("a unit was not run")?;
            let mut segment_errors: Vec<ShapeError> =
                errors.by_ref().take(to.errors - from.errors).collect();
            scrub_internal_names(&mut segment_errors, &ctx.synthetic_text);
            let mut segment_applications: Vec<LayerApplication> = applications
                .by_ref()
                .take(to.applications - from.applications)
                .collect();
            scrub_applications(&mut segment_applications, &ctx.synthetic_text);
            phases[i][phase] = Segment {
                errors: segment_errors,
                applications: segment_applications,
                assignments: std::sync::Arc::new(
                    assignments
                        .by_ref()
                        .take(to.assignments - from.assignments)
                        .collect(),
                ),
            };
        }
    }
    if errors.next().is_some() || applications.next().is_some() || assignments.next().is_some() {
        return Err("output outside every unit".to_string());
    }
    for (i, run) in runs.iter_mut().enumerate().filter(|(i, _)| dirty[*i]) {
        *run = Some(UnitRun {
            phases: std::mem::take(&mut phases[i]),
            deps: std::mem::take(&mut deps[i]),
        });
    }
    Ok(runs)
}

/// Build the context of a propagation pass over `node` and run `body` in it.
/// Returns what the pass produced, and what `body` returned.
#[allow(clippy::too_many_arguments)]
fn with_propagation<T>(
    node: Node,
    text: &str,
    imports: &ImportScopes,
    layer_records: &[LayerAssignment],
    self_attr_layers: &HashMap<String, Vec<ScopedSelfAttrLayer>>,
    aliases: &HashMap<String, Vec<ScopedSelfAttrAlias>>,
    self_attr_writes: &[SelfAttrWrite],
    hierarchy: &ClassHierarchy,
    search_roots: &[PathBuf],
    read_file: &dyn Fn(&PathBuf) -> Option<String>,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
    scopes: &mut [FunctionShapeScope],
    body: impl FnOnce(&mut ShapeCtx<'_>) -> Result<T, String>,
) -> Result<(PropagateOutput, T), String> {
    let mut applications = Vec::new();
    let mut errors = Vec::new();
    let mut assignment_shapes = Vec::new();
    let mut vmap_targets: HashMap<String, VmapInfo> = HashMap::new();

    let mut function_nodes = Vec::new();
    let mut class_ranges = Vec::new();
    collect_function_definition_nodes(node, &mut function_nodes, &mut class_ranges);
    let class_starts: Vec<usize> = class_ranges.iter().map(|r| r.0).collect();
    let mut scope_function_nodes: Vec<Option<Node>> = std::iter::once(None)
        .chain(function_nodes.into_iter().map(Some))
        .collect();
    // Defensive: the two traversals are structurally identical, but if a
    // future edit to either desyncs them, pad/truncate rather than panic —
    // lazy seeding simply won't find a body for any scope past the mismatch.
    scope_function_nodes.resize(scopes.len(), None);

    // Snapshot BEFORE any assignment/specialization runs — see the
    // `original_shapes` field doc for why this must never be re-derived
    // from the (mutable) live scopes later.
    let original_shapes: Vec<HashMap<String, Vec<String>>> =
        scopes.iter().map(|s| s.shapes.clone()).collect();

    let links = build_scope_links(scopes, &scope_function_nodes, &class_ranges, text);
    let submodules = extract_submodules(
        node,
        text,
        imports,
        hierarchy,
        self_attr_writes,
        scopes,
        &scope_function_nodes,
    );

    let mut ctx = ShapeCtx {
        text,
        imports,
        layer_records,
        self_attr_layers,
        aliases,
        attr_writes: self_attr_writes,
        layer_dims: Default::default(),
        root: node,
        hierarchy,
        search_roots,
        read_file,
        max_depth,
        cache,
        scopes,
        vmap_targets: &mut vmap_targets,
        applications: &mut applications,
        errors: &mut errors,
        assignment_shapes: &mut assignment_shapes,
        synthetic_counter: 0,
        synthetics: HashMap::new(),
        synthetic_text: HashMap::new(),
        scalar_dims: HashMap::new(),
        tuple_carries: HashSet::new(),
        scope_function_nodes: &scope_function_nodes,
        active_specializations: Vec::new(),
        specialized_scopes: std::collections::HashSet::new(),
        original_shapes: &original_shapes,
        links: &links,
        submodules: &submodules,
        global_names: Default::default(),
        classes: ScopeIndex::from_ranges(class_ranges),
        class_starts,
        module_writes: HashMap::new(),
        class_body_names: Default::default(),
        global_scopes: Default::default(),
        module_values: Default::default(),
        module_index: Default::default(),
        annotation_symbols: Default::default(),
        int_names: HashSet::new(),
        int_consts: Default::default(),
        call_time: Vec::new(),
        inner_masked: Vec::new(),
        inner_scalars: HashSet::new(),
        inner_arrays: HashSet::new(),
    };

    let result = body(&mut ctx)?;
    Ok(((applications, errors, assignment_shapes), result))
}

fn process_value_statements(statements: Vec<Node>, ctx: &mut ShapeCtx) {
    for stmt in statements {
        let errors_before = ctx.errors.len();
        let _ = shape_of_expression(stmt, ctx);
        // Fix up error variable names: for return/yield/assert contexts,
        // there is no LHS variable, so set the variable name to empty.
        for err in &mut ctx.errors[errors_before..] {
            err.variable = String::new();
        }
    }
}

/// Replay the assignments under `root` in source order. `only_scope` keeps
/// the items whose innermost scope is that one.
fn process_assignment_items(
    root: Node,
    items: Vec<(Lhs, Node, Node)>,
    only_scope: Option<usize>,
    ctx: &mut ShapeCtx,
) {
    let in_class = directly_in_class(root);
    replay_items(collect_flow(root, ctx.text, ctx.imports, in_class), items, only_scope, ctx);
}

/// `process_assignment_items`, with the branches and loops already found. A
/// branch gives each arm the state before it and joins the arms afterwards
/// (see `flow`); a loop body is replayed once.
fn replay_items(
    flow: Vec<Marked>,
    items: Vec<(Lhs, Node, Node)>,
    only_scope: Option<usize>,
    ctx: &mut ShapeCtx,
) {
    let mut replay = Replay::new(flow);
    for (lhs, rhs_node, assignment_node) in items {
        let byte = rhs_node.start_byte();
        let scope = ctx.links.scope_index.lookup(byte);
        if let Some(only) = only_scope
            && scope != Some(only)
        {
            continue;
        }
        replay.advance(byte, ctx);
        if let Some(scope) = scope {
            replay.wrote(scope, &lhs, ctx);
        }
        process_assignment_item(lhs, rhs_node, assignment_node, ctx);
    }
    replay.finish(ctx);
}

/// Process one collected assignment item, binding its LHS to the RHS's
/// computed shape (or evicting a stale binding). Shared by the top-level
/// whole-file walk (`propagate_calls`) and lazy call-site parameter seeding
/// (`specialize_callee_call`, which re-collects and replays just a single
/// callee's own assignments on demand).
fn process_assignment_item(lhs: Lhs, rhs_node: Node, assignment_node: Node, ctx: &mut ShapeCtx) {
    let name = match &lhs {
        Lhs::Single(name) | Lhs::Augmented(name) => Some(name.clone()),
        Lhs::Tuple(_) => None,
    };
    if let (Lhs::Single(name), "integer") = (&lhs, rhs_node.kind()) {
        ctx.int_names.insert(name.clone());
    }
    let scope = ctx.links.scope_index.lookup(rhs_node.start_byte());
    let in_class = ctx.in_class_body(assignment_node.start_byte(), scope);
    if in_class && scope != Some(0) {
        return;
    }
    process_assignment_item_inner(lhs, rhs_node, assignment_node, ctx);
    if let (Some(name), Some(0), false) = (name, scope, in_class) {
        ctx.note_module_write(&name);
    }
}

fn process_assignment_item_inner(
    lhs: Lhs,
    rhs_node: Node,
    assignment_node: Node,
    ctx: &mut ShapeCtx,
) {
    // Inlay hints display non-annotated assignments only (an annotated
    // assignment already shows the user's written type). `None` here means
    // "bind the shape but emit no inlay record".
    let display_line = if assignment_node.child_by_field_name("type").is_none() {
        Some(assignment_node.start_position().row as u32)
    } else {
        None
    };

    // Tuple-pattern unpacking (issue #30) — bind each element and move on.
    let lhs_name = match lhs {
        Lhs::Single(name) => name,
        Lhs::Augmented(name) => {
            handle_augmented_assignment(&name, rhs_node, assignment_node, display_line, ctx);
            return;
        }
        Lhs::Tuple(names) => {
            handle_tuple_assignment(&names, rhs_node, display_line, ctx);
            return;
        }
    };
    let scope_idx = match ctx.links.scope_index.lookup(rhs_node.start_byte()) {
        Some(idx) => idx,
        None => return,
    };

    // Special-case vmap *recording* before delegating to
    // shape_of_expression.  vmap bindings like `vf = jax.vmap(f)`
    // produce no output shape of their own but need to be recorded
    // so that later calls to `vf(...)` can be expanded.
    //
    // TODO: Consider threading lhs_name through ShapeCtx so that
    // shape_of_call can own the vmap-recording path too, eliminating
    // this duplication.  Currently shape_of_call doesn't know the
    // LHS variable name for the assignment.
    if rhs_node.kind() == "call"
        && let Some(func_node) = rhs_node.child_by_field_name("function") {
            let target = func_node.utf8_text(ctx.text.as_bytes()).ok().unwrap_or("");
            let resolved = resolve_call_target(target, ctx.imports.at(func_node.start_byte()));
            if let Some(KnownFunction::Vmap) = classify_known_function(&resolved) {
                let args_node = rhs_node.child_by_field_name("arguments");
                if let Some(an) = args_node
                    && let Ok(args) = extract_call_arguments(an, ctx.text)
                        && let Some(info) = parse_vmap_call(&args, rhs_node.start_byte()) {
                            ctx.vmap_targets.insert(lhs_name.clone(), info);
                        }
                // vmap binding has no output shape — skip to next assignment.
                return;
            }
        }

    // Also need to handle the old-style layer applications. The
    // layer-application path in shape_of_call doesn't know the LHS
    // variable name, so we do a pre-check here for layer calls.
    // Inline expression arguments (e.g. `layer(jnp.exp(x))`) are
    // resolved via resolve_call_args so nested inputs get a shape.
    //
    // TODO: Same as vmap above — threading lhs_name through ShapeCtx
    // would let shape_of_call own the layer-application path, removing
    // this duplication.
    if rhs_node.kind() == "call"
        && let Some(func_node) = rhs_node.child_by_field_name("function") {
            let target = func_node.utf8_text(ctx.text.as_bytes()).ok().unwrap_or("");
            let call_byte = rhs_node.start_byte();
            if !ctx.is_loop_module(func_node, target)
                && let Some(record) =
                    find_scoped_layer(ctx.layer_records, ctx.scopes, &ctx.links.scope_index, call_byte, target)
            {
                let kind = ctx.layer_at(&record.kind, record.byte_position);
                let args_node = rhs_node.child_by_field_name("arguments");
                if let Some(an) = args_node
                    && let Ok(raw_args) = extract_call_arguments(an, ctx.text)
                        // Recursively evaluate inline expression args (e.g.
                        // `layer(jnp.exp(x))`) so the input resolves to a
                        // synthetic name with a known shape.
                        && let Some(args) =
                            resolve_call_args(raw_args, an, scope_idx, ctx)
                        && let Some(CallArgument::Positional { value: input }) =
                            args.first().cloned()
                        {
                            let application = LayerApplication {
                                variable: lhs_name.clone(),
                                layer: target.to_string(),
                                input,
                                extra_inputs: extra_layer_inputs(&args, &kind),
                                kind,
                                range: an.range(),
                            };
                            let result = apply_layer_application(&application, &ctx.scope_shapes(scope_idx));
                            match result {
                                Ok(Some(output)) => {
                                    ctx.record_binding(
                                        scope_idx,
                                        &lhs_name,
                                        output,
                                        display_line,
                                        call_byte,
                                    );
                                }
                                Ok(None) => {
                                    ctx.evict_binding(scope_idx, &lhs_name);
                                }
                                Err(message) => {
                                    ctx.evict_binding(scope_idx, &lhs_name);
                                    ctx.errors.push(ShapeError::mismatch(lhs_name.clone(), message, application.range));
                                }
                            }
                            ctx.applications.push(application);
                            return;
                        }
            }
        }

    // Delegate to the recursive evaluator.
    let errors_before = ctx.errors.len();
    let result = shape_of_expression(rhs_node, ctx);

    // Fix up error variable names: shape_of_expression doesn't know the
    // LHS variable name, so it records errors with placeholder names
    // (the function target or expression text). Replace with the actual
    // LHS name.
    for err in &mut ctx.errors[errors_before..] {
        err.variable = lhs_name.clone();
    }

    if let Some(shape) = result {
        let carry = is_tuple_value(rhs_node, scope_idx, ctx);
        ctx.record_binding(scope_idx, &lhs_name, shape, display_line, rhs_node.start_byte());
        if carry && ctx.scopes[scope_idx].shapes.contains_key(&lhs_name) {
            ctx.tuple_carries.insert((scope_idx, lhs_name));
        }
    } else if display_line.is_some() {
        // Non-annotated reassignment with an unshapeable RHS: the old
        // binding is stale now (issue #46). Annotated assignments keep
        // their user-written shape.
        ctx.evict_binding(scope_idx, &lhs_name);
    }
}

/// Left-hand side of an assignment we propagate shapes through.
enum Lhs {
    /// `x = ...`
    Single(String),
    /// `x += ...` / `x @= ...` etc. — combines the existing LHS shape with
    /// the RHS shape instead of replacing it.
    Augmented(String),
    /// `a, b = ...` / `(a, b) = ...` / `[a, b] = ...`. Each element is a
    /// `TupleTarget` (a plain name, or one level of nested tuple pattern).
    Tuple(Vec<TupleTarget>),
}

/// One top-level element of a tuple-assignment LHS pattern. Supports one
/// level of nesting (`out, (h, c) = self.lstm(x)`), needed for RNN
/// `(output, state)` returns where `state` is itself an LSTM `(h, c)` pair
/// (see `tuple_rhs_shapes`'s `LayerKind::Rnn` arm).
#[derive(Debug, Clone)]
enum TupleTarget {
    /// A plain identifier target, or `None` for `_`/non-identifier elements
    /// (skipped).
    Name(Option<String>),
    /// A nested `(a, b, ...)` pattern one level down. Every inner name is
    /// bound to the *same* shape as this top-level element (`handle_tuple_
    /// assignment` clones it per name) — correct for LSTM's `(h, c)`, which
    /// share one shape under this analyzer's approximation.
    Nested(Vec<Option<String>>),
    /// `*rest`: Python binds a list, never a tensor, so the name is always
    /// unknown (a stale binding is evicted). It still takes a position in
    /// the pattern, which a literal RHS needs to align the other targets.
    Star(Option<String>),
}

impl TupleTarget {
    /// All identifier names bound by this target (flattening one level of
    /// nesting), for coverage accounting.
    fn names(&self) -> Vec<&str> {
        match self {
            TupleTarget::Name(Some(n)) => vec![n.as_str()],
            TupleTarget::Name(None) | TupleTarget::Star(None) => vec![],
            TupleTarget::Star(Some(n)) => vec![n.as_str()],
            TupleTarget::Nested(inner) => inner.iter().flatten().map(String::as_str).collect(),
        }
    }
}

/// Parse one nested tuple/list pattern's identifier children (used for both
/// the outer LHS pattern and, one level down, a nested `(h, c)` group).
/// `_` and non-identifier elements map to `None` (skipped).
fn parse_pattern_names(pattern: Node, text: &str) -> Vec<Option<String>> {
    let mut names = Vec::new();
    for k in 0..pattern.named_child_count() {
        let el = pattern.named_child(k as u32).unwrap();
        let name = if el.kind() == "identifier" {
            match el.utf8_text(text.as_bytes()) {
                Ok("_") | Err(_) => None,
                Ok(t) => Some(t.to_string()),
            }
        } else {
            None
        };
        names.push(name);
    }
    names
}

/// Collect all assignment statements (identifier and tuple-pattern LHS) in
/// source order. Returns (lhs, rhs_node, assignment_node).
fn collect_assignment_items<'a>(
    node: Node<'a>,
    text: &str,
) -> Result<Vec<(Lhs, Node<'a>, Node<'a>)>, String> {
    let mut result = Vec::new();
    collect_items_recursive(node, text, &mut result)?;
    result.sort_by_key(|(_, rhs, _)| rhs.start_byte());
    Ok(result)
}

fn collect_items_recursive<'a>(
    node: Node<'a>,
    text: &str,
    out: &mut Vec<(Lhs, Node<'a>, Node<'a>)>,
) -> Result<(), String> {
    for i in 0..node.named_child_count() {
        collect_items_of_child(node.named_child(i as u32).unwrap(), text, out)?;
    }
    Ok(())
}

/// The items `collect_items_recursive` finds for one child of its node.
fn collect_items_of_child<'a>(
    child: Node<'a>,
    text: &str,
    out: &mut Vec<(Lhs, Node<'a>, Node<'a>)>,
) -> Result<(), String> {
    if child.kind() == "expression_statement" {
        // Extract assignments from within expression_statement, but
        // do NOT recurse into it — the inner walk already handled
        // all its children.
        for j in 0..child.named_child_count() {
            let inner = child.named_child(j as u32).unwrap();
            if inner.kind() == "assignment" || inner.kind() == "augmented_assignment" {
                push_assignment_item(inner, text, out);
            }
        }
        return Ok(());
    } else if child.kind() == "assignment" || child.kind() == "augmented_assignment" {
        push_assignment_item(child, text, out);
        // Also don't recurse into bare assignment nodes to avoid
        // double-counting.
        return Ok(());
    }
    collect_items_recursive(child, text, out)
}

fn push_assignment_item<'a>(
    assignment: Node<'a>,
    text: &str,
    out: &mut Vec<(Lhs, Node<'a>, Node<'a>)>,
) {
    let (Some(lhs), Some(rhs)) = (
        assignment.child_by_field_name("left"),
        assignment.child_by_field_name("right"),
    ) else {
        return;
    };

    match lhs.kind() {
        "identifier" => {
            if let Ok(name) = lhs.utf8_text(text.as_bytes()) {
                let lhs = if assignment.kind() == "augmented_assignment" {
                    Lhs::Augmented(name.to_string())
                } else {
                    Lhs::Single(name.to_string())
                };
                out.push((lhs, rhs, assignment));
            }
        }
        "pattern_list" | "tuple_pattern" | "list_pattern" => {
            let mut targets = Vec::new();
            for k in 0..lhs.named_child_count() {
                let el = lhs.named_child(k as u32).unwrap();
                let target = match el.kind() {
                    "identifier" => TupleTarget::Name(match el.utf8_text(text.as_bytes()) {
                        Ok("_") | Err(_) => None,
                        Ok(t) => Some(t.to_string()),
                    }),
                    "tuple_pattern" | "pattern_list" | "list_pattern" => {
                        TupleTarget::Nested(parse_pattern_names(el, text))
                    }
                    "list_splat_pattern" => TupleTarget::Star(
                        el.named_child(0)
                            .filter(|n| n.kind() == "identifier")
                            .and_then(|n| n.utf8_text(text.as_bytes()).ok())
                            .filter(|t| *t != "_")
                            .map(str::to_string),
                    ),
                    _ => TupleTarget::Name(None),
                };
                targets.push(target);
            }
            out.push((Lhs::Tuple(targets), rhs, assignment));
        }
        _ => {}
    }
}

/// Propagate `x += expr` / `x @= expr`: the new shape of `x` combines its
/// existing shape with the RHS shape (broadcast for elementwise ops, matmul
/// for `@=`) rather than replacing it. Missing shapes skip silently.
fn handle_augmented_assignment(
    name: &str,
    rhs_node: Node,
    assignment_node: Node,
    display_line: Option<u32>,
    ctx: &mut ShapeCtx,
) {
    let Some(scope_idx) = ctx.links.scope_index.lookup(rhs_node.start_byte()) else {
        return;
    };
    let Some(lhs_shape) = ctx.resolve_shape(name, scope_idx) else {
        return;
    };
    let Some(rhs_shape) = shape_of_expression(rhs_node, ctx) else {
        return;
    };

    let op_text = assignment_node
        .child_by_field_name("operator")
        .and_then(|op| op.utf8_text(ctx.text.as_bytes()).ok())
        .unwrap_or("+=");
    let result = match op_text {
        "@=" => apply_matmul_shape(&lhs_shape, &rhs_shape, name, "rhs"),
        "-=" => apply_elementwise_shape(&lhs_shape, &rhs_shape, BinaryOp::Sub),
        "*=" => apply_elementwise_shape(&lhs_shape, &rhs_shape, BinaryOp::Mul),
        "/=" => apply_elementwise_shape(&lhs_shape, &rhs_shape, BinaryOp::Div),
        // +=, and any other in-place op (//=, **=, %=, bitwise) is
        // elementwise-broadcasting for shape purposes.
        _ => apply_elementwise_shape(&lhs_shape, &rhs_shape, BinaryOp::Add),
    };
    match result {
        Ok(Some(shape)) => {
            ctx.record_binding(scope_idx, name, shape, display_line, rhs_node.start_byte())
        }
        Ok(None) => {}
        Err(message) => ctx.errors.push(ShapeError::mismatch(name.to_string(), message, assignment_node.range())),
    }
}

/// Bind a tuple-pattern assignment's element shapes (issue #30).
fn handle_tuple_assignment(
    targets: &[TupleTarget],
    rhs_node: Node,
    display_line: Option<u32>,
    ctx: &mut ShapeCtx,
) {
    let Some(scope_idx) = ctx.links.scope_index.lookup(rhs_node.start_byte()) else {
        return;
    };
    // `a, b = x.shape`: each name is an integer scalar (zero-rank) that
    // stands for one dim of `x`.
    let mut unpacked_dims = None;
    let elem_shapes = match literal_elements(rhs_node) {
        Some(elems) => tuple_literal_shapes(targets, &elems, ctx),
        None => match shape_attribute_dims(rhs_node, ctx) {
            Some(dims) => {
                let n = dims.len();
                unpacked_dims = Some(dims);
                (n == targets.len()).then(|| vec![Some(Vec::new()); n])
            }
            None => tuple_rhs_shapes(rhs_node, targets, scope_idx, ctx),
        },
    };
    let Some(elem_shapes) = elem_shapes else {
        // Unmodelled multi-output RHS: prior bindings for the targets are
        // stale now (issue #46).
        for name in targets.iter().flat_map(TupleTarget::names) {
            ctx.evict_binding(scope_idx, name);
        }
        return;
    };
    for (target, shape) in targets.iter().zip(elem_shapes) {
        match target {
            TupleTarget::Name(Some(name)) => match shape {
                Some(shape) => {
                    ctx.record_binding(scope_idx, name, shape, display_line, rhs_node.start_byte())
                }
                None => ctx.evict_binding(scope_idx, name),
            },
            TupleTarget::Name(None) | TupleTarget::Star(None) => {}
            TupleTarget::Star(Some(name)) => ctx.evict_binding(scope_idx, name),
            // Nested `(h, c)`-style group one level down: every inner name
            // binds to the same element shape (see `TupleTarget::Nested`).
            TupleTarget::Nested(inner) => {
                for name in inner.iter().flatten() {
                    match &shape {
                        Some(shape) => ctx.record_binding(
                            scope_idx,
                            name,
                            shape.clone(),
                            display_line,
                            rhs_node.start_byte(),
                        ),
                        None => ctx.evict_binding(scope_idx, name),
                    }
                }
            }
        }
    }
    if let Some(TupleTarget::Name(Some(name))) = targets.first()
        && ctx.scopes[scope_idx].shapes.contains_key(name)
        && scan_carry_is_tuple(rhs_node, scope_idx, ctx)
    {
        ctx.tuple_carries.insert((scope_idx, name.clone()));
    }
    for (target, dim) in targets.iter().zip(unpacked_dims.into_iter().flatten()) {
        if let TupleTarget::Name(Some(name)) = target
            && ctx.scopes[scope_idx].shapes.contains_key(name)
        {
            ctx.scalar_dims.insert((scope_idx, name.clone()), dim);
        }
    }
}

/// The `init` argument of a `jax.lax.scan(f, init, xs)` call.
fn scan_init_node<'t>(args_node: Node<'t>, text: &str) -> Option<Node<'t>> {
    find_keyword_arg_node(args_node, "init", text.as_bytes())
        .and_then(|kw| kw.child_by_field_name("value"))
        .or_else(|| find_positional_arg_node(args_node, 1))
}

/// Whether `rhs` is a `jax.lax.scan` call whose `init` is a tuple, so its
/// first result is a tuple carry.
fn scan_carry_is_tuple(rhs: Node, scope_idx: usize, ctx: &ShapeCtx) -> bool {
    let Some(func) = rhs.child_by_field_name("function").filter(|_| rhs.kind() == "call") else {
        return false;
    };
    let Ok(target) = func.utf8_text(ctx.text.as_bytes()) else {
        return false;
    };
    matches!(
        classify_known_function(&resolve_call_target(target, ctx.imports.at(func.start_byte()))),
        Some(KnownFunction::Scan)
    ) && rhs
        .child_by_field_name("arguments")
        .and_then(|args| scan_init_node(args, ctx.text))
        .is_some_and(|init| is_tuple_value(init, scope_idx, ctx))
}

/// Whether `node` evaluates to a tuple: a tuple literal, or a name whose
/// current binding is a tuple carry.
fn is_tuple_value(node: Node, scope_idx: usize, ctx: &ShapeCtx) -> bool {
    match node.kind() {
        "tuple" | "expression_list" => true,
        "parenthesized_expression" => node
            .named_child(0)
            .is_some_and(|inner| is_tuple_value(inner, scope_idx, ctx)),
        "identifier" => node
            .utf8_text(ctx.text.as_bytes())
            .is_ok_and(|name| ctx.tuple_carries.contains(&(scope_idx, name.to_string()))),
        _ => false,
    }
}

/// Whether the name read at `at` is provably an array rather than a tuple
/// carry: it has a non-scalar shape annotation, or every assignment to it in
/// its scope is an array expression. Flow-insensitive, like `name_is_scalar`.
fn name_is_array(
    at: Node,
    name: &str,
    scope_idx: usize,
    ctx: &ShapeCtx,
    visiting: &mut Vec<(usize, String)>,
) -> bool {
    if ctx.original_shapes[scope_idx].get(name).is_some_and(|shape| !shape.is_empty()) {
        return true;
    }
    match nearest_binding(at, name, ctx) {
        (_, NameBinding::Unbound) => false,
        (owner, NameBinding::Opaque) => ctx.inner_arrays.contains(&(owner.id(), name.to_string())),
        (owner, NameBinding::Values(values)) => {
            let key = (owner.id(), name.to_string());
            if visiting.contains(&key) {
                return true;
            }
            visiting.push(key);
            let array = values.iter().all(|v| is_array_expr(*v, ctx, visiting));
            visiting.pop();
            array
        }
    }
}

/// Whether `node` is provably one array: arithmetic on an array, a name that
/// is an array, a numpy, jax or torch constructor or elementwise function, or
/// a shape-only method on an array. Anything else may return a tuple.
fn is_array_expr(node: Node, ctx: &ShapeCtx, visiting: &mut Vec<(usize, String)>) -> bool {
    let text = |n: Node| n.utf8_text(ctx.text.as_bytes()).unwrap_or("");
    match node.kind() {
        "parenthesized_expression" => node
            .named_child(0)
            .is_some_and(|inner| is_array_expr(inner, ctx, visiting)),
        "unary_operator" => node
            .child_by_field_name("argument")
            .is_some_and(|operand| is_array_expr(operand, ctx, visiting)),
        "binary_operator" => ["left", "right"].iter().any(|side| {
            node.child_by_field_name(side)
                .is_some_and(|operand| is_array_expr(operand, ctx, visiting))
        }),
        "identifier" => ctx.links.scope_index.lookup(node.start_byte())
            .is_some_and(|scope_idx| name_is_array(node, text(node), scope_idx, ctx, visiting)),
        "subscript" => node
            .child_by_field_name("value")
            .is_some_and(|value| is_array_expr(value, ctx, visiting)),
        "call" => {
            let Some(func) = node.child_by_field_name("function") else {
                return false;
            };
            if func.kind() == "attribute"
                && let Some(obj) = func.child_by_field_name("object")
                && !is_imported(ctx.imports.at(obj.start_byte()), text(obj))
                && is_array_expr(obj, ctx, visiting)
            {
                return func.child_by_field_name("attribute").is_some_and(|method| {
                    matches!(
                        text(method),
                        "reshape" | "view" | "transpose" | "permute" | "astype" | "flatten"
                            | "ravel" | "squeeze" | "unsqueeze" | "clone" | "detach" | "copy"
                            | "contiguous" | "expand" | "float" | "double" | "long" | "int"
                    )
                });
            }
            let resolved = resolve_call_target(text(func), ctx.imports.at(func.start_byte()));
            let Some((name, module)) = resolved.parts.split_last() else {
                return false;
            };
            if matches!(module, [m] if m == "numpy" || m == "torch")
                || module == ["jax", "numpy"]
                || module == ["jax", "random"]
            {
                is_shape_preserving_call(&resolved)
                    || matches!(
                        name.as_str(),
                        "zeros" | "ones" | "empty" | "full" | "zeros_like" | "ones_like"
                            | "empty_like" | "full_like" | "rand" | "randn" | "randint"
                            | "rand_like" | "randn_like" | "arange" | "linspace" | "eye"
                            | "identity" | "array" | "asarray" | "tensor" | "reshape"
                            | "transpose" | "matmul" | "einsum" | "concatenate" | "concat"
                            | "cat" | "stack" | "squeeze" | "unsqueeze" | "expand_dims"
                            | "swapaxes" | "moveaxis" | "broadcast_to" | "tile" | "repeat"
                            | "flip" | "roll" | "tril" | "triu" | "outer" | "normal"
                            | "uniform" | "bernoulli" | "sum" | "mean" | "prod" | "std" | "var"
                            | "cumsum" | "cumprod"
                    )
            } else {
                false
            }
        }
        _ => false,
    }
}

/// The element nodes of a tuple/list literal RHS (`x, y` / `(x, y)` /
/// `[x, y]`, parentheses peeled), or `None` when the RHS is anything else.
fn literal_elements(rhs: Node) -> Option<Vec<Node>> {
    let mut node = rhs;
    while node.kind() == "parenthesized_expression" {
        node = node.named_child(0)?;
    }
    if !matches!(node.kind(), "expression_list" | "tuple" | "list") {
        return None;
    }
    Some(
        (0..node.named_child_count())
            .filter_map(|i| node.named_child(i as u32))
            .filter(|c| c.kind() != "comment")
            .collect(),
    )
}

/// Per-target shapes for `a, b = x, y` and its starred forms. Every element is
/// evaluated first and only then bound by the caller, so `a, b = b, a` reads
/// the old values. `None` when the lengths cannot match, an element is
/// splatted, or the pattern has more than one star: Python raises or the
/// split is unknowable, so every target goes unknown.
fn tuple_literal_shapes(
    targets: &[TupleTarget],
    elems: &[Node],
    ctx: &mut ShapeCtx,
) -> Option<Vec<Option<Vec<String>>>> {
    if elems.iter().any(|e| e.kind() == "list_splat") {
        return None;
    }
    let mut stars = targets
        .iter()
        .enumerate()
        .filter(|(_, t)| matches!(t, TupleTarget::Star(_)))
        .map(|(i, _)| i);
    let star = stars.next();
    if stars.next().is_some() {
        return None;
    }
    match star {
        None if elems.len() != targets.len() => return None,
        Some(_) if elems.len() + 1 < targets.len() => return None,
        _ => {}
    }

    // Target index each element feeds; `None` for elements the star absorbs.
    let target_of = |ei: usize| match star {
        Some(k) if ei < k => Some(ei),
        Some(k) if ei >= elems.len() - (targets.len() - k - 1) => {
            Some(targets.len() - (elems.len() - ei))
        }
        Some(_) => None,
        None => Some(ei),
    };

    let mut shapes: Vec<Option<Vec<String>>> = vec![None; targets.len()];
    for (ei, elem) in elems.iter().enumerate() {
        let errors_before = ctx.errors.len();
        let shape = shape_of_expression(*elem, ctx);
        let target = target_of(ei);
        let name = match target.map(|ti| &targets[ti]) {
            Some(TupleTarget::Name(Some(n))) => n.clone(),
            Some(TupleTarget::Nested(inner)) => {
                inner.iter().flatten().next().cloned().unwrap_or_default()
            }
            _ => String::new(),
        };
        for err in &mut ctx.errors[errors_before..] {
            err.variable = name.clone();
        }
        let Some(ti) = target else { continue };
        // A nested target unpacks its element: only a tuple/list literal
        // shares one shape across its parts, a tensor splits along axis 0.
        let unpackable = !matches!(targets[ti], TupleTarget::Nested(_))
            || literal_elements(*elem).is_some();
        if unpackable {
            shapes[ti] = shape;
        }
    }
    Some(shapes)
}

/// The dims of `x` for the RHS `x.shape`, when `x` has a known shape.
fn shape_attribute_dims(rhs: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    if rhs.kind() != "attribute" {
        return None;
    }
    let attr = rhs.child_by_field_name("attribute")?;
    if attr.utf8_text(ctx.text.as_bytes()).ok()? != "shape" {
        return None;
    }
    shape_of_expression(rhs.child_by_field_name("object")?, ctx)
}

/// Compute one shape per element of a tuple-unpacked RHS. Returns `None` if the
/// RHS isn't a multi-output form we model. Per-element `None` means "skip this
/// target" (e.g. an integer dim from `x.shape`).
///
/// Modelled forms:
/// - `x.shape` → one zero-rank (scalar) element per dim of `x`.
/// - `jnp.split` / `np.split` / `array_split` / `tensor_split` → the per-array
///   shapes from `compute_split_shapes`.
///
/// Other multi-output functions (`meshgrid`, `svd`, `eig`, …) are not modelled
/// yet and fall through to `None`.
fn tuple_rhs_shapes(
    rhs: Node,
    targets: &[TupleTarget],
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<Option<Vec<String>>>> {
    let n_targets = targets.len();
    match rhs.kind() {
        // `final_mean, final_var = final_carry` / `h_final, c_final =
        // carry_final` — unpacking a plain variable that itself holds a
        // homogeneous-shape tuple carry (bound via `shape_of_tuple`'s
        // same-shape rule, e.g. from a preceding `final_carry, _ =
        // jax.lax.scan(...)`). Every unpacked name gets that one shape, same
        // convention as `TupleTarget::Nested` for a single-statement `(h, c)`
        // pattern. A name that is provably an array unpacks along its first
        // dim (`a, b = x` with `x` `[2, 4]` gives `[4]` each). Any other name
        // could be either, so it stays unknown.
        "identifier" => {
            let name = rhs.utf8_text(ctx.text.as_bytes()).ok()?;
            let shape = ctx.resolve_shape(name, scope_idx)?;
            if ctx.tuple_carries.contains(&(scope_idx, name.to_string())) {
                return Some((0..n_targets).map(|_| Some(shape.clone())).collect());
            }
            if targets.iter().any(|t| matches!(t, TupleTarget::Nested(_)))
                || !name_is_array(rhs, name, scope_idx, ctx, &mut Vec::new())
            {
                return None;
            }
            let (first, rest) = shape.split_first()?;
            if first.parse::<usize>().is_ok_and(|rows| rows != n_targets) {
                return None;
            }
            Some((0..n_targets).map(|_| Some(rest.to_vec())).collect())
        }
        "call" => {
            let func = rhs.child_by_field_name("function")?;
            let target = func.utf8_text(ctx.text.as_bytes()).ok()?;

            // Method-call tuple forms on a plain local receiver: `values,
            // indices = x.topk(k)`, `a, b, c = x.chunk(3)`, `a, b =
            // x.unbind(0)`, `a, b = x.split(2)`, `values, indices =
            // x.kthvalue(k)` / `x.median(dim=1)` / `x.mode(dim=1)`.
            // Distinguished from a qualified free-function path (receiver is
            // an import alias like `jnp`) and from the `self.<attr>` layer
            // case below (handled separately) by requiring the receiver be a
            // plain, non-`self`, non-imported identifier.
            if func.kind() == "attribute"
                && let Some(obj) = func.child_by_field_name("object")
                && obj.kind() == "identifier"
                && let Ok(receiver) = obj.utf8_text(ctx.text.as_bytes())
                && receiver != "self"
                && !is_imported(ctx.imports.at(obj.start_byte()), receiver)
                && let Some(method_name_node) = func.child_by_field_name("attribute")
                && let Ok(method_name) = method_name_node.utf8_text(ctx.text.as_bytes())
                && let Some(known) = classify_method_call(method_name)
                && matches!(
                    known,
                    KnownFunction::TopK
                        | KnownFunction::Chunk
                        | KnownFunction::Unbind
                        | KnownFunction::TorchSplit
                        | KnownFunction::KthValue
                        | KnownFunction::MedianDim
                )
            {
                let args_node = rhs.child_by_field_name("arguments")?;
                let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
                let mut method_args = Vec::with_capacity(raw_args.len() + 1);
                method_args.push(CallArgument::Positional {
                    value: receiver.to_string(),
                });
                method_args.extend(raw_args);
                return tuple_multi_output_shapes(
                    &known,
                    &method_args,
                    args_node,
                    n_targets,
                    scope_idx,
                    ctx,
                );
            }

            // `attn_out, attn_weights = self.attn(q, k, v)` where self.attn
            // is a MultiheadAttention built in __init__: output has the
            // query's shape; weights are (..., L, S) with default
            // average_attn_weights.
            if let Some(attr) = target.strip_prefix("self.")
                && matches!(
                    ctx.self_attr_layer_at(attr, rhs.start_byte()),
                    Some(LayerKind::MultiheadAttention { .. })
                )
            {
                if n_targets != 2 {
                    return None;
                }
                let args_node = rhs.child_by_field_name("arguments")?;
                let args = extract_call_arguments(args_node, ctx.text).ok()?;
                let mut positionals = args.iter().filter_map(|a| match a {
                    CallArgument::Positional { value } => Some(value.as_str()),
                    _ => None,
                });
                let query = positionals.next()?;
                let key = positionals.next().unwrap_or(query);
                let query_shape = ctx.resolve_shape(query, scope_idx)?;
                if query_shape.len() < 2 {
                    return None;
                }
                let weights = ctx.resolve_shape(key, scope_idx).and_then(|key_shape| {
                    let s = key_shape.get(key_shape.len().checked_sub(2)?)?.clone();
                    let mut w = query_shape.clone();
                    w.pop();
                    w.push(s);
                    Some(w)
                });
                return Some(vec![Some(query_shape), weights]);
            }

            // `out, h = self.gru(x)` / `out, (h, c) = self.lstm(x)` —
            // self.<attr> resolves to an RNN full-sequence layer
            // (`LayerKind::Rnn`) built in `__init__`. Element 0 (`out`) is
            // the full sequence output: same rule as `apply_layer_kind`'s
            // `Rnn` arm (last dim -> `hidden_size`, all other dims
            // preserved). Element 1 is the final state — for LSTM's nested
            // `(h, c)` pattern both members are bound to this same shape
            // (see `handle_tuple_assignment`'s `TupleTarget::Nested` arm),
            // which is exactly right for the *shape* (real torch `h`/`c` do
            // share one shape) even though only `h` is truly "the state" for
            // a GRU/RNN. The shape itself is approximated by dropping the
            // leading (sequence) axis and replacing the last dim with
            // `hidden_size` — this assumes `batch_first=False` (torch's
            // default, not tracked by `LayerKind::Rnn`) and `num_layers *
            // num_directions == 1` (also not tracked), so for a batched
            // input it is missing the real leading `num_layers*num_
            // directions` axis that torch's actual `h_n`/`c_n` carries.
            // Documented as an approximation in TO_IMPLEMENT.md.
            if let Some(attr) = target.strip_prefix("self.") {
                let rnn_hidden_size = match ctx.self_attr_layer_at(attr, rhs.start_byte()) {
                    Some(LayerKind::Rnn { hidden_size, .. }) => Some(hidden_size),
                    _ => None,
                };
                if let Some(hidden_size) = rnn_hidden_size {
                    if n_targets != 2 || hidden_size.contains(UNPROVEN_DIM) {
                        return None;
                    }
                    let args_node = rhs.child_by_field_name("arguments")?;
                    let args = extract_call_arguments(args_node, ctx.text).ok()?;
                    let input = args.iter().find_map(|a| match a {
                        CallArgument::Positional { value } => Some(value.as_str()),
                        _ => None,
                    })?;
                    let input_shape = ctx.resolve_shape(input, scope_idx)?;
                    if input_shape.len() < 2 {
                        return None;
                    }

                    let mut output = input_shape.clone();
                    let last = output.len() - 1;
                    output[last] = hidden_size.clone();

                    let mut state = input_shape[1..].to_vec();
                    if let Some(last_dim) = state.last_mut() {
                        *last_dim = hidden_size;
                    }

                    return Some(vec![Some(output), Some(state)]);
                }
            }

            // `h, c = self.lstm(x, (h0, c0))` — self.<attr> resolves to a
            // single-step RNN cell (`LayerKind::RnnCell`: LSTMCell/GRUCell/
            // RNNCell) built in `__init__`. Same "both share one shape"
            // approximation as the `Rnn` arm's `(h, c)` state above, applied
            // to the single-step rule (`apply_layer_kind`'s `RnnCell` arm:
            // last dim -> `hidden_size`, rank preserved, min rank 1 for the
            // unbatched convention).
            if let Some(attr) = target.strip_prefix("self.") {
                let cell_hidden_size = match ctx.self_attr_layer_at(attr, rhs.start_byte()) {
                    Some(LayerKind::RnnCell { hidden_size, .. }) => Some(hidden_size),
                    _ => None,
                };
                if let Some(hidden_size) = cell_hidden_size {
                    if n_targets != 2 || hidden_size.contains(UNPROVEN_DIM) {
                        return None;
                    }
                    let args_node = rhs.child_by_field_name("arguments")?;
                    let args = extract_call_arguments(args_node, ctx.text).ok()?;
                    let input = args.iter().find_map(|a| match a {
                        CallArgument::Positional { value } => Some(value.as_str()),
                        _ => None,
                    })?;
                    let input_shape = ctx.resolve_shape(input, scope_idx)?;
                    if input_shape.is_empty() {
                        return None;
                    }
                    let mut output = input_shape.clone();
                    let last = output.len() - 1;
                    output[last] = hidden_size;
                    return Some(vec![Some(output.clone()), Some(output)]);
                }
            }

            let resolved = resolve_call_target(target, ctx.imports.at(rhs.start_byte()));
            let known = classify_known_function(&resolved);

            // `final_carry, ys = jax.lax.scan(body, init, xs)` — the carry is
            // invariant, so final_carry gets init's shape. `init` is
            // evaluated as a real node (not just an identifier lookup) so an
            // inline tuple carry (`scan(body, (h0, c0), xs)`) or a
            // previously-bound tuple-literal variable (`init = (mean0,
            // var0)`) both resolve via `shape_of_tuple`'s homogeneous-shape
            // rule, same as a plain single-array carry. `ys` needs the
            // body's per-step output shape: when `body` is a same-file
            // function/closure, seed its params (carry <- init's shape, the
            // per-step element <- `xs`'s shape minus its leading axis) and
            // evaluate its body lazily (`scan_body_ys_shape`); `ys` is then
            // the body's second return element with `xs`'s leading dim
            // prepended. Anything not modelled (qualified/`self.` body,
            // missing `xs`, un-traceable body return, …) evicts any stale
            // `ys` binding rather than guessing.
            if matches!(known, Some(KnownFunction::Scan)) {
                if n_targets != 2 {
                    return None;
                }
                let args_node = rhs.child_by_field_name("arguments")?;
                let body_node = find_keyword_arg_node(args_node, "f", ctx.text.as_bytes())
                    .and_then(|kw| kw.child_by_field_name("value"))
                    .or_else(|| find_positional_arg_node(args_node, 0));
                let init_node = scan_init_node(args_node, ctx.text)?;
                let init_shape = shape_of_expression(init_node, ctx)?;
                let init_is_tuple = is_tuple_value(init_node, scope_idx, ctx);
                let xs_node = find_keyword_arg_node(args_node, "xs", ctx.text.as_bytes())
                    .and_then(|kw| kw.child_by_field_name("value"))
                    .or_else(|| find_positional_arg_node(args_node, 2));

                let ys_shape = match (body_node, xs_node) {
                    (Some(body_node), Some(xs_node)) => {
                        scan_body_ys_shape(
                            body_node,
                            &init_shape,
                            init_is_tuple,
                            xs_node,
                            rhs.start_byte(),
                            ctx,
                        )
                    }
                    _ => None,
                };

                return Some(vec![Some(init_shape), ys_shape]);
            }

            // JAX tuple returns use the same validation as single-name calls,
            // but keep each output distinct. Resolve inline array expressions.
            if matches!(known, Some(KnownFunction::LaxTopK | KnownFunction::QrMultiply)) {
                let args_node = rhs.child_by_field_name("arguments")?;
                let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
                let args = resolve_call_args(raw_args, args_node, scope_idx, ctx)?;
                let result = if matches!(known, Some(KnownFunction::LaxTopK)) {
                    crate::known_functions::compute_top_k_shape(&args, &ctx.scope_shapes(scope_idx))
                        .map(|shape| shape.map(|s| vec![s.clone(), s]))
                } else {
                    crate::known_functions::compute_qr_multiply_shapes(&args, &ctx.scope_shapes(scope_idx))
                };
                return match result {
                    Ok(Some(outputs)) if outputs.len() == n_targets => {
                        Some(outputs.into_iter().map(Some).collect())
                    }
                    Ok(_) => None,
                    Err(message) => {
                        ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
                        None
                    }
                };
            }

            // `sorted_keys, sorted_values = jax.lax.sort_key_val(keys, values)`
            // — each output preserves its own operand's shape.
            if matches!(known, Some(KnownFunction::LaxSortKeyVal)) {
                if n_targets != 2 {
                    return None;
                }
                let args_node = rhs.child_by_field_name("arguments")?;
                let args = extract_call_arguments(args_node, ctx.text).ok()?;
                let mut positionals = args.iter().filter_map(|a| match a {
                    CallArgument::Positional { value } => Some(value.as_str()),
                    _ => None,
                });
                let keys = positionals.next()?;
                let values = positionals.next()?;
                let keys_shape = ctx.resolve_shape(keys, scope_idx)?;
                let values_shape = ctx.resolve_shape(values, scope_idx)?;
                return Some(vec![Some(keys_shape), Some(values_shape)]);
            }

            // `a, b, c = np.hsplit(x, 3)` / `vsplit` / `dsplit` — like
            // `split` but the axis is implied by the function name, not an
            // argument.
            if matches!(
                known,
                Some(KnownFunction::HSplit | KnownFunction::VSplit | KnownFunction::DSplit)
            ) {
                let args_node = rhs.child_by_field_name("arguments")?;
                let args = extract_call_arguments(args_node, ctx.text).ok()?;
                return match compute_fixed_axis_split_shapes(
                    &known?,
                    &args,
                    &ctx.scope_shapes(scope_idx),
                ) {
                    Ok(Some(shapes)) => Some(shapes.into_iter().map(Some).collect()),
                    Ok(None) => None,
                    Err(message) => {
                        ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
                        None
                    }
                };
            }

            // `i0, i1, ... = jnp.nonzero(x)` — one 1D output per input
            // dimension, all sharing the same data-dependent length.
            if matches!(known, Some(KnownFunction::Nonzero)) {
                if n_targets == 0 {
                    return None;
                }
                let args_node = rhs.child_by_field_name("arguments")?;
                let args = extract_call_arguments(args_node, ctx.text).ok()?;
                let input_name = args.iter().find_map(|a| match a {
                    CallArgument::Positional { value } => Some(value.as_str()),
                    _ => None,
                })?;
                let input_shape = ctx.resolve_shape(input_name, scope_idx)?;
                if input_shape.len() != n_targets {
                    return None;
                }
                let dim = format!("nonzero({input_name})");
                return Some(vec![Some(vec![dim]); n_targets]);
            }

            // `packed, ps = einops.pack([...], pattern)` — the packed array
            // gets a real shape (restricted case, see
            // `compute_einops_pack_shape`); `ps` (packed_shapes) is a list
            // of shape-tuples, not an array, so it's always `None`.
            if matches!(known, Some(KnownFunction::EinopsPack)) {
                if n_targets != 2 {
                    return None;
                }
                let args_node = rhs.child_by_field_name("arguments")?;
                let args = extract_call_arguments(args_node, ctx.text).ok()?;
                return match compute_einops_pack_shape(&args, &ctx.scope_shapes(scope_idx)) {
                    Ok(shape) => Some(vec![shape, None]),
                    Err(message) => {
                        ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
                        None
                    }
                };
            }

            // `a, b, ... = einops.unpack(packed, ps, pattern)` — genuinely
            // dynamic (depends on the runtime `ps` value); conservatively
            // unknown for every target.
            if matches!(known, Some(KnownFunction::EinopsUnpack)) {
                if n_targets == 0 {
                    return None;
                }
                return Some(vec![None; n_targets]);
            }

            // `solution, residuals, rank, sv = torch.linalg.lstsq(A, B)` (or
            // any prefix of that tuple) — only `solution`'s shape is
            // derivable; the rest are algorithm/data-dependent.
            if matches!(known, Some(KnownFunction::LinalgLstsq)) {
                let args_node = rhs.child_by_field_name("arguments")?;
                let args = extract_call_arguments(args_node, ctx.text).ok()?;
                let mut positionals = args.iter().filter_map(|a| match a {
                    CallArgument::Positional { value } => Some(value.as_str()),
                    _ => None,
                });
                let a_name = positionals.next()?;
                let b_name = positionals.next()?;
                let a_shape = ctx.resolve_shape(a_name, scope_idx)?;
                let b_shape = ctx.resolve_shape(b_name, scope_idx)?;
                let solution = apply_known_linalg_lstsq_solution(&a_shape, &b_shape)?;
                let mut result = vec![Some(solution)];
                result.resize(n_targets, None);
                return Some(result);
            }

            if matches!(
                known,
                Some(
                    KnownFunction::AtLeast1D
                        | KnownFunction::AtLeast2D
                        | KnownFunction::AtLeast3D
                        | KnownFunction::BroadcastArrays
                )
            ) {
                let args_node = rhs.child_by_field_name("arguments")?;
                let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
                if raw_args.len() != n_targets
                    || raw_args
                        .iter()
                        .any(|arg| !matches!(arg, CallArgument::Positional { .. }))
                {
                    return None;
                }
                let args = resolve_call_args(raw_args, args_node, scope_idx, ctx)?;
                let known = known.as_ref()?;
                if matches!(known, KnownFunction::BroadcastArrays) {
                    return match apply_known_function(known, &args, &ctx.scope_shapes(scope_idx)) {
                        Ok(Some(shape)) => Some(vec![Some(shape); n_targets]),
                        Ok(None) => None,
                        Err(message) => {
                            ctx.errors.push(ShapeError::mismatch(
                                String::new(),
                                message,
                                args_node.range(),
                            ));
                            None
                        }
                    };
                }
                if n_targets < 2 {
                    return None;
                }
                let mut shapes = Vec::with_capacity(n_targets);
                for arg in &args {
                    match apply_known_function(
                        known,
                        std::slice::from_ref(arg),
                        &ctx.scope_shapes(scope_idx),
                    ) {
                        Ok(Some(shape)) => shapes.push(Some(shape)),
                        Ok(None) => return None,
                        Err(message) => {
                            ctx.errors.push(ShapeError::mismatch(
                                String::new(),
                                message,
                                args_node.range(),
                            ));
                            return None;
                        }
                    }
                }
                return Some(shapes);
            }

            // Multi-output linear algebra + meshgrid (v1: default modes only —
            // any keyword argument like full_matrices/mode/indexing skips).
            if matches!(
                known,
                Some(
                    KnownFunction::LinalgSvd
                        | KnownFunction::LinalgQr
                        | KnownFunction::LinalgEig
                        | KnownFunction::Meshgrid
                )
            ) {
                let args_node = rhs.child_by_field_name("arguments")?;
                let args = extract_call_arguments(args_node, ctx.text).ok()?;
                if args
                    .iter()
                    .any(|a| matches!(a, CallArgument::Keyword { .. }))
                {
                    return None;
                }
                return linalg_tuple_shapes(&known?, &args, n_targets, scope_idx, ctx);
            }

            // `values, indices = torch.topk(x, 3)` / `torch.chunk(x, 3)` /
            // `torch.unbind(x)` / `torch.kthvalue(x, 2)` /
            // `torch.median(x, dim=1)` / `torch.mode(x, dim=1)` /
            // `torch.split(x, 3, dim=0)` — the qualified free-function forms
            // of the method-call dispatch handled at the top of this arm.
            if matches!(
                known,
                Some(
                    KnownFunction::TopK
                        | KnownFunction::MinMax
                        | KnownFunction::Chunk
                        | KnownFunction::Unbind
                        | KnownFunction::KthValue
                        | KnownFunction::MedianDim
                        | KnownFunction::TorchSplit
                )
            ) {
                let args_node = rhs.child_by_field_name("arguments")?;
                let mut args = extract_call_arguments(args_node, ctx.text).ok()?;
                if matches!(known, Some(KnownFunction::TopK | KnownFunction::MinMax)) {
                    if let Some((i, input)) = args.iter_mut().enumerate().find(|(_, arg)| {
                        matches!(arg, CallArgument::Positional { .. })
                            || matches!(arg, CallArgument::Keyword { name, .. }
                                if matches!(name.as_str(), "a" | "x" | "input" | "array" | "y"))
                    }) {
                        *input = resolve_call_arg(input.clone(), args_node.named_child(i as u32)?, scope_idx, ctx)?;
                    }
                } else {
                    args = resolve_call_args(args, args_node, scope_idx, ctx)?;
                }
                return tuple_multi_output_shapes(&known?, &args, args_node, n_targets, scope_idx, ctx);
            }

            if !matches!(known, Some(KnownFunction::Split | KnownFunction::ArraySplit)) {
                return None;
            }
            let args_node = rhs.child_by_field_name("arguments")?;
            let args = extract_call_arguments(args_node, ctx.text).ok()?;
            let result = if matches!(known, Some(KnownFunction::ArraySplit)) {
                compute_array_split_shapes(&args, &ctx.scope_shapes(scope_idx))
            } else {
                compute_split_shapes(&args, &ctx.scope_shapes(scope_idx))
            };
            match result {
                Ok(Some(shapes)) => Some(shapes.into_iter().map(Some).collect()),
                Ok(None) => None,
                Err(message) => {
                    ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
                    None
                }
            }
        }
        _ => None,
    }
}

/// Shared math for tuple-unpacked `topk`/`chunk`/`unbind`/`kthvalue`/
/// `median`/`mode` calls, in either their free-function form (`args[0]` is
/// the input) or their method-call form (`args[0]` is the receiver,
/// prepended by the caller) — the argument layout is identical either way.
fn tuple_multi_output_shapes(
    known: &KnownFunction,
    args: &[CallArgument],
    args_node: Node,
    n_targets: usize,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<Option<Vec<String>>>> {
    match known {
        KnownFunction::TopK => {
            if n_targets != 2 {
                return None;
            }
            match apply_known_topk_shape(args, &ctx.scope_shapes(scope_idx)) {
                Ok(Some(shape)) => Some(vec![Some(shape.clone()), Some(shape)]),
                Ok(None) => None,
                Err(message) => {
                    ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
                    None
                }
            }
        }
        KnownFunction::KthValue => {
            if n_targets != 2 {
                return None;
            }
            match apply_known_kthvalue_shape(args, &ctx.scope_shapes(scope_idx)) {
                Ok(Some(shape)) => Some(vec![Some(shape.clone()), Some(shape)]),
                Ok(None) => None,
                Err(message) => {
                    ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
                    None
                }
            }
        }
        KnownFunction::MinMax | KnownFunction::MedianDim => {
            if n_targets != 2 {
                return None;
            }
            let positional_count = args
                .iter()
                .filter(|a| matches!(a, CallArgument::Positional { .. }))
                .count();
            let has_dim = positional_count >= 2
                || args
                    .iter()
                    .any(|a| matches!(a, CallArgument::Keyword { name, .. } if name == "dim" || name == "axis"));
            if !has_dim && matches!(known, KnownFunction::MedianDim) {
                // No-dim scalar form (`torch.median(x)`) isn't a 2-tuple.
                return None;
            }
            match apply_known_reduction(args, &ctx.scope_shapes(scope_idx)) {
                Ok(Some(shape)) => Some(vec![Some(shape.clone()), Some(shape)]),
                Ok(None) => None,
                Err(message) => {
                    ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
                    None
                }
            }
        }
        KnownFunction::Unbind => {
            if n_targets == 0 {
                return None;
            }
            match compute_unbind_shape(args, &ctx.scope_shapes(scope_idx), n_targets) {
                Ok(Some(shape)) => Some(vec![Some(shape); n_targets]),
                Ok(None) => None,
                Err(message) => {
                    ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
                    None
                }
            }
        }
        KnownFunction::Chunk => match compute_chunk_shapes(args, &ctx.scope_shapes(scope_idx)) {
            Ok(Some(shapes)) => Some(shapes.into_iter().map(Some).collect()),
            Ok(None) => None,
            Err(message) => {
                ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
                None
            }
        },
        // Real torch semantics (chunk *size*, not section count) — covers
        // both `torch.split(x, ...)` (free function) and `x.split(...)`
        // (method, receiver prepended by the caller). `n_targets` lets the
        // literal-size / symbolic-axis-dim case still bind a chunk list.
        KnownFunction::TorchSplit => {
            match compute_torch_split_shapes(args, &ctx.scope_shapes(scope_idx), Some(n_targets)) {
                Ok(Some(shapes)) => Some(shapes.into_iter().map(Some).collect()),
                Ok(None) => None,
                Err(message) => {
                    ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
                    None
                }
            }
        }
        _ => None,
    }
}

/// `min(a, b)` as a dim: computed when both concrete, `a` when they match
/// textually, an opaque `min(a,b)` symbol otherwise.
fn min_dim(a: &str, b: &str) -> String {
    if let (Ok(x), Ok(y)) = (a.parse::<usize>(), b.parse::<usize>()) {
        return x.min(y).to_string();
    }
    if a == b {
        return a.to_string();
    }
    format!("min({},{})", a, b)
}

/// Per-target shapes for multi-output linalg calls and meshgrid, default
/// modes only (numpy semantics):
/// - `u, s, vt = svd(a)`: a (m, n) → (m, m), (min(m,n),), (n, n)
/// - `q, r = qr(a)`: a (m, n) → (m, min(m,n)), (min(m,n), n)
/// - `evals, evecs = eig/eigh(a)`: a (…, n) → (n,), (n, n)
/// - `gx, gy = meshgrid(xs, ys)`: 'xy' indexing → both (ny, nx)
fn linalg_tuple_shapes(
    known: &KnownFunction,
    args: &[CallArgument],
    n_targets: usize,
    scope_idx: usize,
    ctx: &ShapeCtx,
) -> Option<Vec<Option<Vec<String>>>> {
    let positionals: Vec<&str> = args
        .iter()
        .filter_map(|a| match a {
            CallArgument::Positional { value } => Some(value.as_str()),
            _ => None,
        })
        .collect();

    match known {
        KnownFunction::Meshgrid => {
            if positionals.len() != n_targets {
                return None;
            }
            let mut lens = Vec::with_capacity(positionals.len());
            for name in &positionals {
                let shape = ctx.resolve_shape(name, scope_idx)?;
                let [len] = shape.as_slice() else {
                    return None;
                };
                lens.push(len.clone());
            }
            // Default 'xy' indexing swaps the first two axes.
            if lens.len() >= 2 {
                lens.swap(0, 1);
            }
            Some(vec![Some(lens); n_targets])
        }
        _ => {
            let input = positionals.first()?;
            let shape = ctx.resolve_shape(input, scope_idx)?;
            let [m, n] = shape.as_slice() else {
                return None;
            };
            match known {
                KnownFunction::LinalgSvd => {
                    if n_targets != 3 {
                        return None;
                    }
                    let k = min_dim(m, n);
                    Some(vec![
                        Some(vec![m.clone(), m.clone()]),
                        Some(vec![k]),
                        Some(vec![n.clone(), n.clone()]),
                    ])
                }
                KnownFunction::LinalgQr => {
                    if n_targets != 2 {
                        return None;
                    }
                    let k = min_dim(m, n);
                    Some(vec![
                        Some(vec![m.clone(), k.clone()]),
                        Some(vec![k, n.clone()]),
                    ])
                }
                KnownFunction::LinalgEig => {
                    if n_targets != 2 {
                        return None;
                    }
                    Some(vec![
                        Some(vec![n.clone()]),
                        Some(vec![n.clone(), n.clone()]),
                    ])
                }
                _ => None,
            }
        }
    }
}

/// Collect the inner expression nodes from return_statement, yield, and
/// assert_statement nodes. These are value-position contexts where
/// shape errors should be detected but no shape is assigned to a variable.
fn collect_value_statements<'a>(
    node: Node<'a>,
    text: &str,
) -> Result<Vec<Node<'a>>, String> {
    let mut result = Vec::new();
    collect_value_statements_recursive(node, text, &mut result)?;
    result.sort_by_key(|n| n.start_byte());
    Ok(result)
}

fn collect_value_statements_recursive<'a>(
    node: Node<'a>,
    text: &str,
    out: &mut Vec<Node<'a>>,
) -> Result<(), String> {
    for i in 0..node.named_child_count() {
        collect_value_statements_of_child(node.named_child(i as u32).unwrap(), text, out)?;
    }
    Ok(())
}

/// The value statements `collect_value_statements_recursive` finds for one
/// child of its node.
fn collect_value_statements_of_child<'a>(
    child: Node<'a>,
    text: &str,
    out: &mut Vec<Node<'a>>,
) -> Result<(), String> {
    match child.kind() {
        "return_statement" | "yield" | "assert_statement" => {
            // The value expression is the first named child.
            // If it's an expression_list (tuple), expand into individual
            // expressions so each element gets evaluated.
            if let Some(expr) = child.named_child(0) {
                if expr.kind() == "expression_list" {
                    for j in 0..expr.named_child_count() {
                        if let Some(elem) = expr.named_child(j as u32) {
                            out.push(elem);
                        }
                    }
                } else {
                    out.push(expr);
                }
            }
        }
        "expression_statement" => {
            // Bare call expressions like `jnp.matmul(x, y)` whose result
            // is discarded.  Also handle yield/return/assert inside
            // expression_statement wrappers.
            // If the immediate child is a call, evaluate it so that
            // shape errors are detected even without an assignment.
            if let Some(expr) = child.named_child(0) {
                match expr.kind() {
                    "call" => out.push(expr),
                    "yield" | "return_statement" | "assert_statement" => {
                        // Unwrap the value expression from these
                        // statement-like wrappers.
                        if let Some(inner) = expr.named_child(0) {
                            if inner.kind() == "expression_list" {
                                for j in 0..inner.named_child_count() {
                                    if let Some(elem) = inner.named_child(j as u32) {
                                        out.push(elem);
                                    }
                                }
                            } else {
                                out.push(inner);
                            }
                        }
                    }
                    _ => {}
                }
            }
            // Don't recurse — assignments inside expression_statement
            // are already handled by collect_assignment_items.
        }
        _ => {
            collect_value_statements_recursive(child, text, out)?;
        }
    }
    Ok(())
}

/// Expand a call to a vmap-bound name.
///
/// `info` describes the vmap binding (wrapped function name, in_axes, out_axes).
/// `args` are the call's arguments (positional variable names).
/// `caller_shapes` are the caller's known shapes.
/// `scopes` are all function scopes (used to find the wrapped function).
///
/// Returns:
/// - `Ok(Some(shape))` — output shape with batch dim prepended.
/// - `Ok(None)` — silently skipped (missing shapes, no annotations, etc.).
/// - `Err(msg)` — shape error (rank mismatch, batch dim disagreement, etc.).
fn apply_vmap_call(
    info: &VmapInfo,
    args: &[CallArgument],
    caller_shapes: &dyn ShapeLookup,
    links: &ScopeLinks,
    scopes: &[FunctionShapeScope],
) -> Result<Option<Vec<String>>, String> {
    apply_vmap_call_chain(
        &[(info.in_axes, info.out_axes)],
        &info.wrapped,
        info.site,
        args,
        caller_shapes,
        links,
        scopes,
    )
}

/// The peeled arguments, each with its name, and the size of every peeled axis.
type Peeled<'a> = (Vec<(&'a str, Vec<String>)>, Vec<String>);

/// The positional arguments of a vmapped call with one leading axis peeled per
/// level of `axes`, and the size of each axis. Checks that the arguments agree
/// on every size. `None` when an argument has no shape or there is none.
fn peel_vmap_args<'a>(
    axes: &[(isize, isize)],
    args: &'a [CallArgument],
    caller_shapes: &dyn ShapeLookup,
) -> Result<Option<Peeled<'a>>, String> {
    let mut current: Vec<(&str, Vec<String>)> = Vec::new();
    for arg in args {
        let CallArgument::Positional { value } = arg else {
            // Skip non-positional args silently (v1 doesn't pass kwargs through vmap).
            continue;
        };
        let Some(shape) = caller_shapes.shape(value.as_str()) else {
            // Arg has no known shape — skip silently.
            return Ok(None);
        };
        current.push((value.as_str(), shape.clone()));
    }
    if current.is_empty() {
        // No positional args at all — can't determine batch dims.
        return Ok(None);
    }

    let mut batch_dims: Vec<String> = Vec::with_capacity(axes.len());
    for &(in_axes, _) in axes {
        let mut level_dim: Option<String> = None;
        let mut next = Vec::with_capacity(current.len());
        for (name, shape) in &current {
            match peel_batch_dim(shape, in_axes) {
                Ok((peeled, dim)) => {
                    if let Some(ref existing) = level_dim {
                        if existing != &dim {
                            return Err(format!(
                                "vmap input batch dims disagree: '{}' vs '{}'",
                                existing, dim
                            ));
                        }
                    } else {
                        level_dim = Some(dim);
                    }
                    next.push((*name, peeled));
                }
                Err(msg) => {
                    return Err(format!(
                        "vmap: argument '{}' rank insufficient for in_axes={}: {}",
                        name, in_axes, msg
                    ));
                }
            }
        }
        current = next;
        // `current` is non-empty (checked above and preserved per level), so
        // `level_dim` is always `Some` here.
        batch_dims.push(level_dim.unwrap());
    }
    Ok(Some((current, batch_dims)))
}

/// Generalized form of `apply_vmap_call` for nested vmaps, e.g.
/// `jax.vmap(jax.vmap(f))(x)`. `axes` holds `(in_axes, out_axes)` pairs
/// ordered outermost-to-innermost — one entry per vmap layer wrapping `f`.
/// Peels one leading batch dim per level (checking cross-argument agreement
/// at each level independently), resolves `f`'s shape on the fully-peeled
/// per-example inputs, then re-prepends the batch dims outward-to-inward
/// in reverse.
fn apply_vmap_call_chain(
    axes: &[(isize, isize)],
    wrapped: &str,
    site: usize,
    args: &[CallArgument],
    caller_shapes: &dyn ShapeLookup,
    links: &ScopeLinks,
    scopes: &[FunctionShapeScope],
) -> Result<Option<Vec<String>>, String> {
    let Some((current, batch_dims)) = peel_vmap_args(axes, args, caller_shapes)? else {
        return Ok(None);
    };

    // Find the wrapped function's FunctionShapeScope.
    let callee = match find_callee_scope(wrapped, site, links, scopes) {
        Some(idx) => &scopes[idx],
        None => return Ok(None), // wrapped function not found — skip silently
    };

    // If the callee has no jaxtyping annotations at all, skip.
    if callee.shapes.is_empty() && callee.return_shape.is_none() {
        return Ok(None);
    }

    // Map positional arg shapes to param names (using param_order),
    // then bind and substitute.
    let param_names = &callee.param_order;
    let arg_shapes: Vec<(&str, Vec<String>)> = current
        .iter()
        .enumerate()
        .filter_map(|(idx, (_, shape))| param_names.get(idx).map(|p| (p.as_str(), shape.clone())))
        .collect();

    let result = bind_and_substitute(callee, wrapped, &arg_shapes)?;

    // If return_shape is None after substitution, no output to propagate.
    let Some(substituted) = result else {
        return Ok(None);
    };

    // Re-prepend the batch dims, innermost level first.
    let mut output = substituted;
    for (&(_, out_axes), dim) in axes.iter().zip(batch_dims).rev() {
        output = prepend_batch_dim(output, out_axes, dim);
    }

    Ok(Some(output))
}

/// Resolve an inline vmap application `vmap(callable)(args)` (issue #35),
/// including nested `vmap(vmap(callable))(args)` chains.
///
/// `inner_call` is the outermost `vmap(...)` call; `outer_args_node` holds
/// the arguments applied to it. Returns the batched output shape, or `None`
/// if the inner call isn't a vmap chain we can model.
fn shape_of_inline_vmap(
    inner_call: Node,
    outer_args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let (callable_node, axes) = collect_vmap_axes(inner_call, ctx)?;
    if callable_node.kind() == "lambda" {
        return vmapped_lambda(callable_node, &axes, outer_args_node, scope_idx, ctx);
    }
    let callable = callable_node.utf8_text(ctx.text.as_bytes()).ok()?.to_string();

    let outer_args = extract_call_arguments(outer_args_node, ctx.text).ok()?;

    // Case A: the vmapped callable is `self.<attr>` resolving to a known layer.
    if let Some(attr) = callable.strip_prefix("self.")
        && let Some(layer) = ctx
            .self_attr_layer_at(attr, outer_args_node.start_byte())
    {
        let outer_args = resolve_call_args(outer_args, outer_args_node, scope_idx, ctx)?;
        return apply_inline_vmap_layer_chain(
            (&callable, &layer),
            &outer_args,
            &axes,
            scope_idx,
            outer_args_node.range(),
            ctx,
        );
    }

    // Case A2: `self.<attr>` is a user module of this file, or a field that is
    // not one (which never falls back to a function of the same name).
    if let Some(attr) = callable.strip_prefix("self.")
        && let Some(field) = ctx
            .submodule_at(attr, outer_args_node.start_byte())
            .map(|field| field.module.clone())
    {
        let Some(Field::One(Module::Local(module))) = field else {
            return None;
        };
        let outer_args = resolve_call_args(outer_args, outer_args_node, scope_idx, ctx)?;
        return apply_vmapped_submodule(&module, &callable, &outer_args, &axes, scope_idx, outer_args_node, ctx);
    }

    // Case B: the callable is a bare identifier — a user function in this file.
    if !callable.contains('.') {
        let outer_args = resolve_call_args(outer_args, outer_args_node, scope_idx, ctx)?;
        return match apply_vmap_call_chain(
            &axes,
            &callable,
            inner_call.start_byte(),
            &outer_args,
            &ctx.scope_shapes(scope_idx),
            ctx.links,
            ctx.scopes,
        ) {
            Ok(shape) => shape,
            Err(message) => {
                ctx.errors.push(ShapeError::mismatch(String::new(), message, outer_args_node.range()));
                None
            }
        };
    }

    None
}

/// `jax.vmap(lambda x: ...)(xs)`, also nested: the body is evaluated on the
/// shapes of one example, then the batch dims are put back. A keyword argument,
/// a tuple, or an argument without a shape leaves the result unknown.
fn vmapped_lambda(
    lambda: Node,
    axes: &[(isize, isize)],
    args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let mut cursor = args_node.walk();
    let all_arrays = args_node
        .named_children(&mut cursor)
        .filter(|arg| arg.kind() != "comment")
        .all(|arg| is_single_array_argument(arg, scope_idx, ctx));
    if !all_arrays {
        return None;
    }
    let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
    let args = resolve_call_args(raw_args, args_node, scope_idx, ctx)?;
    batched_lambda(lambda, axes, &args, args_node, scope_idx, ctx)
}

/// `jax.lax.map(lambda x: ..., xs)`: the lambda runs on one row of `xs`.
fn map_lambda(
    lambda: Node,
    args_node: Node,
    xs: &[CallArgument],
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let mut cursor = args_node.walk();
    let nodes: Vec<Node> = args_node
        .named_children(&mut cursor)
        .filter(|arg| arg.kind() != "comment")
        .collect();
    let [_, xs_node] = nodes[..] else {
        return None;
    };
    if !is_single_array_argument(xs_node, scope_idx, ctx) {
        return None;
    }
    batched_lambda(lambda, &[(0, 0)], xs, args_node, scope_idx, ctx)
}

fn batched_lambda(
    lambda: Node,
    axes: &[(isize, isize)],
    args: &[CallArgument],
    args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let peeled = peel_vmap_args(axes, args, &ctx.scope_shapes(scope_idx));
    let (examples, batch_dims) = match peeled {
        Ok(Some((examples, batch_dims))) => (examples, batch_dims),
        Ok(None) => return None,
        Err(message) => {
            ctx.errors.push(ShapeError::mismatch(String::new(), message, args_node.range()));
            return None;
        }
    };
    let examples = examples.into_iter().map(|(_, shape)| shape).collect();
    let mut output = shape_of_batched_lambda(lambda, scope_idx, examples, ctx)?;
    for (&(_, out_axes), dim) in axes.iter().zip(batch_dims).rev() {
        output = prepend_batch_dim(output, out_axes, dim);
    }
    Some(output)
}

/// `jax.vmap(self.blk)(x)` for a user module: peel the batch dim of every
/// argument, call the module on the per-example shapes, and restore the batch
/// dims. An argument without a shape, a keyword argument, or batch dims that
/// differ leave the result unknown.
fn apply_vmapped_submodule(
    module: &Submodule,
    display: &str,
    args: &[CallArgument],
    axes: &[(isize, isize)],
    scope_idx: usize,
    args_node: Node,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let mut shapes = Vec::with_capacity(args.len());
    for arg in args {
        let CallArgument::Positional { value } = arg else {
            return None;
        };
        shapes.push(ctx.resolve_shape(value, scope_idx)?);
    }
    if shapes.is_empty() {
        return None;
    }
    let mut batch_dims = Vec::with_capacity(axes.len());
    for &(in_axes, _) in axes {
        let mut level: Option<String> = None;
        for shape in &mut shapes {
            let (peeled, dim) = peel_batch_dim(shape, in_axes).ok()?;
            if level.as_ref().is_some_and(|l| !dims_canonically_equal(l, &dim)) {
                return None;
            }
            level = Some(dim);
            *shape = peeled;
        }
        batch_dims.push(level?);
    }

    let byte = args_node.start_byte();
    let peeled_args: Vec<CallArgument> = shapes
        .into_iter()
        .zip(args)
        .map(|(shape, arg)| {
            let (CallArgument::Positional { value } | CallArgument::Keyword { value, .. }) = arg;
            let text = ctx.synthetic_text.get(value).unwrap_or(value).clone();
            CallArgument::Positional {
                value: ctx.bind_synthetic(shape, scope_idx, byte, &text),
            }
        })
        .collect();
    let mut output = apply_submodule_call(module, display, &peeled_args, byte, scope_idx, args_node.range(), ctx)?;
    for (&(_, out_axes), dim) in axes.iter().zip(batch_dims).rev() {
        output = prepend_batch_dim(output, out_axes, dim);
    }
    Some(output)
}

/// `Block(features=64)(x)` for a user module of this file, the flax idiom of
/// constructing and applying a submodule in one expression.
fn shape_of_inline_module(
    ctor_call: Node,
    outer_args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let imports = ctx.imports.at(ctor_call.start_byte());
    let module = match constructed_module(ctor_call, ctx.text, imports, ctx.hierarchy) {
        Some(module) => Module::Local(module),
        None => {
            // The arguments are evaluated once the callee is a class.
            let module = imported_module_of(ctor_call, ctx.text, imports)?;
            imported_class(&module, ctx)?;
            Module::Imported(module)
        }
    };
    // The callee names a class only if no scope around the call binds the
    // first name of it: a parameter named like the imported module is not it.
    let callee = ctor_call.child_by_field_name("function")?.utf8_text(ctx.text.as_bytes()).ok()?;
    if shadowed_locally(ctor_call, callee.split('.').next()?, ctx) {
        return None;
    }
    let raw_args = extract_call_arguments(outer_args_node, ctx.text).ok()?;
    let args = resolve_call_args(raw_args, outer_args_node, scope_idx, ctx)?;
    let display = ctor_call.utf8_text(ctx.text.as_bytes()).ok()?;
    apply_module(
        &module,
        display,
        &args,
        outer_args_node.start_byte(),
        scope_idx,
        outer_args_node.range(),
        ctx,
    )
}

/// Unwind a (possibly nested) `vmap(vmap(...(callable)))` call chain.
///
/// `node` must be a `call` node whose function resolves to `vmap`
/// (`KnownFunction::Vmap`); its first positional argument is either the
/// wrapped callable (base case) or another `vmap(...)` call (recursive
/// case, e.g. `jax.vmap(jax.vmap(self.layer))`).
///
/// Returns the innermost callable node plus the `(in_axes, out_axes)` pairs
/// for each vmap layer, ordered outermost-to-innermost, or `None` if `node`
/// isn't a vmap call, or an inner positional-arg call isn't a recognized
/// vmap either (in which case the whole chain is left unmodeled).
fn collect_vmap_axes<'t>(
    node: Node<'t>,
    ctx: &ShapeCtx,
) -> Option<(Node<'t>, Vec<(isize, isize)>)> {
    let func_node = node.child_by_field_name("function")?;
    let target = func_node.utf8_text(ctx.text.as_bytes()).ok()?;
    let resolved = resolve_call_target(target, ctx.imports.at(func_node.start_byte()));
    if !matches!(classify_known_function(&resolved), Some(KnownFunction::Vmap)) {
        return None;
    }

    let args_node = node.child_by_field_name("arguments")?;
    let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
    let in_axes = parse_int_keyword(&raw_args, "in_axes", 0)?;
    let out_axes = parse_int_keyword(&raw_args, "out_axes", 0)?;
    let callable_node = find_positional_arg_node(args_node, 0)?;

    if callable_node.kind() == "call" {
        let (base, mut rest) = collect_vmap_axes(callable_node, ctx)?;
        rest.insert(0, (in_axes, out_axes));
        return Some((base, rest));
    }

    Some((callable_node, vec![(in_axes, out_axes)]))
}

/// Resolve `Layer(...)(x)` — a catalogued layer constructed and applied in
/// one expression, e.g. flax's `nn.Dense(features=64)(x)` or
/// `eqx.nn.Linear(3, 5)(x)`.
/// The inputs of a layer call after the first. Only a loss reads one: its
/// target, which is the second positional argument or `target=`.
fn extra_layer_inputs(args: &[CallArgument], kind: &LayerKind) -> Vec<String> {
    if !matches!(kind, LayerKind::Loss { .. }) {
        return Vec::new();
    }
    args.iter()
        .skip(1)
        .find_map(|arg| match arg {
            CallArgument::Positional { value } => Some(value.clone()),
            CallArgument::Keyword { name, value } if name == "target" => Some(value.clone()),
            CallArgument::Keyword { .. } => None,
        })
        .into_iter()
        .collect()
}

fn shape_of_inline_layer(
    ctor_call: Node,
    outer_args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let kind = classify_inline_constructor(ctor_call, ctx.text, ctx.imports.at(ctor_call.start_byte()))?;
    let kind = ctx.layer_at(&kind, ctor_call.start_byte());
    let raw_args = extract_call_arguments(outer_args_node, ctx.text).ok()?;
    let args = resolve_call_args(raw_args, outer_args_node, scope_idx, ctx)?;
    let CallArgument::Positional { value: input } = args.first()?.clone() else {
        return None;
    };
    let layer = ctor_call
        .utf8_text(ctx.text.as_bytes())
        .unwrap_or("")
        .to_string();
    let application = LayerApplication {
        variable: String::new(),
        layer,
        input,
        extra_inputs: extra_layer_inputs(&args, &kind),
        kind,
        range: outer_args_node.range(),
    };
    match apply_layer_application(&application, &ctx.scope_shapes(scope_idx)) {
        Ok(Some(output)) => Some(output),
        Ok(None) => None,
        Err(message) => {
            ctx.errors.push(ShapeError::mismatch(String::new(), message, outer_args_node.range()));
            None
        }
    }
}

/// Apply a layer batched over its leading axis: peel the batch dim at
/// `in_axes`, run the layer's normal shape rule on the per-element shape,
/// then re-insert the batch dim at `out_axes`.
fn apply_inline_vmap_layer(
    layer: (&str, &LayerKind),
    outer_args: &[CallArgument],
    in_axes: isize,
    out_axes: isize,
    scope_idx: usize,
    range: tree_sitter::Range,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    apply_inline_vmap_layer_chain(layer, outer_args, &[(in_axes, out_axes)], scope_idx, range, ctx)
}

/// Generalized form of `apply_inline_vmap_layer` for nested vmaps, e.g.
/// `jax.vmap(jax.vmap(self.layer))(x)`. `axes` holds `(in_axes, out_axes)`
/// pairs ordered outermost-to-innermost. Peels one leading batch dim per
/// level from the first positional arg's shape, runs the layer's normal
/// shape rule on the fully-peeled per-example shape, then re-prepends the
/// batch dims outward-to-inward in reverse.
fn apply_inline_vmap_layer_chain(
    layer: (&str, &LayerKind),
    outer_args: &[CallArgument],
    axes: &[(isize, isize)],
    scope_idx: usize,
    range: tree_sitter::Range,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let (layer_name, layer) = layer;
    let input_name = outer_args.iter().find_map(|a| match a {
        CallArgument::Positional { value } => Some(value.clone()),
        _ => None,
    })?;
    let mut shape = ctx.resolve_shape(&input_name, scope_idx)?;

    let mut batch_dims: Vec<String> = Vec::with_capacity(axes.len());
    for &(in_axes, _) in axes {
        let (peeled, dim) = peel_batch_dim(&shape, in_axes).ok()?;
        batch_dims.push(dim);
        shape = peeled;
    }

    // MultiheadAttention's real return is an `(output, weights)` tuple;
    // `apply_layer_application` can't express that and always returns
    // `Ok(None)` for it. A single-assignment vmap'd call site
    // (`out = jax.vmap(self.attn)(q, k, v)`) still wants *something* — the
    // query's per-example shape is unchanged by attention (same rule as the
    // direct-call MHA special case in `shape_of_call`), so just re-prepend
    // the peeled batch dims onto it.
    if matches!(layer, LayerKind::MultiheadAttention { .. }) {
        let mut output = shape;
        for (&(_, out_axes), dim) in axes.iter().zip(batch_dims).rev() {
            output = prepend_batch_dim(output, out_axes, dim);
        }
        return Some(output);
    }

    let mut output = match apply_layer_kind(layer, &shape, layer_name, &input_name) {
        Ok(Some(output)) => output,
        Ok(None) => return None,
        Err(message) => {
            ctx.errors.push(ShapeError::mismatch(String::new(), message, range));
            return None;
        }
    };
    for (&(_, out_axes), dim) in axes.iter().zip(batch_dims).rev() {
        output = prepend_batch_dim(output, out_axes, dim);
    }
    Some(output)
}

/// The function that the bare name `target` means at byte `call`: the closure
/// of the nearest enclosing function that defines one, else the function of
/// the module. A method is not visible through a bare name, and a closure is
/// visible only inside its enclosing function.
///
/// `None` also when the name is defined twice in that place
/// (`ScopeLinks::redefined`), when a class body holds the call and defines the
/// name, and for a recursive call.
fn find_callee_scope(
    target: &str,
    call: usize,
    links: &ScopeLinks,
    scopes: &[FunctionShapeScope],
) -> Option<usize> {
    let call_scope = links.scope_index.lookup(call);
    let in_body_of = |class: (usize, usize)| {
        class.0 <= call
            && call < class.1
            && call_scope.is_none_or(|s| s == 0 || class.0 > scopes[s].start_byte)
    };
    if links.methods.get(target).is_some_and(|m| m.iter().any(|&(class, _)| in_body_of(class))) {
        return None;
    }
    let mut best: Option<(usize, usize)> = None; // (enclosing levels to it, scope index)
    for (i, scope) in scopes.iter().enumerate().skip(1) {
        if scope.function_name.as_deref() != Some(target) || links.class_of[i].is_some() {
            continue;
        }
        let levels = match links.parent[i] {
            None => usize::MAX,
            Some(parent) => {
                let mut chain = std::iter::successors(call_scope, |&up| links.parent[up]);
                match chain.position(|up| up == parent) {
                    Some(levels) => levels,
                    None => continue,
                }
            }
        };
        if best.is_none_or(|(nearest, _)| levels < nearest) {
            best = Some((levels, i));
        }
    }
    let (_, i) = best?;
    let recursive = scopes[i].start_byte <= call && call < scopes[i].end_byte;
    (!links.redefined[i] && !recursive).then_some(i)
}

/// `super()`, or `super(<class>, <receiver>)`.
fn is_super_call(node: Node, text: &str) -> bool {
    node.kind() == "call"
        && node
            .child_by_field_name("function")
            .is_some_and(|f| &text[f.byte_range()] == "super")
}

/// How a `def` in a class body binds its first parameter.
#[derive(Clone, Copy, PartialEq)]
enum MethodKind {
    Instance,
    Class,
    Static,
    /// A descriptor that a call does not run like a function.
    Property,
}

fn method_kind(function: Node, text: &str) -> MethodKind {
    let Some(definition) = function.parent().filter(|p| p.kind() == "decorated_definition") else {
        return MethodKind::Instance;
    };
    let mut kind = MethodKind::Instance;
    let mut cursor = definition.walk();
    for decorator in definition.named_children(&mut cursor).filter(|n| n.kind() == "decorator") {
        let name = text[decorator.byte_range()].trim_start_matches('@').trim();
        match name.rsplit('.').next().unwrap_or(name) {
            "staticmethod" => kind = MethodKind::Static,
            "classmethod" => kind = MethodKind::Class,
            "property" | "cached_property" | "setter" | "getter" | "deleter" => {
                return MethodKind::Property;
            }
            _ => {}
        }
    }
    kind
}

/// One call of a function of this file.
struct UserCall<'c, 't> {
    /// The name of the function, for the argument errors.
    name: &'c str,
    /// The callee as the call writes it, for the error.
    display: String,
    args: &'c [CallArgument],
    args_node: Node<'t>,
    call_byte: usize,
}

/// Evaluate a call of the function or method `callee_idx`. `Some` when the
/// call has an answer or an error, `None` when the caller can try other rules.
///
/// Gated on the callee being FULLY annotated (checked against the
/// call-site-independent `original_shapes` snapshot, never the live, mutable
/// `scopes[...].shapes`): the bare-return trace reads `callee.shapes.get(name)`
/// directly, which is only sound when every one of the callee's params was
/// statically annotated (so its body shapes are the same regardless of which
/// call site asks). Once lazy call-site parameter seeding has ever specialized
/// a partially-annotated callee, its `shapes` map holds a PARTICULAR call
/// site's seeded values, and reusing them for a call site with different
/// argument shapes would silently propagate the wrong shape.
/// Partially-annotated callees always go through `apply_seeded_user_function`
/// instead, which re-seeds fresh from `original_shapes` on every call.
fn apply_user_function(
    callee_idx: usize,
    call: UserCall,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Option<Vec<String>>> {
    let UserCall { name, display, args, args_node, call_byte } = call;
    if !callee_all_params_annotated(callee_idx, ctx) {
        return apply_seeded_user_function(callee_idx, call_byte, args, scope_idx, ctx).map(Some);
    }
    // A closure that reads an enclosing function's variable has body shapes
    // that depend on the call, like a partially annotated callee: only its
    // `-> ReturnType` annotation is call-independent.
    let captures = ctx.links.captures[callee_idx];
    if let Some(result) = bind_user_function_args(
        &ctx.scopes[callee_idx],
        name,
        args,
        &ctx.scope_shapes(scope_idx),
        ctx.scope_function_nodes[callee_idx].map(|node| (node, ctx.text)),
    ) {
        match result {
            Ok(Some(shape)) if !captures || ctx.scopes[callee_idx].return_shape.is_some() => {
                return Some(Some(shape));
            }
            Ok(Some(_)) | Ok(None) => {}
            Err(message) => {
                ctx.errors.push(ShapeError::mismatch(display, message, args_node.range()));
                return Some(None);
            }
        }
    }
    if captures {
        return apply_seeded_user_function(callee_idx, call_byte, args, scope_idx, ctx).map(Some);
    }
    None
}

/// Bind a callee's parameter dims to provided arg shapes and substitute
/// into the callee's return shape.
///
/// `positional_arg_shapes` is `[(param_name, shape)]` — callers must
/// map argument variable names to the callee's declared parameter names
/// before calling this function. Both the user-function path and the
/// vmap path do this mapping upstream.
///
/// Returns:
/// - `Ok(Some(substituted_return_shape))` if binding and substitution
///   succeeded and the callee has a return annotation.
/// - `Ok(None)` if the callee has no return annotation (or no shapes at all).
/// - `Err(msg)` if there's a rank or dim mismatch.
fn bind_and_substitute(
    callee: &FunctionShapeScope,
    target_name: &str,
    positional_arg_shapes: &[(&str, Vec<String>)],
) -> Result<Option<Vec<String>>, String> {
    // If the callee has no jaxtyping annotations at all, skip.
    if callee.shapes.is_empty() && callee.return_shape.is_none() {
        return Ok(None);
    }

    let binding = build_dim_binding(callee, target_name, positional_arg_shapes)?;

    // Substitute into return_shape.
    let Some(ref return_shape) = callee.return_shape else {
        return Ok(None);
    };

    Ok(Some(substitute_dims(return_shape, &binding)))
}

/// Build a binding of declared param-dim-name → resolved arg dim from the
/// caller's arg shapes, validating rank/dim compatibility along the way.
/// Extracted from `bind_and_substitute` so the traced-return fallback (see
/// `trace_user_function_return`) can reuse the same binding without also
/// requiring a `-> ReturnType` annotation.
fn build_dim_binding(
    callee: &FunctionShapeScope,
    target_name: &str,
    positional_arg_shapes: &[(&str, Vec<String>)],
) -> Result<HashMap<String, String>, String> {
    let mut binding: HashMap<String, String> = HashMap::new();
    for (param_name, arg_shape) in positional_arg_shapes {
        let Some(param_shape) = callee.shapes.get(*param_name) else {
            continue;
        };

        if param_shape.len() != arg_shape.len() {
            return Err(format!(
                "call to {}: argument '{}' expected rank {}, got rank {}",
                target_name,
                param_name,
                param_shape.len(),
                arg_shape.len()
            ));
        }

        for (dim_idx, (param_dim, arg_dim)) in param_shape.iter().zip(arg_shape.iter()).enumerate()
        {
            let is_param_concrete = param_dim.parse::<usize>().is_ok();

            if is_param_concrete {
                if param_dim != arg_dim {
                    return Err(format!(
                        "call to {}: argument '{}' dim {} expected {}, got {}",
                        target_name, param_name, dim_idx, param_dim, arg_dim
                    ));
                }
            } else if let Some(existing) = binding.get(param_dim) {
                if !dims_canonically_equal(existing, arg_dim) {
                    return Err(format!(
                        "call to {}: dim '{}' cannot be both '{}' and '{}'",
                        target_name, param_dim, existing, arg_dim
                    ));
                }
            } else {
                binding.insert(param_dim.clone(), arg_dim.clone());
            }
        }
    }
    Ok(binding)
}

fn substitute_dims(shape: &[String], binding: &HashMap<String, String>) -> Vec<String> {
    shape
        .iter()
        .map(|dim| binding.get(dim).cloned().unwrap_or_else(|| dim.clone()))
        .collect()
}

/// Fallback for a same-file `self.<method>(...)` call whose callee has no
/// `-> ReturnType` annotation (the common `forward(self, x: Float[Array,
/// "..."])` style with only param annotations): trace the callee's body for
/// a single, non-nested `return <name>` statement and use `<name>`'s shape
/// as already computed in the callee's *own* function scope.
///
/// This works without any cross-function seeding because methods are
/// analyzed in one global source-order pass — a callee defined earlier in
/// the same class (like `forward` before a later `loss` that calls it) has
/// already had its own body's assignments shaped using its own param
/// annotations by the time any call site is reached. Only a bare-identifier
/// return is traced; anything else (an inline expression, a tuple, no
/// `return`, or more than one distinct `return` target) stays `None` —
/// matching the project's "approximate, not exhaustive" convention rather
/// than building a general expression re-evaluator across scopes.
fn trace_user_function_return(
    callee: &FunctionShapeScope,
    target_name: &str,
    positional_arg_shapes: &[(&str, Vec<String>)],
    (func_node, text): (Node, &str),
) -> Result<Option<Vec<String>>, String> {
    let binding = build_dim_binding(callee, target_name, positional_arg_shapes)?;
    let Some(body) = func_node.child_by_field_name("body") else {
        return Ok(None);
    };
    let mut return_name = None;
    if collect_bare_return_identifiers(body, text, &mut return_name).is_none() {
        return Ok(None);
    }
    let Some(return_name) = return_name else {
        return Ok(None);
    };
    let Some(return_shape) = callee.shapes.get(&return_name) else {
        return Ok(None);
    };
    Ok(Some(substitute_dims(return_shape, &binding)))
}

/// Walk a function body collecting `return <identifier>` targets. Returns
/// `None` (via the `?` at call sites) as soon as a non-identifier or a
/// second, differently-named return target is seen, signalling "not
/// traceable" to the caller.
fn collect_bare_return_identifiers(
    block: Node,
    text: &str,
    found: &mut Option<String>,
) -> Option<()> {
    for i in 0..block.named_child_count() {
        let child = block.named_child(i as u32)?;
        match child.kind() {
            "return_statement" => {
                let value = child.named_child(0)?;
                if value.kind() != "identifier" {
                    return None;
                }
                let name = value.utf8_text(text.as_bytes()).ok()?.to_string();
                match found {
                    Some(existing) if *existing != name => return None,
                    _ => *found = Some(name),
                }
            }
            "function_definition" | "class_definition" | "lambda" => continue,
            _ => collect_bare_return_identifiers(child, text, found)?,
        }
    }
    Some(())
}

// ── Lazy call-site parameter seeding ────────────────────────────────────
//
// The mechanism above (`bind_user_function_args` / `bind_and_substitute` /
// `trace_user_function_return`) requires the callee's OWN annotations (or,
// for the bare-identifier-return fallback, shapes the callee's body already
// computed from its own annotations in the one global whole-file pass) —
// it can never shape a callee whose parameters aren't annotated, even when
// every call site passes fully-shaped arguments (`llm.txt`'s "Known
// architectural limit"). The functions below are the extension: resolve a
// call to a same-file function/closure/method scope, seed its UN-annotated
// params from this call site's argument shapes, and evaluate its body on
// demand ("lazily", i.e. only when a call site actually needs it).

/// Recursion/cycle guard depth cap for `specialize_callee_call` — deep but
/// non-cyclic call chains bail rather than recursing unboundedly.
const MAX_SPECIALIZATION_DEPTH: usize = 8;

/// Outcome of lazily evaluating a same-file callee's body with seeded
/// parameter shapes.
#[derive(Debug, Default)]
struct SpecializedReturn {
    /// Shape of the return expression when it is NOT a bare tuple literal —
    /// covers a bare-identifier return, a general `return <expr>`, and the
    /// "every `return` statement names the same identifier" case.
    single: Option<Vec<String>>,
    /// Per-element shapes when the (single) return statement's expression
    /// is a bare tuple literal `return (a, b, ...)` — needed by `lax.scan`
    /// body seeding, which needs the carry and the per-step output
    /// individually rather than collapsed into one shape.
    tuple: Option<Vec<Option<Vec<String>>>>,
}

/// This function's own `return` statement expression node(s), not
/// descending into nested function/class/lambda bodies (a `return` there
/// belongs to a different scope). Mirrors `collect_bare_return_identifiers`
/// but collects the expression node itself rather than requiring/parsing a
/// bare identifier, so the caller can evaluate arbitrary expressions.
fn collect_own_return_exprs<'t>(block: Node<'t>, out: &mut Vec<Node<'t>>) {
    for i in 0..block.named_child_count() {
        let Some(child) = block.named_child(i as u32) else {
            continue;
        };
        match child.kind() {
            "return_statement" => {
                if let Some(value) = child.named_child(0) {
                    out.push(value);
                }
            }
            "function_definition" | "class_definition" | "lambda" => continue,
            _ => collect_own_return_exprs(child, out),
        }
    }
}

/// Evaluate the callee's own return statement(s) against its (specialized)
/// scope. A single return statement is evaluated via the full recursive
/// `shape_of_expression` machinery — a bare tuple literal's elements are
/// kept distinct (`tuple`), anything else (bare identifier or general
/// expression) is one shape (`single`). Multiple return statements are only
/// resolved when every one is the same bare identifier (mirrors the older,
/// more conservative `collect_bare_return_identifiers` convention) — genuinely
/// divergent return expressions across branches aren't unifiable without
/// deeper control-flow analysis, so they stay `None`.
fn trace_specialized_return(body: Node, callee_idx: usize, ctx: &mut ShapeCtx) -> SpecializedReturn {
    let mut returns = Vec::new();
    collect_own_return_exprs(body, &mut returns);

    match returns.len() {
        0 => SpecializedReturn::default(),
        1 => {
            let node = returns[0];
            // `return a, b` (no enclosing parens) parses as an
            // `expression_list`, not a `tuple` node — structurally
            // identical (same named-child layout), just without parens.
            if node.kind() == "tuple" || node.kind() == "expression_list" {
                let mut elems = Vec::with_capacity(node.named_child_count());
                for i in 0..node.named_child_count() {
                    let Some(child) = node.named_child(i as u32) else {
                        continue;
                    };
                    elems.push(shape_of_expression(child, ctx));
                }
                SpecializedReturn {
                    single: None,
                    tuple: Some(elems),
                }
            } else {
                SpecializedReturn {
                    single: shape_of_expression(node, ctx),
                    tuple: None,
                }
            }
        }
        _ => {
            let mut name: Option<&str> = None;
            for n in &returns {
                if n.kind() != "identifier" {
                    return SpecializedReturn::default();
                }
                let Ok(text) = n.utf8_text(ctx.text.as_bytes()) else {
                    return SpecializedReturn::default();
                };
                match name {
                    None => name = Some(text),
                    Some(existing) if existing != text => return SpecializedReturn::default(),
                    _ => {}
                }
            }
            let single = name.and_then(|n| ctx.scopes[callee_idx].shapes.get(n).cloned());
            SpecializedReturn { single, tuple: None }
        }
    }
}

/// Lazily evaluate a same-file callee scope's body with `seeded_params`
/// merged on top of its own jaxtyping annotations, in an ephemeral
/// specialized copy of its scope — so one call site's argument shapes never
/// corrupt another's.
///
/// SPECIALIZATION vs GLOBAL STATE: for editor UX (hover/inlay inside the
/// callee body), the FIRST successful specialization of a given scope is
/// written back to the real scope (+ `assignment_shapes`/`errors`) —
/// first-call-wins, tracked by `ctx.specialized_scopes`. Every later
/// specialization of the same scope (a different call site, possibly with
/// different argument shapes) reverts its own mutations at the end, so it
/// can't overwrite the first call's hover info or leak errors that only
/// apply to its own seeding.
///
/// `replace_annotations`: seeded shapes replace the annotations of the
/// parameters they name, so the body is evaluated in the caller's dims. Plain
/// seeding only fills the parameters without an annotation.
///
/// Recursion/cycles: re-entering a scope already being specialized (a
/// direct or mutual recursive call) returns `None` immediately, and the
/// active-specialization stack is capped at `MAX_SPECIALIZATION_DEPTH`.
fn specialize_callee_call(
    callee_idx: usize,
    seeded_params: HashMap<String, Vec<String>>,
    carry_params: Vec<String>,
    call_byte: usize,
    replace_annotations: bool,
    ctx: &mut ShapeCtx,
) -> Option<SpecializedReturn> {
    if ctx.active_specializations.contains(&callee_idx)
        || ctx.active_specializations.len() >= MAX_SPECIALIZATION_DEPTH
    {
        return None;
    }
    let func_node = (*ctx.scope_function_nodes.get(callee_idx)?)?;

    // Always start from the call-site-independent baseline (`original_
    // shapes`), NEVER from the live `scopes[callee_idx].shapes` — the live
    // map may already hold a prior call site's seeded params/locals
    // (first-call-wins write-back), and cloning that would leak this
    // call's un-annotated params from a DIFFERENT call site's shapes.
    let mut specialized_shapes = ctx
        .original_shapes
        .get(callee_idx)
        .cloned()
        .unwrap_or_default();
    for (name, shape) in seeded_params {
        if replace_annotations {
            specialized_shapes.insert(name, shape);
        } else {
            specialized_shapes.entry(name).or_insert(shape);
        }
    }

    // A fully annotated callee already has its own hover shapes from the
    // whole-file pass; a call that replaces annotations must not overwrite them.
    let is_first = !ctx.specialized_scopes.contains(&callee_idx)
        && !(replace_annotations && callee_all_params_annotated(callee_idx, ctx));
    let saved_shapes = std::mem::replace(&mut ctx.scopes[callee_idx].shapes, specialized_shapes);
    let saved_carries: Vec<_> = ctx
        .tuple_carries
        .extract_if(|(scope, _)| *scope == callee_idx)
        .collect();
    ctx.tuple_carries
        .extend(carry_params.into_iter().map(|param| (callee_idx, param)));
    let errors_before = ctx.errors.len();
    let assignments_before = ctx.assignment_shapes.len();

    ctx.active_specializations.push(callee_idx);
    let reads_enclosing_live = ctx.links.parent[callee_idx].is_some_and(|parent| {
        ctx.links.scope_index.lookup(call_byte)
            .is_some_and(|call_scope| ctx.call_sees_live(call_scope, parent))
    });
    if reads_enclosing_live {
        ctx.call_time.push(callee_idx);
    }

    // Re-collect just this callee's own assignments (cheap — bounded by its
    // own body size, not the whole file) and replay them through the same
    // per-assignment logic the top-level pass uses. Items whose innermost
    // scope isn't exactly `callee_idx` belong to a nested closure defined
    // inside this body — that closure gets its own scope and is evaluated
    // lazily, on its own, if/when it is itself called.
    let body_assignments = collect_assignment_items(func_node, ctx.text).unwrap_or_default();
    process_assignment_items(func_node, body_assignments, Some(callee_idx), ctx);

    let ret = match func_node.child_by_field_name("body") {
        Some(body) => trace_specialized_return(body, callee_idx, ctx),
        None => SpecializedReturn::default(),
    };

    ctx.active_specializations.pop();
    if reads_enclosing_live {
        ctx.call_time.pop();
    }

    if is_first {
        ctx.specialized_scopes.insert(callee_idx);
        // The whole-file pass may already have reported an error inside this
        // body; replaying it must not report it a second time.
        let mut kept = errors_before;
        for i in errors_before..ctx.errors.len() {
            if !ctx.errors[..errors_before].contains(&ctx.errors[i]) {
                ctx.errors.swap(kept, i);
                kept += 1;
            }
        }
        ctx.errors.truncate(kept);
    } else {
        ctx.scopes[callee_idx].shapes = saved_shapes;
        ctx.tuple_carries.retain(|(scope, _)| *scope != callee_idx);
        ctx.tuple_carries.extend(saved_carries);
        ctx.errors.truncate(errors_before);
        ctx.assignment_shapes.truncate(assignments_before);
    }

    Some(ret)
}

/// Whether every one of a callee's params (`all_params`, skipping a leading
/// `self`/`cls`) was statically annotated at extraction time — checked
/// against the immutable `original_shapes` snapshot, NOT the live (mutable)
/// `scopes[...].shapes`, so a prior specialization's write-back for a
/// DIFFERENT call site never makes this incorrectly report "fully
/// annotated" (see `original_shapes`'s doc comment).
fn callee_all_params_annotated(callee_idx: usize, ctx: &ShapeCtx) -> bool {
    let params = without_receiver(&ctx.scopes[callee_idx].all_params);
    let Some(original) = ctx.original_shapes.get(callee_idx) else {
        return false;
    };
    params.iter().all(|p| original.contains_key(p))
}

/// Map a call's resolved arguments onto the callee's FULL declared
/// parameter list (`all_params` — annotated and un-annotated, in
/// declaration order), skipping a leading `self`/`cls` (same convention as
/// `resolution::bind_call_arguments`). Returns `(param_name, shape)` for
/// every param whose incoming argument shape is resolvable in the caller's
/// scope; an unresolvable argument just means that param goes unseeded (not
/// a hard bail) — any body expression that actually depends on it stays
/// dark, same as any other unresolvable lookup elsewhere in this evaluator.
fn seed_params_from_call(
    all_params: &[String],
    args: &[CallArgument],
    caller_scope_idx: usize,
    ctx: &ShapeCtx,
) -> (HashMap<String, Vec<String>>, Vec<String>) {
    let params = without_receiver(all_params);
    let mut seeded = HashMap::new();
    let mut carries = Vec::new();
    let is_carry = |value: &str| {
        ctx.tuple_carries
            .contains(&(caller_scope_idx, value.to_string()))
    };
    let mut positional_idx = 0usize;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if let Some(param) = params.get(positional_idx)
                    && let Some(shape) = ctx.resolve_shape(value, caller_scope_idx)
                {
                    seeded.insert(param.clone(), shape);
                    if is_carry(value) {
                        carries.push(param.clone());
                    }
                }
                positional_idx += 1;
            }
            CallArgument::Keyword { name, value } => {
                if params.iter().any(|p| p == name)
                    && let Some(shape) = ctx.resolve_shape(value, caller_scope_idx)
                {
                    seeded.insert(name.clone(), shape);
                    if is_carry(value) {
                        carries.push(name.clone());
                    }
                }
            }
        }
    }
    (seeded, carries)
}

/// Fallback for a same-file call whose callee (a top-level function, a
/// nested closure, or a `self.<method>`) has at least one un-annotated
/// parameter: seed the un-annotated params from this call site's resolved
/// argument shapes and evaluate the callee's body on demand. Fully annotated
/// callees use `bind_user_function_args` instead; this returns `None` for
/// them rather than redoing the annotation/bare-return-trace work.
fn apply_seeded_user_function(
    callee_idx: usize,
    call_byte: usize,
    args: &[CallArgument],
    caller_scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let all_params = ctx.scopes[callee_idx].all_params.clone();
    let (seeded, carries) = seed_params_from_call(&all_params, args, caller_scope_idx, ctx);
    let result = specialize_callee_call(callee_idx, seeded, carries, call_byte, false, ctx)?;
    specialized_single(result)
}

/// The one array shape a specialized return stands for.
fn specialized_single(result: SpecializedReturn) -> Option<Vec<String>> {
    match result.tuple {
        Some(elems) => {
            // A bare-tuple return in a single-assignment context is only
            // representable as one array shape when every element shares
            // the same shape (mirrors `shape_of_tuple`'s homogeneous-carry
            // convention already used for scan/RNN state tuples).
            let mut shape: Option<Vec<String>> = None;
            for elem in elems {
                let elem = elem?;
                match &shape {
                    None => shape = Some(elem),
                    Some(s) if *s == elem => {}
                    Some(_) => return None,
                }
            }
            shape
        }
        None => result.single,
    }
}

/// A method's parameters without the leading `self`/`cls`.
fn without_receiver(all_params: &[String]) -> &[String] {
    match all_params.first().map(String::as_str) {
        Some("self") | Some("cls") => &all_params[1..],
        _ => all_params,
    }
}

/// Call a user module of this file: run its `__call__` (or `forward`) on the
/// argument shapes. Annotations on the callee's parameters are checked against
/// the arguments, then the body is evaluated in the caller's dims.
///
/// A dim in the result that names a constructor parameter or attribute of the
/// callee (`dim`, `self.features`, see `callee_names`) is replaced by the value
/// the constructor call gave it. The result stays unknown when no such value
/// exists, or when the same name also appears in an argument shape (two
/// meanings of one name).
fn apply_submodule_call(
    module: &Submodule,
    display: &str,
    args: &[CallArgument],
    call_byte: usize,
    caller_scope: usize,
    range: tree_sitter::Range,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    if args.iter().any(|arg| matches!(arg, CallArgument::Positional { value } if value.starts_with('*'))) {
        return None;
    }
    let callee_idx = ctx.module_callee(module)?;
    let (seeded, carries) =
        seed_params_from_call(&ctx.scopes[callee_idx].all_params, args, caller_scope, ctx);

    // An annotated parameter that this call does not bind would leave the
    // callee's own dim names in the body.
    let declared = &ctx.original_shapes[callee_idx];
    let mut bound: Vec<(&str, Vec<String>)> = Vec::new();
    for param in without_receiver(&ctx.scopes[callee_idx].all_params) {
        let Some(annotation) = declared.get(param) else {
            continue;
        };
        match seeded.get(param) {
            Some(shape) => bound.push((param, shape.clone())),
            None if annotation.iter().all(|dim| dim.parse::<usize>().is_ok()) => {}
            None => return None,
        }
    }
    let binding = match bind_module_dims(declared, display, &bound) {
        Ok(binding) => binding,
        Err(message) => {
            ctx.errors.push(ShapeError::mismatch(display.to_string(), message, range));
            return None;
        }
    };

    let caller_names: HashSet<String> = seeded
        .values()
        .flatten()
        .flat_map(|dim| dim.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.')))
        .map(str::to_string)
        .collect();
    let result = match ctx.scopes[callee_idx].return_shape.clone() {
        Some(annotation) => {
            let binding = binding?;
            // A dim named like a constructor argument that holds a different
            // integer: the module rejects this argument at run time.
            let names = ctx.callee_names(module);
            let conflicts = |(name, dim): (&String, &String)| {
                matches!(names(name), Some(Some(given))
                    if given != *dim && given.parse::<usize>().is_ok() && dim.parse::<usize>().is_ok())
            };
            if binding.iter().any(conflicts) {
                return None;
            }
            substitute_dims(&annotation, &binding)
        }
        None => {
            let ret = specialize_callee_call(callee_idx, seeded, carries, call_byte, true, ctx)?;
            specialized_single(ret)?
        }
    };

    let names = ctx.callee_names(module);
    result
        .iter()
        .map(|dim| rebind_callee_dim(dim, &names, &caller_names))
        .collect()
}

/// Bind the dim names of a module's annotated parameters to the argument
/// dims. Only a rank mismatch or two different integers is an error. A named
/// dim compared with an integer or with another name may be equal at run time,
/// so those never report. `Ok(None)` means the annotations cannot be bound
/// exactly: a dim is not an integer or a plain name (`...`, `d*2`), or one name
/// is bound to two different dims. Only a return annotation needs the binding.
fn bind_module_dims(
    param_shapes: &HashMap<String, Vec<String>>,
    target_name: &str,
    args: &[(&str, Vec<String>)],
) -> Result<Option<HashMap<String, String>>, String> {
    let mut binding: HashMap<String, String> = HashMap::new();
    let mut exact = true;
    for (param, arg) in args {
        let Some(declared) = param_shapes.get(*param) else {
            continue;
        };
        if declared.iter().any(|dim| dim == "_" || !dim.chars().all(|c| c.is_alphanumeric() || c == '_')) {
            exact = false;
            continue;
        }
        if declared.len() != arg.len() {
            return Err(format!(
                "call to {target_name}: argument '{param}' expected rank {}, got rank {}",
                declared.len(),
                arg.len()
            ));
        }
        for (i, (want, got)) in declared.iter().zip(arg).enumerate() {
            if want.parse::<usize>().is_ok() {
                if got.parse::<usize>().is_ok() && want != got {
                    return Err(format!(
                        "call to {target_name}: argument '{param}' dim {i} expected {want}, got {got}"
                    ));
                }
            } else if let Some(existing) = binding.get(want) {
                exact &= dims_canonically_equal(existing, got);
            } else {
                binding.insert(want.clone(), got.clone());
            }
        }
    }
    Ok(exact.then_some(binding))
}

/// Replace the names in `dim` that belong to the callee instance (see
/// `apply_submodule_call`). `caller_names` holds the names, and `self.<name>`
/// tokens, found in the argument shapes.
fn rebind_callee_dim(
    dim: &str,
    names: &dyn Fn(&str) -> Option<Option<String>>,
    caller_names: &HashSet<String>,
) -> Option<String> {
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = String::with_capacity(dim.len());
    let mut rest = dim;
    while let Some(c) = rest.chars().next() {
        if !is_word(c) {
            out.push(c);
            rest = &rest[c.len_utf8()..];
            continue;
        }
        let end = rest.find(|c: char| !is_word(c)).unwrap_or(rest.len());
        let word = &rest[..end];
        rest = &rest[end..];
        if word == "self"
            && let Some(after_dot) = rest.strip_prefix('.')
        {
            let end = after_dot.find(|c: char| !is_word(c)).unwrap_or(after_dot.len());
            let attr = &after_dot[..end];
            rest = &after_dot[end..];
            if caller_names.contains(&format!("self.{attr}")) {
                return None;
            }
            out.push_str(&names(attr)??);
        } else if let Some(value) = names(word) {
            let value = value?;
            if value != word && caller_names.contains(word) {
                return None;
            }
            out.push_str(&value);
        } else {
            out.push_str(word);
        }
    }
    Some(out)
}

/// `jax.lax.scan` body seeding (the flagship consumer of lazy call-site
/// parameter seeding): given the scan's `body` callable node, `init`'s
/// already-computed shape, and the `xs` node, seed `body`'s first param
/// with the carry shape and its second with `xs`'s shape minus its leading
/// (scan) axis, evaluate `body`'s return lazily, and return `ys` — the
/// body's second return element (`y`) with `xs`'s leading dim prepended
/// back on. Only a bare same-file function/closure `body` is modelled (a
/// qualified name or `self.<method>` isn't traced here); `None` for
/// anything not statically derivable (missing `xs` shape, body not found,
/// body's return isn't a 2-tuple, …).
fn scan_body_ys_shape(
    body_node: Node,
    init_shape: &[String],
    init_is_tuple: bool,
    xs_node: Node,
    scan_call_byte: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    if body_node.kind() == "lambda" {
        let scope_idx = ctx.links.scope_index.lookup(scan_call_byte)?;
        if !is_single_array_argument(xs_node, scope_idx, ctx) {
            return None;
        }
        let xs_shape = shape_of_expression(xs_node, ctx)?;
        let (leading, rest) = xs_shape.split_first()?;
        let output = scan_lambda_output(
            body_node,
            init_shape.to_vec(),
            init_is_tuple,
            rest.to_vec(),
            scope_idx,
            ctx,
        )?;
        return Some(std::iter::once(leading.clone()).chain(output).collect());
    }
    let body_name = body_node.utf8_text(ctx.text.as_bytes()).ok()?;
    if body_name.contains('.') {
        return None;
    }
    let xs_shape = shape_of_expression(xs_node, ctx)?;
    let (leading, rest) = xs_shape.split_first()?;
    let leading = leading.clone();
    let rest = rest.to_vec();

    let callee_idx = find_callee_scope(body_name, scan_call_byte, ctx.links, ctx.scopes)?;
    let all_params = ctx.scopes[callee_idx].all_params.clone();
    let mut seeded = HashMap::new();
    let mut carries = Vec::new();
    if let Some(carry_param) = all_params.first() {
        seeded.insert(carry_param.clone(), init_shape.to_vec());
        if init_is_tuple {
            carries.push(carry_param.clone());
        }
    }
    if let Some(elem_param) = all_params.get(1) {
        seeded.insert(elem_param.clone(), rest);
    }

    let result = specialize_callee_call(callee_idx, seeded, carries, scan_call_byte, false, ctx)?;
    let mut elems = result.tuple?;
    if elems.len() != 2 {
        return None;
    }
    let y_shape = elems.remove(1)?;
    let mut out = vec![leading];
    out.extend(y_shape);
    Some(out)
}

/// Bind a callee's declared parameter shapes to the caller's argument shapes
/// and substitute into its return shape. Shared by `shape_of_call`
/// (same-file helpers, callee found via `find_callee_scope`) and
/// `apply_imported_user_function` (cross-file helpers, callee found by
/// resolving the import and extracting its jaxtyping annotations on disk).
///
/// Returns `None` if the callee has no jaxtyping annotations at all (let the
/// caller fall through to the known-function branch); `Some(Ok(None))` if
/// annotations exist but an argument's shape couldn't be resolved (v1
/// intentionally bails on the whole call rather than partially validating);
/// `Some(Ok(Some(shape)))` / `Some(Err(msg))` otherwise.
///
/// v1 limitations (documented in PR #34, still true for the cross-file
/// extension):
/// - Only positional arguments are matched. Keyword args that match a param
///   name are honoured; otherwise the call is skipped with Ok(None).
/// - Fresh output dims (not in the binding) pass through unchanged.
fn bind_user_function_args(
    callee: &FunctionShapeScope,
    target: &str,
    args: &[CallArgument],
    caller_shapes: &dyn ShapeLookup,
    // The existing function node and its source enable body tracing for
    // same-file calls. Imported helpers pass None and use annotations only.
    source: Option<(Node, &str)>,
) -> Option<Result<Option<Vec<String>>, String>> {
    // If the callee has no jaxtyping annotations at all, fall through to
    // the known-function branch.
    if callee.shapes.is_empty() && callee.return_shape.is_none() {
        return None;
    }

    // Resolve each positional arg to a shape from the caller's scope.
    // We match positional args to declared params in declaration order.
    let param_names = &callee.param_order;
    let mut arg_shapes: Vec<(&str, Vec<String>)> = Vec::new(); // (param_name, arg_shape)
    let mut positional_idx = 0usize;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                let Some(param_name) = param_names.get(positional_idx) else {
                    break; // more positional args than annotated params — ignore extras
                };
                let Some(shape) = caller_shapes.shape(value.as_str()) else {
                    // Arg has no known shape (literal, untracked variable, etc.)
                    // — skip entire function. v1 intentionally bails early
                    // rather than partially validating some args, to avoid
                    // noisy diagnostics on calls that mix typed and untyped
                    // arguments.
                    return Some(Ok(None));
                };
                arg_shapes.push((param_name, shape.clone()));
                positional_idx += 1;
            }
            CallArgument::Keyword { name, value } => {
                // v1: honour keyword args whose name matches a declared param.
                if callee.shapes.contains_key(name) {
                    let Some(shape) = caller_shapes.shape(value.as_str()) else {
                        return Some(Ok(None));
                    };
                    arg_shapes.push((name, shape.clone()));
                }
                // Non-matching keyword args are silently ignored in v1.
            }
        }
    }

    // Delegate to the shared bind_and_substitute helper — or, when the
    // callee has no `-> ReturnType` annotation and its source node is available
    // (same-file path), the body-tracing fallback (see
    // `trace_user_function_return`).
    if callee.return_shape.is_some() {
        Some(bind_and_substitute(callee, target, &arg_shapes))
    } else if let Some(source) = source {
        Some(trace_user_function_return(callee, target, &arg_shapes, source))
    } else {
        Some(Ok(None))
    }
}

/// Cross-file counterpart of the same-file path in `shape_of_call`: resolve
/// `target` through the import map to a function in another file, extract its
/// jaxtyping parameter + return shape annotations, and apply the same
/// bind-and-substitute logic. `None` means "not an imported function we can
/// resolve, or it has no jaxtyping annotations" — the caller falls through to
/// the known-function branch, same contract as `bind_user_function_args`.
#[allow(clippy::too_many_arguments)]
fn apply_imported_user_function(
    target: &str,
    args: &[CallArgument],
    caller_shapes: &dyn ShapeLookup,
    import_map: &HashMap<String, ImportPath>,
    search_roots: &[PathBuf],
    read_file: &dyn Fn(&PathBuf) -> Option<String>,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
) -> Option<Result<Option<Vec<String>>, String>> {
    let callee = match resolve_imported_function_shape(
        target, import_map, search_roots, read_file, max_depth, cache,
    ) {
        Ok(Some(scope)) => scope,
        Ok(None) => return None,
        Err(message) => return Some(Err(message)),
    };
    bind_user_function_args(&callee, target, args, caller_shapes, None)
}

fn apply_matmul_shape(
    left: &[String],
    right: &[String],
    left_name: &str,
    right_name: &str,
) -> Result<Option<Vec<String>>, String> {
    // Rank < 2: return Ok(None) for v1
    if left.len() < 2 || right.len() < 2 {
        return Ok(None);
    }

    // Batch dims broadcast numpy-style: align right, missing or literal-1
    // dims broadcast, anything else must match exactly (symbolic equality).
    let left_batch = &left[..left.len() - 2];
    let right_batch = &right[..right.len() - 2];
    let batch_rank = left_batch.len().max(right_batch.len());
    let mut batch = Vec::with_capacity(batch_rank);
    for k in (1..=batch_rank).rev() {
        let l = left_batch.len().checked_sub(k).map(|i| &left_batch[i]);
        let r = right_batch.len().checked_sub(k).map(|i| &right_batch[i]);
        match (l, r) {
            (Some(l), Some(r)) if dims_canonically_equal(l, r) => batch.push(l.clone()),
            (Some(l), Some(r)) if l == "1" => batch.push(r.clone()),
            (Some(l), Some(r)) if r == "1" => batch.push(l.clone()),
            (Some(l), Some(r)) => {
                return Err(format!(
                    "matmul batch dimension mismatch: {} has {}, {} has {} (align from the right)",
                    left_name, l, right_name, r
                ));
            }
            (Some(d), None) | (None, Some(d)) => batch.push(d.clone()),
            (None, None) => unreachable!(),
        }
    }

    // Last dim of LHS must equal second-to-last dim of RHS.
    // Invariant: left.len() >= 2 and right.len() >= 2 (guard above).
    let lhs_last = left
        .last()
        .expect("invariant: left.len() >= 2 checked above");
    let rhs_second_last = &right[right.len() - 2];

    if !dims_canonically_equal(lhs_last, rhs_second_last) {
        return Err(format!(
            "matmul dimension mismatch: {} last dim {} != {} second-to-last dim {}",
            left_name, lhs_last, right_name, rhs_second_last
        ));
    }

    let mut output = batch;
    output.push(left[left.len() - 2].clone());
    output.push(
        right
            .last()
            .expect("invariant: right.len() >= 2 checked above")
            .clone(),
    );
    Ok(Some(output))
}

fn apply_elementwise_shape(
    left: &[String],
    right: &[String],
    op: BinaryOp,
) -> Result<Option<Vec<String>>, String> {
    // A resolved rank-0 ("scalar") shape — e.g. a plain `decay: float`
    // function parameter (seeded as `[]` by `extract_jaxtyping_shapes`), or
    // an int unpacked from `x.shape` — broadcasts against anything, same as
    // a literal scalar (numpy semantics: a 0-d array is a scalar).
    if left.is_empty() && right.is_empty() {
        return Ok(Some(Vec::new()));
    }
    if left.is_empty() {
        return Ok(Some(right.to_vec()));
    }
    if right.is_empty() {
        return Ok(Some(left.to_vec()));
    }

    let op_symbol = match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::MatMul => unreachable!(),
    };

    let rank = left.len().max(right.len());
    let mut result = Vec::with_capacity(rank);
    for i in 0..rank {
        // right-align: None = leading pad of the shorter operand
        let a = (i + left.len()).checked_sub(rank).map(|j| &left[j]);
        let b = (i + right.len()).checked_sub(rank).map(|j| &right[j]);
        let dim = match (a, b) {
            (Some(a), None) => a.clone(),
            (None, Some(b)) => b.clone(),
            (Some(a), Some(b)) => {
                if dims_canonically_equal(a, b) || b == "1" {
                    a.clone()
                } else if a == "1" {
                    b.clone()
                } else {
                    // ponytail: unequal non-"1" dims are incompatible, including
                    // distinct symbolic names ("batch" vs "seq"). Strict in v1;
                    // upgrade to a unification step if it proves too strict.
                    return Err(format!(
                        "elementwise {} incompatible shapes, got [{}] and [{}]",
                        op_symbol,
                        left.join(", "),
                        right.join(", ")
                    ));
                }
            }
            (None, None) => unreachable!("i < max(left.len(), right.len())"),
        };
        result.push(dim);
    }

    Ok(Some(result))
}

/// Check if a resolved call target is a shape-preserving function
/// (elementwise math or activation) that isn't in the `KnownFunction` enum.
///
/// These functions always return the input array's shape unchanged,
/// so we can handle them with a simple "return first arg's shape" rule
/// without adding each one to the enum.
fn is_shape_preserving_call(resolved: &ResolvedTarget) -> bool {
    let Some((name, module)) = resolved.parts.split_last() else {
        return false;
    };
    let name: &str = name;

    // jax.nn activations
    if module == ["jax", "nn"] {
        // `one_hot` is intentionally excluded — it appends a `num_classes`
        // dim rather than preserving shape; see `KnownFunction::OneHot`.
        return matches!(name,
            "relu" | "sigmoid" | "softplus" | "silu" | "swish" | "gelu"
            | "elu" | "leaky_relu" | "selu" | "hard_sigmoid" | "hard_silu"
            | "hard_tanh" | "hard_swish" | "mish" | "celu" | "log_sigmoid"
            | "log_softmax" | "softmax" | "standardize"
        );
    }

    // flax.linen activations (same set as jax.nn — flax re-exports them)
    if module == ["flax", "linen"] {
        return matches!(
            name,
            "relu" | "sigmoid" | "softplus" | "silu" | "swish" | "gelu"
                | "elu" | "leaky_relu" | "selu" | "celu" | "log_sigmoid"
                | "log_softmax" | "softmax" | "tanh" | "standardize"
        );
    }

    // jax.numpy / numpy elementwise math
    if module == ["jax", "numpy"] || module == ["numpy"] {
        return matches!(name,
            "exp" | "exp2" | "expm1" | "log" | "log2" | "log10" | "log1p"
            | "sqrt" | "cbrt" | "square" | "abs" | "fabs" | "sign" | "conj"
            | "real" | "imag" | "angle" | "clip" | "ceil" | "floor" | "trunc"
            | "rint" | "round" | "around" | "fix" | "positive" | "negative"
            | "absolute" | "reciprocal"
            | "sin" | "cos" | "tan" | "arcsin" | "arccos" | "arctan"
            | "sinh" | "cosh" | "tanh" | "arcsinh" | "arccosh" | "arctanh"
            | "deg2rad" | "rad2deg" | "degrees" | "radians"
            | "add" | "subtract" | "multiply" | "divide" | "true_divide"
            | "floor_divide" | "power" | "float_power" | "remainder" | "mod"
            | "fmod" | "maximum" | "minimum" | "fmax" | "fmin" | "copysign"
            | "heaviside" | "logical_not" | "logical_and" | "logical_or"
            | "logical_xor" | "equal" | "not_equal" | "less" | "less_equal"
            | "greater" | "greater_equal" | "isnan" | "isfinite" | "isinf"
            | "isneginf" | "isposinf" | "signbit" | "nextafter" | "nan_to_num"
            | "arctan2" | "hypot" | "logaddexp"
        );
    }

    // torch elementwise math + activations
    if module == ["torch"] {
        return matches!(name,
            "exp" | "log" | "sqrt" | "abs" | "sign" | "clip" | "ceil" | "floor"
            | "sin" | "cos" | "tan" | "arcsin" | "arccos" | "arctan"
            | "sinh" | "cosh" | "tanh" | "sigmoid" | "relu" | "gelu"
            | "silu" | "mish" | "softplus" | "softsign" | "prelu"
            | "add" | "sub" | "mul" | "div" | "pow" | "neg"
            | "isnan" | "isfinite" | "isinf"
        ) || is_binary_elementwise_call(resolved);
    }

    false
}

/// The two-operand members of `is_shape_preserving_call`: numpy, jax and torch
/// broadcast both operands, so the result is not the first operand's shape.
/// `torch.max` and `torch.min` have their own rule in `KnownFunction`.
fn is_binary_elementwise_call(resolved: &ResolvedTarget) -> bool {
    let Some((name, module)) = resolved.parts.split_last() else {
        return false;
    };
    if module == ["jax", "numpy"] || module == ["numpy"] {
        return matches!(name.as_str(),
            "add" | "subtract" | "multiply" | "divide" | "true_divide"
            | "floor_divide" | "power" | "float_power" | "remainder" | "mod"
            | "fmod" | "maximum" | "minimum" | "fmax" | "fmin" | "copysign"
            | "heaviside" | "logical_and" | "logical_or" | "logical_xor"
            | "equal" | "not_equal" | "less" | "less_equal" | "greater"
            | "greater_equal" | "nextafter" | "arctan2" | "hypot" | "logaddexp"
        );
    }
    module == ["torch"]
        && matches!(name.as_str(),
            "add" | "sub" | "mul" | "div" | "pow" | "maximum" | "minimum" | "fmax"
            | "fmin" | "fmod" | "remainder" | "atan2" | "floor_divide"
            | "true_divide" | "multiply" | "subtract" | "divide" | "eq" | "ne"
            | "lt" | "le" | "gt" | "ge" | "logical_and" | "logical_or"
            | "logical_xor" | "hypot" | "logaddexp" | "copysign"
        )
}

#[cfg(test)]
mod shape_of_expression_tests {
    use super::*;
    use crate::python_ast::extract_jaxtyping_shapes;
    use tree_sitter::Parser;

    fn parse(code: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();
        parser.parse(code, None).unwrap()
    }

    fn analyze(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        analyze_layer_shapes(tree.root_node(), code, &[], |_| None, 5, None).unwrap()
    }

    fn shape(dims: &[&str]) -> Vec<String> {
        dims.iter().map(|dim| dim.to_string()).collect()
    }

    fn analyze_simple(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        let imports = ImportScopes::new(tree.root_node(), code).unwrap();
        let layer_records = Vec::new();
        let mut scopes = extract_jaxtyping_shapes(tree.root_node(), code).unwrap();

        let mut applications = Vec::new();
        let mut errors = Vec::new();
        let mut assignment_shapes = Vec::new();
        let mut vmap_targets = HashMap::new();
        let self_attr_layers = HashMap::new();
        let aliases = HashMap::new();
        let search_roots: Vec<PathBuf> = Vec::new();
        let read_file: &dyn Fn(&PathBuf) -> Option<String> = &|_: &PathBuf| None;

        let assignments = collect_assignment_items(tree.root_node(), code).unwrap();

        let mut function_nodes = Vec::new();
        let mut class_ranges = Vec::new();
        collect_function_definition_nodes(tree.root_node(), &mut function_nodes, &mut class_ranges);
        let mut scope_function_nodes: Vec<Option<Node>> = std::iter::once(None)
            .chain(function_nodes.into_iter().map(Some))
            .collect();
        scope_function_nodes.resize(scopes.len(), None);
        let original_shapes: Vec<HashMap<String, Vec<String>>> =
            scopes.iter().map(|s| s.shapes.clone()).collect();

        let links = build_scope_links(&scopes, &scope_function_nodes, &class_ranges, code);

        let mut ctx = ShapeCtx {
            text: code,
            imports: &imports,
            layer_records: &layer_records,
            self_attr_layers: &self_attr_layers,
            aliases: &aliases,
            attr_writes: &[],
            layer_dims: Default::default(),
            root: tree.root_node(),
            hierarchy: &ClassHierarchy::default(),
            search_roots: &search_roots,
            read_file,
            max_depth: 5,
            cache: None,
            scopes: &mut scopes,
            vmap_targets: &mut vmap_targets,
            applications: &mut applications,
            errors: &mut errors,
            assignment_shapes: &mut assignment_shapes,
            synthetic_counter: 0,
            synthetics: HashMap::new(),
            synthetic_text: HashMap::new(),
            scalar_dims: HashMap::new(),
            tuple_carries: HashSet::new(),
            scope_function_nodes: &scope_function_nodes,
            active_specializations: Vec::new(),
            specialized_scopes: std::collections::HashSet::new(),
            original_shapes: &original_shapes,
            links: &links,
            submodules: &Submodules::default(),
            global_names: Default::default(),
            classes: ScopeIndex::default(),
            class_starts: Vec::new(),
            module_writes: HashMap::new(),
            class_body_names: Default::default(),
            global_scopes: Default::default(),
            module_values: Default::default(),
            module_index: Default::default(),
            annotation_symbols: Default::default(),
            int_names: HashSet::new(),
            int_consts: Default::default(),
            call_time: Vec::new(),
            inner_masked: Vec::new(),
            inner_scalars: HashSet::new(),
            inner_arrays: HashSet::new(),
        };

        for (lhs, rhs_node, _assignment_node) in assignments {
            let Lhs::Single(lhs_name) = lhs else {
                continue;
            };
            let scope_idx = match ctx.links.scope_index.lookup(rhs_node.start_byte()) {
                Some(idx) => idx,
                None => continue,
            };
            let result = shape_of_expression(rhs_node, &mut ctx);
            if let Some(shape) = result {
                ctx.scopes[scope_idx].shapes.insert(lhs_name, shape);
            }
        }

        let layers = HashMap::new();
        LayerShapeAnalysis {
            scopes,
            layers,
            applications,
            errors,
            assignment_shapes: Vec::new(),
        }
    }

    fn find_shape<'a>(analysis: &'a LayerShapeAnalysis, var: &str) -> Option<&'a Vec<String>> {
        analysis.scopes.iter().find_map(|s| s.shapes.get(var))
    }

    #[test]
    fn test_nested_call_jnp_exp_reshape() {
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "4 8"]):
    y = jnp.exp(jnp.reshape(x, (4, 8)))
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["4", "8"]))
        );
    }

    #[test]
    fn test_unary_minus_preserves_shape() {
        let code = r#"
def f(x: Float[Array, "3 5"]):
    y = -x
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3", "5"]))
        );
    }

    #[test]
    fn test_parenthesized_expression_preserves_shape() {
        let code = r#"
def f(x: Float[Array, "3 5"]):
    y = (x)
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3", "5"]))
        );
    }

    #[test]
    fn test_chained_method_call_reshape_sum() {
        let code = r#"
def f(x: Float[Array, "3 4"]):
    y = x.reshape(3, 4).sum(axis=1)
"#;
        let analysis = analyze_simple(code);
        // reshape(3,4) → [3, 4], sum(axis=1) → [3]
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3"]))
        );
    }

    #[test]
    fn test_existing_single_call_still_works() {
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"]):
    y = jnp.reshape(x, (5, 3))
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["5", "3"]))
        );
    }

    #[test]
    fn test_nested_call_two_levels() {
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "12"]):
    y = jnp.exp(jnp.reshape(x, (3, 4)))
"#;
        let analysis = analyze_simple(code);
        // reshape [12] → [3, 4], exp preserves shape → [3, 4]
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3", "4"]))
        );
    }

    #[test]
    fn test_jax_nn_softplus_shape_preserving() {
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"]):
    y = jax.nn.softplus(x)
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3", "5"]))
        );
    }

    #[test]
    fn test_tuple_literal_homogeneous_shape() {
        let code = r#"
def f(a: Float[Array, "features"], b: Float[Array, "features"]):
    pair = (a, b)
"#;
        let analysis = analyze_simple(code);
        assert_eq!(find_shape(&analysis, "pair"), Some(&shape(&["features"])));
    }

    #[test]
    fn test_tuple_literal_heterogeneous_shape_stays_unshaped() {
        let code = r#"
def f(a: Float[Array, "m"], b: Float[Array, "n"]):
    pair = (a, b)
"#;
        let analysis = analyze_simple(code);
        assert_eq!(find_shape(&analysis, "pair"), None);
    }

    #[test]
    fn test_tuple_literal_unresolvable_element_stays_unshaped() {
        let code = r#"
def f(a: Float[Array, "features"], b):
    pair = (a, b)
"#;
        let analysis = analyze_simple(code);
        assert_eq!(find_shape(&analysis, "pair"), None);
    }

    #[test]
    fn test_chained_method_reshape_then_flatten() {
        let code = r#"
def f(x: Float[Array, "3 4"]):
    y = x.reshape(3, 4).flatten()
"#;
        let analysis = analyze_simple(code);
        // reshape(3,4) → [3, 4], flatten() → [12]
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["12"]))
        );
    }

    #[test]
    fn test_unary_tilde_preserves_shape() {
        let code = r#"
def f(x: Float[Array, "3 5"]):
    y = ~x
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3", "5"]))
        );
    }

    /// A lone (single-entry, whole-file-range) scoped alias binding, for
    /// tests that don't care about class scoping.
    fn lone_alias(value: &str) -> Vec<ScopedSelfAttrAlias> {
        vec![ScopedSelfAttrAlias {
            class_start: 0,
            class_end: usize::MAX,
            value: value.to_string(),
        }]
    }

    #[test]
    fn test_normalize_dim_self_attr_alias() {
        let mut aliases: HashMap<String, Vec<ScopedSelfAttrAlias>> = HashMap::new();
        aliases.insert("dt_rank".to_string(), lone_alias("dt_rank"));
        aliases.insert("d_state".to_string(), lone_alias("d_state"));
        let h = ClassHierarchy::default();
        // single token
        assert_eq!(normalize_dim("self.dt_rank", &aliases, &h, 0), "dt_rank");
        // inside an expression, multiple tokens
        assert_eq!(
            normalize_dim("self.dt_rank + self.d_state", &aliases, &h, 0),
            "dt_rank + d_state"
        );
        // unaliased self.attr is left untouched
        assert_eq!(normalize_dim("self.unknown", &aliases, &h, 0), "self.unknown");
        // prefix collision: self.dt_rank must not match self.dt_rank2
        assert_eq!(normalize_dim("self.dt_rank2", &aliases, &h, 0), "self.dt_rank2");
        // no self. → unchanged
        assert_eq!(normalize_dim("seq_length", &aliases, &h, 0), "seq_length");
    }

    #[test]
    fn test_normalize_dim_class_scoped_alias_no_cross_class_collision() {
        // Two classes both alias `self.rank`, to different identifiers.
        // `resolve_alias_at` must pick the binding whose class range
        // contains the use-site byte, not silently take whichever was
        // inserted last (the bug the file-global alias map had).
        let mut aliases: HashMap<String, Vec<ScopedSelfAttrAlias>> = HashMap::new();
        aliases.insert(
            "rank".to_string(),
            vec![
                ScopedSelfAttrAlias {
                    class_start: 0,
                    class_end: 100,
                    value: "dt_rank".to_string(),
                },
                ScopedSelfAttrAlias {
                    class_start: 100,
                    class_end: 200,
                    value: "other_rank".to_string(),
                },
            ],
        );

        let h = ClassHierarchy::standalone(&[(0, 100), (100, 200)]);
        assert_eq!(normalize_dim("self.rank", &aliases, &h, 50), "dt_rank");
        assert_eq!(normalize_dim("self.rank", &aliases, &h, 150), "other_rank");
        // Outside every class range: left untouched.
        assert_eq!(normalize_dim("self.rank", &aliases, &h, 250), "self.rank");
    }

    #[test]
    fn test_subscript_integer_drops_leading_axis() {
        let code = r#"
def f(x: Float[Array, "3 5"]):
    y = x[0]
"#;
        let analysis = analyze_simple(code);
        // x[0] drops the leading axis -> rank-1.
        assert_eq!(find_shape(&analysis, "y"), Some(&shape(&["5"])));
    }

    #[test]
    fn test_parenthesized_call_preserves_shape() {
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"]):
    y = (jnp.sum(x, axis=1))
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["3"]))
        );
    }

    #[test]
    fn test_jax_nn_softplus_wrapping_shaped_call() {
        // Acceptance criteria: jax.nn.softplus(some_already_shaped_call(...)) gets a shape
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"]):
    y = jax.nn.softplus(jnp.reshape(x, (5, 3)))
"#;
        let analysis = analyze_simple(code);
        assert_eq!(
            find_shape(&analysis, "y"),
            Some(&shape(&["5", "3"]))
        );
    }

    #[test]
    fn test_no_synth_keys_leak_into_scope_shapes() {
        // After analyzing code with nested calls, no __synth_* keys should
        // appear in any scope's public shapes map.
        let code = r#"
import jax.numpy as jnp
def f(x: Float[Array, "3 5"]):
    y = jnp.exp(jnp.reshape(x, (5, 3)))
"#;
        let analysis = analyze_simple(code);
        for scope in &analysis.scopes {
            for key in scope.shapes.keys() {
                assert!(
                    !key.starts_with("__synth_"),
                    "__synth_* key leaked into scope.shapes: {}",
                    key
                );
            }
        }
    }

    #[test]
    fn test_broadcast_equal_shapes_unchanged() {
        let l = shape(&["batch", "features"]);
        assert_eq!(
            apply_elementwise_shape(&l, &l, BinaryOp::Add),
            Ok(Some(shape(&["batch", "features"])))
        );
    }

    #[test]
    fn test_broadcast_trailing_one_each_side() {
        let a = shape(&["3", "1"]);
        let b = shape(&["3", "5"]);
        assert_eq!(
            apply_elementwise_shape(&a, &b, BinaryOp::Mul),
            Ok(Some(shape(&["3", "5"])))
        );
        assert_eq!(
            apply_elementwise_shape(&b, &a, BinaryOp::Mul),
            Ok(Some(shape(&["3", "5"])))
        );
    }

    #[test]
    fn test_broadcast_rank_mismatch_leading_dims_pass_through() {
        let x = shape(&["batch", "f"]);
        let bias = shape(&["f"]);
        assert_eq!(
            apply_elementwise_shape(&x, &bias, BinaryOp::Add),
            Ok(Some(shape(&["batch", "f"])))
        );
        assert_eq!(
            apply_elementwise_shape(&bias, &x, BinaryOp::Add),
            Ok(Some(shape(&["batch", "f"])))
        );
    }

    #[test]
    fn test_broadcast_rank_zero_broadcasts_like_scalar() {
        // A resolved rank-0 ("scalar") shape — e.g. a plain `decay: float`
        // function parameter seeded as `[]` by `extract_jaxtyping_shapes`
        // — broadcasts against anything, same as a literal scalar (numpy
        // semantics: a 0-d array is a scalar). Needed for lazy call-site
        // parameter seeding to shape binops like `mean + (1 - decay) * delta`.
        let x = shape(&["3", "5"]);
        assert_eq!(apply_elementwise_shape(&[], &x, BinaryOp::Add), Ok(Some(x.clone())));
        assert_eq!(apply_elementwise_shape(&x, &[], BinaryOp::Add), Ok(Some(x)));
        assert_eq!(
            apply_elementwise_shape(&[], &[], BinaryOp::Add),
            Ok(Some(Vec::new()))
        );
    }

    #[test]
    fn test_broadcast_incompatible_concrete_dims() {
        assert!(apply_elementwise_shape(&shape(&["3"]), &shape(&["4"]), BinaryOp::Add).is_err());
    }

    #[test]
    fn test_broadcast_incompatible_symbolic_dims() {
        assert!(
            apply_elementwise_shape(&shape(&["batch"]), &shape(&["seq"]), BinaryOp::Add).is_err()
        );
    }

    #[test]
    fn test_minmax_tuple_reduction_and_binding_safety() {
        let code = r#"import jax.numpy as jnp
from jax.numpy import minmax as mm
def f(x: Float[Array, "2 3 5"]):
    low, high = mm(jnp.exp(x), axis=1, keepdims=True)
    kw_low, kw_high = mm(a=jnp.exp(x), axis=2)
    reordered_low, reordered_high = mm(axis=1, a=jnp.exp(x))
    all_low, all_high = jnp.minmax(x)
    single = jnp.minmax(x, axis=1)
    extra_a, extra_b, extra_c = jnp.minmax(x, axis=1)
    bad_low, bad_high = mm(x, axis=3)
"#;
        let analysis = analyze(code);
        for name in ["low", "high"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["2", "1", "5"])));
        }
        for name in ["kw_low", "kw_high"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["2", "3"])));
        }
        for name in ["reordered_low", "reordered_high"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["2", "5"])));
        }
        for name in ["all_low", "all_high"] {
            assert_eq!(find_shape(&analysis, name), Some(&Vec::new()));
        }
        for name in ["single", "extra_a", "extra_b", "extra_c", "bad_low", "bad_high"] {
            assert_eq!(find_shape(&analysis, name), None);
        }
        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        let range = &analysis.errors[0].range;
        assert_eq!(&code[range.start_byte..range.end_byte], "(x, axis=3)");
    }

    #[test]
    fn test_topk_tuple_axis_alias_nested_and_bounds() {
        let code = r#"import jax.numpy as jnp
from jax.numpy import top_k as tk
import torch
def f(x: Float[Array, "2 3 5"], k: int):
    v, i = jnp.top_k(jnp.exp(x), 2, axis=1)
    v_alias, i_alias = tk(x, 3, axis=-1)
    single = jnp.top_k(x, 2, axis=0)
    extra_a, extra_b, extra_c = tk(x, 2, axis=0)
    symbolic_v, symbolic_i = tk(x, k, axis=1)
    unknown_v, unknown_i = tk(x, jnp.sum(x), axis=1)
    bad_k_v, bad_k_i = tk(x, 6, axis=2)
    bad_v, bad_i = tk(x, 2, axis=3)
def g(t: Float[Array, "2 3 5"]):
    v_kw, i_kw = torch.topk(input=torch.exp(t), k=1, dim=0)
    v_torch, i_torch = torch.topk(t, 4, dim=2)
"#;
        let analysis = analyze(code);
        for name in ["v", "i"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["2", "2", "5"])));
        }
        for name in ["v_alias", "i_alias"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["2", "3", "3"])));
        }
        for name in ["symbolic_v", "symbolic_i"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["2", "k", "5"])));
        }
        for name in ["v_kw", "i_kw"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["1", "3", "5"])));
        }
        for name in ["v_torch", "i_torch"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["2", "3", "4"])));
        }
        for name in ["single", "extra_a", "extra_b", "extra_c", "unknown_v", "unknown_i", "bad_k_v", "bad_k_i", "bad_v", "bad_i"] {
            assert_eq!(find_shape(&analysis, name), None);
        }
        assert_eq!(analysis.errors.len(), 2, "{:?}", analysis.errors);
        assert!(analysis.errors[0].message.contains("top_k k=6 exceeds axis size 5"));
        assert!(analysis.errors[1].message.contains("out of bounds"));
        let ranges: Vec<_> = analysis.errors.iter().map(|error| {
            &code[error.range.start_byte..error.range.end_byte]
        }).collect();
        assert_eq!(ranges, ["(x, 6, axis=2)", "(x, 2, axis=3)"]);
    }

    #[test]
    fn test_multi_input_atleast_keeps_each_array_shape() {
        let code = r#"import jax.numpy as jnp
import torch
def f(s: float, v: Float[Array, "5"], m: Float[Array, "2 5"]):
    a, b = jnp.atleast_1d(s, v)
    c, d = jnp.atleast_2d(jnp.exp(v), m)
    e, g, h = jnp.atleast_3d(s, v, m)
    t, u = torch.atleast_2d(v, m)
    one = jnp.atleast_2d(v)
    not_a, not_b = jnp.atleast_2d((v, v))
    unpacked_a, unpacked_b = jnp.atleast_2d(v)
    unpacked_one, = jnp.atleast_2d(v)
    packed = jnp.atleast_2d(v, m)
    torch_packed = torch.atleast_2d(v, m)
"#;
        let analysis = analyze(code);
        for (name, dims) in [
            ("a", &["1"][..]),
            ("b", &["5"][..]),
            ("c", &["1", "5"][..]),
            ("d", &["2", "5"][..]),
            ("e", &["1", "1", "1"][..]),
            ("g", &["1", "5", "1"][..]),
            ("h", &["2", "5", "1"][..]),
            ("t", &["1", "5"][..]),
            ("u", &["2", "5"][..]),
            ("one", &["1", "5"][..]),
        ] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(dims)), "{name}");
        }
        assert_eq!(find_shape(&analysis, "not_a"), None);
        for name in ["not_b", "unpacked_a", "unpacked_b", "unpacked_one", "packed", "torch_packed"] {
            assert_eq!(find_shape(&analysis, name), None);
        }
        assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    }

    #[test]
    fn test_broadcast_array_tuple_uses_shared_shape_and_reports_mismatch() {
        let code = r#"import jax.numpy as jnp
import torch
def f(x: Float[Array, "3 1"], y: Float[Array, "1 5"], bad: Float[Array, "4 5"]):
    a, b = jnp.broadcast_arrays(jnp.exp(x), y)
    c, d = torch.broadcast_tensors(x, y)
    single = jnp.broadcast_arrays(x, y)
    wrong_a, wrong_b = jnp.broadcast_arrays(x, bad)
"#;
        let analysis = analyze(code);
        for name in ["a", "b", "c", "d"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["3", "5"])));
        }
        for name in ["single", "wrong_a", "wrong_b"] {
            assert_eq!(find_shape(&analysis, name), None);
        }
        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        let range = &analysis.errors[0].range;
        assert_eq!(&code[range.start_byte..range.end_byte], "(x, bad)");
    }

    #[test]
    fn test_resolve_nested_positional_and_keyword_keeps_error_range() {
        let code = r#"import jax.numpy as jnp
def f(x: Float[Array, "2 3"]):
    positional = jnp.sum(jnp.exp(x), axis=1)
    keyword = jnp.sum(x=jnp.exp(x), axis=0)
    bad = jnp.sum(jnp.exp(x), axis=(0, 3))
"#;
        let analysis = analyze(code);
        assert_eq!(find_shape(&analysis, "positional"), Some(&shape(&["2"])));
        assert_eq!(find_shape(&analysis, "keyword"), Some(&shape(&["3"])));
        assert_eq!(find_shape(&analysis, "bad"), None);
        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        let range = &analysis.errors[0].range;
        assert_eq!(
            &code[range.start_byte..range.end_byte],
            "(jnp.exp(x), axis=(0, 3))"
        );
    }

    #[test]
    fn test_polynomial_coefficients_keep_sequence_text() {
        let code = r#"import jax
import jax.numpy as jnp
def f(x: Float[Array, "1 4"], a: Float[Array, "3 1"], b: Float[Array, "3 1"], bad: Float[Array, "2 5"]):
    tuple_coeffs = jax.lax.polynomial(x, (a, b))
    list_coeffs = jax.lax.polynomial(jnp.exp(x), [1., a, 3])
    scalar_coeffs = jax.lax.polynomial(x, (1., 2.))
    keyword_coeffs = jax.lax.polynomial(coeffs=(a, b), x=jnp.exp(x))
    unknown_coeffs = jax.lax.polynomial(x, [missing])
    bad_coeffs = jax.lax.polynomial(x, [bad])
"#;
        let analysis = analyze(code);
        for name in ["tuple_coeffs", "list_coeffs", "keyword_coeffs"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["3", "4"])), "{name}");
        }
        assert_eq!(find_shape(&analysis, "scalar_coeffs"), Some(&shape(&["1", "4"])));
        assert_eq!(find_shape(&analysis, "unknown_coeffs"), None);
        assert_eq!(find_shape(&analysis, "bad_coeffs"), None);
        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        let range = &analysis.errors[0].range;
        assert_eq!(&code[range.start_byte..range.end_byte], "(x, [bad])");
    }

    #[test]
    fn test_new_known_calls_preserve_scalars_and_error_locations() {
        let code = r#"import jax
import jax.numpy as jnp
import numpy as np
import torch
def f(key, dynamic, y: Float[Array, "2 3"], s: float, p: Float[Array, "7"], batched: Float[Array, "2 7"], arr: Float[Array, "n 4"]):
    inv = jax.scipy.special.erfcinv(y=y)
    scalar_inv = jax.scipy.special.erfcinv(y=s)
    logarithm = jax.lax.log2(jnp.exp(y))
    one_minus_square = jax.lax.one_minus_square(y)
    sample = jax.random.multinomial(key, n=3, p=p, shape=())
    unknown = jax.random.multinomial(key, 3, p, shape=dynamic)
    bad_batch = jax.random.multinomial(key, 3, p=batched, shape=())
    padded_valid = np.pad(arr, pad_width=1)
    padded = np.pad(arr, pad_width={0: (1, 2), -1: 3})
    bad_pad = np.pad(arr, pad_width={2: 1})
def g(mat: Float[Array, "2 3 3"], nonsquare: Float[Array, "2 3 4"]):
    root = torch.linalg.matrix_sqrth(A=mat)
    bad_root = torch.linalg.matrix_sqrth(A=nonsquare)
"#;
        let analysis = analyze(code);
        assert_eq!(find_shape(&analysis, "inv"), Some(&shape(&["2", "3"])));
        assert_eq!(find_shape(&analysis, "scalar_inv"), Some(&Vec::new()));
        assert_eq!(find_shape(&analysis, "logarithm"), Some(&shape(&["2", "3"])));
        assert_eq!(find_shape(&analysis, "one_minus_square"), Some(&shape(&["2", "3"])));
        assert_eq!(find_shape(&analysis, "sample"), Some(&shape(&["7"])));
        assert_eq!(find_shape(&analysis, "padded_valid"), Some(&shape(&["n+2", "6"])));
        assert_eq!(find_shape(&analysis, "padded"), Some(&shape(&["n+3", "10"])));
        assert_eq!(find_shape(&analysis, "root"), Some(&shape(&["2", "3", "3"])));
        for name in ["unknown", "bad_batch", "bad_pad", "bad_root"] {
            assert_eq!(find_shape(&analysis, name), None, "{name}");
        }
        let ranges: Vec<_> = analysis.errors.iter().map(|error| {
            &code[error.range.start_byte..error.range.end_byte]
        }).collect();
        assert_eq!(ranges, ["(key, 3, p=batched, shape=())", "(arr, pad_width={2: 1})", "(A=nonsquare)"]);
    }

    #[test]
    fn test_array_split_tuple_defaults_uneven_parts_and_unknown_axes() {
        let code = r#"import numpy as np
import jax.numpy as jnp
import torch
def f(x: Float[Array, "4"], uneven: Float[Array, "7"], symbolic: Float[Array, "n"], axis: int):
    left, right = np.split(x, 2)
    a, b, c = np.array_split(uneven, 3)
    d, e, f = jnp.array_split(uneven, 3)
    g, h, i = torch.tensor_split(uneven, 3)
    single = np.array_split(uneven, 3)
    dyn_a, dyn_b, dyn_c = np.array_split(uneven, 3, axis=axis)
    sym_a, sym_b, sym_c = np.array_split(symbolic, 3)
    wrong_a, wrong_b, wrong_c = np.split(x, 3)
"#;
        let analysis = analyze(code);
        for name in ["left", "right", "b", "c", "e", "f", "h", "i"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["2"])), "{name}");
        }
        for name in ["a", "d", "g"] {
            assert_eq!(find_shape(&analysis, name), Some(&shape(&["3"])), "{name}");
        }
        for name in ["single", "dyn_a", "dyn_b", "dyn_c", "sym_a", "sym_b", "sym_c", "wrong_a", "wrong_b", "wrong_c"] {
            assert_eq!(find_shape(&analysis, name), None, "{name}");
        }
        assert_eq!(analysis.errors.len(), 1, "{:?}", analysis.errors);
        let range = &analysis.errors[0].range;
        assert_eq!(&code[range.start_byte..range.end_byte], "(x, 3)");
    }

    fn reference_global_names(node: Node, text: &str, out: &mut HashSet<String>) {
        if node.kind() == "global_statement" {
            insert_identifiers(node, text, out);
            return;
        }
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            reference_global_names(child, text, out);
        }
    }

    fn global_names_both(code: &str) -> (HashSet<String>, HashSet<String>) {
        let tree = parse(code);
        let (mut fast, mut slow) = (HashSet::new(), HashSet::new());
        collect_global_names(tree.root_node(), code, &mut fast);
        reference_global_names(tree.root_node(), code, &mut slow);
        (fast, slow)
    }

    #[test]
    fn global_statements_are_found_wherever_a_statement_can_stand() {
        let code = "g1 = 0\ndef f():\n    global a, b\n    if c:\n        global d\n    else:\n        global e\n    for i in x:\n        global f1\n    while y:\n        global g\n    try:\n        global h\n    except E:\n        global i1\n    finally:\n        global j\n    with m:\n        global k\n    match z:\n        case 1:\n            global l\n    class C:\n        global n\n        def m(self):\n            global o\n    if q: global p\n@dec\ndef g():\n    global r\nasync def h():\n    async with a:\n        global s\n    async for i in b:\n        global t\nx = lambda: 1\n";
        let (fast, slow) = global_names_both(code);
        assert_eq!(fast, slow);
        assert_eq!(fast.len(), 17, "{fast:?}");
    }

    #[test]
    fn global_statements_of_the_project_files_are_found() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut checked = 0;
        for dir in ["corpus", "bench/stress", "bench"] {
            let Ok(entries) = std::fs::read_dir(root.join(dir)) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "py")
                    && let Ok(code) = std::fs::read_to_string(&path)
                {
                    let (fast, slow) = global_names_both(&code);
                    assert_eq!(fast, slow, "{}", path.display());
                    checked += 1;
                }
            }
        }
        assert!(checked > 20);
    }
}
