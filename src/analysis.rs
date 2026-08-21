use std::collections::HashMap;
use std::path::PathBuf;

use tree_sitter::{Node, Parser};

use crate::known_functions::{
    apply_known_function, apply_known_kthvalue_shape, apply_known_linalg_lstsq_solution,
    apply_known_reduction, apply_known_topk_shape, apply_method_call, classify_known_function,
    classify_method_call, compute_chunk_shapes, compute_einops_pack_shape,
    compute_fixed_axis_split_shapes, compute_split_shapes, compute_torch_split_shapes,
    compute_unbind_shape,
};
use crate::layers::{
    apply_layer_application, apply_layer_kind, classify_inline_constructor,
    extract_layer_assignments_scoped, extract_self_attr_layers_by_class,
};
use crate::python_ast::{
    build_import_map, extract_call_arguments, extract_jaxtyping_shapes,
    extract_self_attr_aliases_by_class,
};
use crate::resolution::{ResolutionCache, resolve_call_target, resolve_imported_function_shape};

use crate::types::*;

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
    let mut scopes = extract_jaxtyping_shapes(node, text)?;
    let import_map = build_import_map(node, text)?;
    let layer_records = extract_layer_assignments_scoped(
        node,
        text,
        &import_map,
        search_roots,
        &read_file,
        max_depth,
        cache,
    )?;
    let self_attr_layers = extract_self_attr_layers_by_class(
        node,
        text,
        &import_map,
        search_roots,
        &read_file,
        max_depth,
        cache,
    )?;
    let self_attr_aliases = extract_self_attr_aliases_by_class(node, text)?;

    let (applications, errors, assignment_shapes) = propagate_calls(
        node,
        text,
        &import_map,
        &layer_records,
        &self_attr_layers,
        &self_attr_aliases,
        search_roots,
        &read_file,
        max_depth,
        cache,
        &mut scopes,
    )?;

    let mut layers = HashMap::new();
    for rec in &layer_records {
        layers.insert(rec.name.clone(), rec.kind.clone());
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

    let import_map = build_import_map(node, text)?;
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
                kind: dark_spot_kind(rhs, text, &import_map),
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
    call_byte: usize,
    target: &str,
) -> Option<&'a LayerKind> {
    let mut best: Option<(usize, usize)> = None;
    for (i, rec) in records.iter().enumerate() {
        if rec.name != target {
            continue;
        }
        let Some(rec_scope_idx) = scope_index_for_byte(scopes, rec.byte_position) else {
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
    best.map(|(i, _)| &records[i].kind)
}

// ── Recursive shape evaluator (#34) ─────────────────────────────────────
//
// Replaces the flat query-based extraction (extract_calls /
// extract_method_calls / extract_binary_ops) with a recursive tree walk.
// For each assignment `lhs = rhs`, we call shape_of_expression(rhs) which
// recurses into nested calls, chained methods, unary wraps, parenthesised
// expressions, identifiers, and subscripts.

/// Context shared across recursive shape_of_expression calls within a single
/// propagate_calls pass.
struct ShapeCtx<'a> {
    text: &'a str,
    import_map: &'a HashMap<String, ImportPath>,
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
    /// for the older `apply_user_function` path) vs. needs seeding.
    original_shapes: &'a [HashMap<String, Vec<String>>],
}

/// Borrowed view of one scope's shapes plus its synthetic bindings.
/// Passed to the apply helpers as `&dyn ShapeLookup` so no merged map is
/// ever cloned (#43).
struct ScopeShapes<'c> {
    shapes: &'c HashMap<String, Vec<String>>,
    synthetics: Option<&'c HashMap<String, Vec<String>>>,
}

impl ShapeLookup for ScopeShapes<'_> {
    fn shape(&self, name: &str) -> Option<&Vec<String>> {
        self.shapes
            .get(name)
            .or_else(|| self.synthetics?.get(name))
    }
}

/// Canonicalize every dim of a shape through the `self.<attr> = <ident>`
/// alias map (no-op when the map is empty or no dim mentions `self.`).
/// `byte` is the call/assignment site used to pick the enclosing class's
/// alias binding when the same attr is aliased differently across classes.
fn normalize_shape(
    shape: Vec<String>,
    aliases: &HashMap<String, Vec<ScopedSelfAttrAlias>>,
    byte: usize,
) -> Vec<String> {
    if aliases.is_empty() {
        return shape;
    }
    shape
        .into_iter()
        .map(|d| normalize_dim(&d, aliases, byte))
        .collect()
}

/// Resolve `self.<attr>`'s aliased identifier in the class enclosing `byte`.
/// Falls back to a lone binding when the use site sits outside any class
/// that defines the attr; ambiguous cross-class matches return `None`.
/// Mirrors `ShapeCtx::self_attr_layer_at`.
fn resolve_alias_at<'a>(
    aliases: &'a HashMap<String, Vec<ScopedSelfAttrAlias>>,
    attr: &str,
    byte: usize,
) -> Option<&'a str> {
    let entries = aliases.get(attr)?;
    entries
        .iter()
        .filter(|e| e.class_start <= byte && byte < e.class_end)
        .min_by_key(|e| e.class_end - e.class_start)
        .or_else(|| (entries.len() == 1).then(|| &entries[0]))
        .map(|e| e.value.as_str())
}

/// Replace each `self.<attr>` token in a dim expression with its aliased value
/// (e.g. `self.dt_rank + self.d_state` → `dt_rank + d_state` when both attrs
/// were assigned from same-named identifiers). Tokens whose attribute has no
/// alias (in the class enclosing `byte`) are left untouched.
fn normalize_dim(dim: &str, aliases: &HashMap<String, Vec<ScopedSelfAttrAlias>>, byte: usize) -> String {
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
            if let Some(val) = resolve_alias_at(aliases, &dim[attr_start..j], byte) {
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
    /// Resolve `self.<attr>` to the layer bound in the class enclosing `byte`.
    /// Falls back to a lone binding when the use site sits outside any class
    /// that defines the attr; ambiguous cross-class matches return None.
    fn self_attr_layer_at(&self, attr: &str, byte: usize) -> Option<&LayerKind> {
        let entries = self.self_attr_layers.get(attr)?;
        entries
            .iter()
            .filter(|e| e.class_start <= byte && byte < e.class_end)
            .min_by_key(|e| e.class_end - e.class_start)
            .or_else(|| (entries.len() == 1).then(|| &entries[0]))
            .map(|e| &e.kind)
    }

    /// Insert a shape under a synthetic name and return that name.
    /// Used when an inline expression (e.g. a nested call) produces a shape
    /// but has no variable binding of its own. The helpers in
    /// `known_functions` look up arguments by name in `shapes`, so we need
    /// a name. `byte` is the originating expression's byte position, used to
    /// resolve `self.<attr>` aliases against the enclosing class.
    fn bind_synthetic(&mut self, shape: Vec<String>, scope_idx: usize, byte: usize) -> String {
        let name = format!("__synth_{}", self.synthetic_counter);
        self.synthetic_counter += 1;
        let shape = normalize_shape(shape, self.aliases, byte);
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
        let shape = normalize_shape(shape, self.aliases, byte);
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
    fn evict_binding(&mut self, scope_idx: usize, name: &str) {
        self.scopes[scope_idx].shapes.remove(name);
    }

    /// Look up a shape by name in the given scope, checking both real
    /// user-visible shapes and the synthetic side-channel.
    fn resolve_shape(&self, name: &str, scope_idx: usize) -> Option<Vec<String>> {
        self.scope_shapes(scope_idx).shape(name).cloned()
    }

    /// Zero-copy `ShapeLookup` over a scope's shapes plus its synthetic
    /// bindings, for passing to the apply helpers.
    fn scope_shapes(&self, scope_idx: usize) -> ScopeShapes<'_> {
        ScopeShapes {
            shapes: &self.scopes[scope_idx].shapes,
            synthetics: self.synthetics.get(&scope_idx),
        }
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
    let scope_idx = scope_index_for_byte(ctx.scopes, node.start_byte())?;
    ctx.resolve_shape(name, scope_idx)
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

/// Binary operator: delegates to the existing `apply_binary_op`-style logic
/// but works on tree-sitter nodes directly rather than `BinaryOpInfo`.
/// A binary-op operand: a scalar literal (`2`, `1.0`) or an array with a
/// resolved shape. Scalars broadcast trivially against any array.
enum BinopOperand {
    Scalar,
    Shaped(Vec<String>),
}

fn binop_operand(node: Node, scope_idx: usize, ctx: &mut ShapeCtx) -> Option<BinopOperand> {
    match node.kind() {
        "integer" | "float" => Some(BinopOperand::Scalar),
        "identifier" => {
            let name = node.utf8_text(ctx.text.as_bytes()).ok()?;
            ctx.resolve_shape(name, scope_idx).map(BinopOperand::Shaped)
        }
        // Nested expressions (calls, parenthesized binops, unary ops, …)
        // recurse through the main evaluator.
        _ => shape_of_expression(node, ctx).map(BinopOperand::Shaped),
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

    let scope_idx = scope_index_for_byte(ctx.scopes, node.start_byte())?;
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

/// Resolve a `call` node's shape.
///
/// This is the main dispatch point that handles:
/// Build the `ShapeError` for a failed method-call shape rule, classifying
/// it as `Approximation` for the one documented case where the rule is known
/// to be unsound rather than genuinely contradictory: `x.transpose(i, j)`
/// dispatches to the same axes-permutation rule as `jnp.transpose`/`.permute`
/// (`apply_known_transpose`), but torch's 2-arg method form swaps exactly
/// those two axes and leaves the rest alone — it's valid for *any* rank, not
/// just rank 2 (see `llm.txt` "Known limitations"). `apply_known_transpose`
/// can't tell the two call styles apart, so an "expected N axes, got 2"
/// error from exactly 2 positional args is this approximation firing, not a
/// real mismatch.
fn shape_error_for_method(
    known: &KnownFunction,
    args: &[CallArgument],
    variable: String,
    message: String,
    range: tree_sitter::Range,
) -> ShapeError {
    let positional_count = args
        .iter()
        .filter(|a| matches!(a, CallArgument::Positional { .. }))
        .count();
    if matches!(known, KnownFunction::Transpose) && positional_count == 2 {
        ShapeError::approximation(variable, message, range)
    } else {
        ShapeError::mismatch(variable, message, range)
    }
}

/// 1. Chained method calls (e.g. `x.reshape(3,4).sum(axis=1)`)
/// 2. Free function calls (e.g. `jnp.exp(...)`)
/// 3. Layer applications
/// 4. vmap bindings and calls
/// 5. User-defined function propagation
/// 6. Known-function resolution
fn shape_of_call(node: Node, ctx: &mut ShapeCtx) -> Option<Vec<String>> {
    let call_byte = node.start_byte();
    let scope_idx = scope_index_for_byte(ctx.scopes, call_byte)?;

    let func_node = node.child_by_field_name("function")?;
    let args_node = node.child_by_field_name("arguments")?;

    // ── Call-of-a-call ──
    // `jax.vmap(self.layer)(x)` (inline vmap, issue #35) or a layer
    // constructed and applied in one expression like `nn.Dense(64)(x)`
    // (the dominant flax style).
    if func_node.kind() == "call" {
        if let Some(shape) = shape_of_inline_vmap(func_node, args_node, scope_idx, ctx) {
            return Some(shape);
        }
        return shape_of_inline_layer(func_node, args_node, scope_idx, ctx);
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
                .cloned()
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
                let receiver_name =
                    ctx.bind_synthetic(receiver_shape, scope_idx, obj_node.start_byte());
                let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
                let args = resolve_call_args(raw_args, args_node, scope_idx, ctx)?;

                if let Some(known) = classify_method_call(&method_name) {
                    let result = apply_method_call(&known, &receiver_name, &args, &ctx.scope_shapes(scope_idx));
                    return match result {
                        Ok(Some(shape)) => Some(shape),
                        Ok(None) => None,
                        Err(message) => {
                            ctx.errors.push(shape_error_for_method(&known, &args, method_name, message, args_node.range()));
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
            if !ctx.import_map.contains_key(&receiver_name) {
                let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
                let args = resolve_call_args(raw_args, args_node, scope_idx, ctx)?;

                if let Some(known) = classify_method_call(&method_name) {
                    let result = apply_method_call(
                        &known,
                        &receiver_name,
                        &args,
                        &ctx.scope_shapes(scope_idx),
                    );
                    return match result {
                        Ok(Some(shape)) => Some(shape),
                        Ok(None) => None,
                        Err(message) => {
                            ctx.errors.push(shape_error_for_method(&known, &args, receiver_name, message, args_node.range()));
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
    let raw_args = extract_call_arguments(args_node, ctx.text).ok()?;
    let args = resolve_call_args(raw_args, args_node, scope_idx, ctx)?;

    // 1. Layer check
    if let Some(kind) = find_scoped_layer(ctx.layer_records, ctx.scopes, call_byte, &target) {
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
            kind: kind.clone(),
            range: args_node.range(),
        };
        let result = apply_layer_application(&application, &ctx.scope_shapes(scope_idx));
        ctx.applications.push(LayerApplication {
            variable: application.variable.clone(),
            layer: application.layer.clone(),
            input: application.input.clone(),
            kind: application.kind.clone(),
            range: application.range,
        });
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
        let resolved_for_map = resolve_call_target(&target, ctx.import_map);
        if matches!(classify_known_function(&resolved_for_map), Some(KnownFunction::LaxMap)) {
            let mut positionals = args.iter().filter_map(|a| match a {
                CallArgument::Positional { value } => Some(value.clone()),
                _ => None,
            });
            let callable = positionals.next()?;
            let rest: Vec<CallArgument> = args.iter().skip(1).cloned().collect();
            if let Some(attr) = callable.strip_prefix("self.")
                && let Some(layer) = ctx.self_attr_layer_at(attr, call_byte).cloned()
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
            if !callable.contains('.') {
                let info = VmapInfo {
                    wrapped: callable,
                    in_axes: 0,
                    out_axes: 0,
                };
                return match apply_vmap_call(&info, &rest, &ctx.scope_shapes(scope_idx), ctx.scopes) {
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

    // 3. User-defined function propagation. `self.<method>(...)` resolves
    // to the method's function scope by bare name — methods are function
    // definitions like any other. When the callee has no `-> ReturnType`
    // annotation (e.g. a typical `forward(self, x: Float[Array, "..."])`),
    // `apply_user_function` falls back to tracing a bare `return <name>`
    // statement against the callee's own already-computed body shapes.
    //
    // Gated on the callee being FULLY annotated (checked against the
    // call-site-independent `original_shapes` snapshot, never the live,
    // mutable `scopes[...].shapes`): `apply_user_function`'s bare-return
    // trace reads `callee.shapes.get(name)` directly, which is only sound
    // when every one of the callee's params was statically annotated (so
    // its body shapes are the same regardless of which call site asks) —
    // once lazy call-site parameter seeding (3b) has ever specialized a
    // partially-annotated callee, its `shapes` map holds a PARTICULAR call
    // site's seeded values, and reusing them here for a call site with
    // different argument shapes would silently propagate the wrong shape.
    // Partially-annotated callees always go through 3b instead, which
    // re-seeds fresh from `original_shapes` on every call.
    let user_target = target.strip_prefix("self.").unwrap_or(&target);
    if !user_target.contains('.') {
        let callee_idx = find_callee_scope(user_target, Some(call_byte), ctx.scopes);
        let fully_annotated = callee_idx.is_none_or(|idx| callee_all_params_annotated(idx, ctx));

        if fully_annotated {
            if let Some(result) = apply_user_function(
                user_target,
                call_byte,
                &args,
                &ctx.scope_shapes(scope_idx),
                ctx.scopes,
                ctx.text,
            ) {
                match result {
                    Ok(Some(shape)) => return Some(shape),
                    Ok(None) => {}
                    Err(message) => {
                        ctx.errors.push(ShapeError::mismatch(target, message, args_node.range()));
                        return None;
                    }
                }
            }
        } else {
            // 3b. Lazy call-site parameter seeding (`llm.txt`'s "Known
            // architectural limit"): the callee is a same-file function,
            // nested closure, or `self.<method>` with at least one
            // un-annotated parameter. Seed the un-annotated params from
            // this call site's resolved argument shapes and evaluate the
            // callee's body on demand.
            if let Some(shape) =
                apply_seeded_user_function(user_target, call_byte, &args, scope_idx, ctx)
            {
                return Some(shape);
            }
        }
    }

    // 4. vmap *recording* (e.g. `vf = jax.vmap(f)`)
    let resolved = resolve_call_target(&target, ctx.import_map);
    if let Some(KnownFunction::Vmap) = classify_known_function(&resolved)
        && let Some(_info) = parse_vmap_call(&args) {
            // The caller (propagate_calls) will handle binding the LHS
            // variable name to the vmap target. For now we just signal
            // that a vmap binding was recorded by returning None (vmap
            // itself has no shape).
            // We return the info via a side channel — but since
            // propagate_calls handles the LHS binding, we use a special
            // marker. Actually, we'll handle vmap recording in
            // propagate_calls directly when we see the assignment.
            // Here we just return None — the vmap binding is done by the
            // caller which has the LHS variable name.
            return None;
        }

    // 5. Known-function check
    if let Some(known) = classify_known_function(&resolved) {
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
        let Some(CallArgument::Positional { value }) = args.first() else {
            return None;
        };
        return ctx.resolve_shape(value, scope_idx);
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
        ctx.import_map,
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

/// Given the raw `CallArgument`s extracted from a call's argument_list node,
/// recursively evaluate any argument that is itself a non-trivial expression
/// (nested call, unary operator, etc.) and bind its shape under a synthetic
/// name so that downstream lookups in `apply_known_function` etc. can find it.
///
/// Simple identifier arguments pass through unchanged.
fn resolve_call_args(
    raw_args: Vec<CallArgument>,
    args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<CallArgument>> {
    let mut resolved = Vec::with_capacity(raw_args.len());

    for (i, arg) in raw_args.into_iter().enumerate() {
        match arg {
            CallArgument::Positional { value } => {
                // Check if the argument text looks like a simple identifier.
                // This is a fast-path heuristic: strings matching
                // [A-Za-z0-9_]+ are either identifiers (which are already
                // in shapes or not) or numeric literals (which won't
                // match any shape entry and are silently skipped).
                // Complex expressions (dots, parens, operators) fall through
                // to the recursive evaluator.
                let looks_like_identifier = value
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_');

                if looks_like_identifier {
                    resolved.push(CallArgument::Positional { value });
                } else {
                    // Find the corresponding named child in args_node.
                    // Positional args map to named children that aren't
                    // keyword_arguments.
                    let child_node = find_positional_arg_node(args_node, i)?;
                    let shape = shape_of_expression(child_node, ctx);
                    if let Some(shape) = shape {
                        let synth_name =
                            ctx.bind_synthetic(shape, scope_idx, child_node.start_byte());
                        resolved.push(CallArgument::Positional {
                            value: synth_name,
                        });
                    } else {
                        // Can't resolve this argument — pass through as-is.
                        // Downstream lookups will fail and return None,
                        // which is the expected fallback.
                        resolved.push(CallArgument::Positional { value });
                    }
                }
            }
            CallArgument::Keyword { name, value } => {
                // For keyword args, check if the value looks like a simple identifier.
                let looks_like_identifier = value
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_');
                if looks_like_identifier {
                    resolved.push(CallArgument::Keyword { name, value });
                } else {
                    // Find the keyword argument node and evaluate its value.
                    let kw_node = find_keyword_arg_node(args_node, &name, ctx.text.as_bytes())?;
                    let value_node = kw_node.child_by_field_name("value")?;
                    let shape = shape_of_expression(value_node, ctx);
                    if let Some(shape) = shape {
                        let synth_name =
                            ctx.bind_synthetic(shape, scope_idx, value_node.start_byte());
                        resolved.push(CallArgument::Keyword {
                            name,
                            value: synth_name,
                        });
                    } else {
                        resolved.push(CallArgument::Keyword { name, value });
                    }
                }
            }
        }
    }

    Some(resolved)
}

/// Find the i-th positional argument node in an argument_list.
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
}

/// Parse a vmap/filter_vmap call's arguments into a `VmapInfo`.
///
/// Returns `None` if:
/// - The first positional arg is not a bare identifier (contains `.` or is
///   not simple).
/// - `in_axes` or `out_axes` is present but not a literal integer (tuple,
///   variable, `None`, etc.).
fn parse_vmap_call(args: &[CallArgument]) -> Option<VmapInfo> {
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

/// Output of `propagate_calls`: layer applications, shape errors, and the
/// per-assignment shape records used for inlay hints.
type PropagateOutput = (Vec<LayerApplication>, Vec<ShapeError>, Vec<AssignmentShape>);

/// Collect every `function_definition` node in `node`'s subtree, in the
/// same pre-order (push-before-recurse) traversal `extract_jaxtyping_shapes`
/// uses to push scopes — so index `i` here corresponds exactly to
/// `scopes[i + 1]` (the module scope at index 0 has no function node).
fn collect_function_definition_nodes<'t>(node: Node<'t>, out: &mut Vec<Node<'t>>) {
    if node.kind() == "function_definition" {
        out.push(node);
    }
    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i as u32) {
            collect_function_definition_nodes(child, out);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn propagate_calls(
    node: Node,
    text: &str,
    import_map: &HashMap<String, ImportPath>,
    layer_records: &[LayerAssignment],
    self_attr_layers: &HashMap<String, Vec<ScopedSelfAttrLayer>>,
    aliases: &HashMap<String, Vec<ScopedSelfAttrAlias>>,
    search_roots: &[PathBuf],
    read_file: &dyn Fn(&PathBuf) -> Option<String>,
    max_depth: usize,
    cache: Option<&ResolutionCache>,
    scopes: &mut [FunctionShapeScope],
) -> Result<PropagateOutput, String> {
    let mut applications = Vec::new();
    let mut errors = Vec::new();
    let mut assignment_shapes = Vec::new();
    let mut vmap_targets: HashMap<String, VmapInfo> = HashMap::new();

    // Collect all assignments (identifier and tuple-pattern LHS) in source order.
    let assignments = collect_assignment_items(node, text)?;

    let mut function_nodes = Vec::new();
    collect_function_definition_nodes(node, &mut function_nodes);
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

    let mut ctx = ShapeCtx {
        text,
        import_map,
        layer_records,
        self_attr_layers,
        aliases,
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
        scope_function_nodes: &scope_function_nodes,
        active_specializations: Vec::new(),
        specialized_scopes: std::collections::HashSet::new(),
        original_shapes: &original_shapes,
    };

    for (lhs, rhs_node, assignment_node) in assignments {
        process_assignment_item(lhs, rhs_node, assignment_node, &mut ctx);
    }

    // Also handle binary-operator expressions in return/yield/assert
    // statements. The recursive evaluator handles the expression itself,
    // but we need to walk these statement types to catch shape errors
    // (e.g., matmul dimension mismatches) that don't appear in assignments.
    let value_statements = collect_value_statements(node, text)?;
    for stmt in value_statements {
        let errors_before = ctx.errors.len();
        let _ = shape_of_expression(stmt, &mut ctx);
        // Fix up error variable names: for return/yield/assert contexts,
        // there is no LHS variable, so set the variable name to empty.
        for err in &mut ctx.errors[errors_before..] {
            err.variable = String::new();
        }
    }

    Ok((applications, errors, assignment_shapes))
}

/// Process one collected assignment item, binding its LHS to the RHS's
/// computed shape (or evicting a stale binding). Shared by the top-level
/// whole-file walk (`propagate_calls`) and lazy call-site parameter seeding
/// (`specialize_callee_call`, which re-collects and replays just a single
/// callee's own assignments on demand).
fn process_assignment_item(lhs: Lhs, rhs_node: Node, assignment_node: Node, ctx: &mut ShapeCtx) {
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
    let scope_idx = match scope_index_for_byte(ctx.scopes, rhs_node.start_byte()) {
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
            let resolved = resolve_call_target(target, ctx.import_map);
            if let Some(KnownFunction::Vmap) = classify_known_function(&resolved) {
                let args_node = rhs_node.child_by_field_name("arguments");
                if let Some(an) = args_node
                    && let Ok(args) = extract_call_arguments(an, ctx.text)
                        && let Some(info) = parse_vmap_call(&args) {
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
            if let Some(kind) =
                find_scoped_layer(ctx.layer_records, ctx.scopes, call_byte, target)
            {
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
                                kind: kind.clone(),
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
        ctx.record_binding(scope_idx, &lhs_name, shape, display_line, rhs_node.start_byte());
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
}

impl TupleTarget {
    /// All identifier names bound by this target (flattening one level of
    /// nesting), for coverage accounting.
    fn names(&self) -> Vec<&str> {
        match self {
            TupleTarget::Name(Some(n)) => vec![n.as_str()],
            TupleTarget::Name(None) => vec![],
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
        let child = node.named_child(i as u32).unwrap();

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
            continue;
        } else if child.kind() == "assignment" || child.kind() == "augmented_assignment" {
            push_assignment_item(child, text, out);
            // Also don't recurse into bare assignment nodes to avoid
            // double-counting.
            continue;
        }

        collect_items_recursive(child, text, out)?;
    }
    Ok(())
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
    let Some(scope_idx) = scope_index_for_byte(ctx.scopes, rhs_node.start_byte()) else {
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
    let Some(scope_idx) = scope_index_for_byte(ctx.scopes, rhs_node.start_byte()) else {
        return;
    };
    let Some(elem_shapes) = tuple_rhs_shapes(rhs_node, targets.len(), scope_idx, ctx) else {
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
            TupleTarget::Name(None) => {}
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
    n_targets: usize,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<Option<Vec<String>>>> {
    match rhs.kind() {
        "attribute" => {
            let attr = rhs.child_by_field_name("attribute")?;
            if attr.utf8_text(ctx.text.as_bytes()).ok()? != "shape" {
                return None;
            }
            let obj = rhs.child_by_field_name("object")?;
            let obj_shape = shape_of_expression(obj, ctx)?;
            if obj_shape.len() != n_targets {
                return None;
            }
            // Each unpacked dim is an integer scalar (zero-rank).
            Some(obj_shape.iter().map(|_| Some(Vec::new())).collect())
        }
        // `final_mean, final_var = final_carry` / `h_final, c_final =
        // carry_final` — unpacking a plain variable that itself holds a
        // homogeneous-shape tuple carry (bound via `shape_of_tuple`'s
        // same-shape rule, e.g. from a preceding `final_carry, _ =
        // jax.lax.scan(...)`, or a `Rnn`/`RnnCell` state tuple). Every
        // unpacked name gets that one shape, same convention as
        // `TupleTarget::Nested` for a single-statement `(h, c)` pattern.
        "identifier" => {
            let name = rhs.utf8_text(ctx.text.as_bytes()).ok()?;
            let shape = ctx.resolve_shape(name, scope_idx)?;
            Some((0..n_targets).map(|_| Some(shape.clone())).collect())
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
                && !ctx.import_map.contains_key(receiver)
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
                    Some(LayerKind::Rnn { hidden_size, .. }) => Some(hidden_size.clone()),
                    _ => None,
                };
                if let Some(hidden_size) = rnn_hidden_size {
                    if n_targets != 2 {
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
                    Some(LayerKind::RnnCell { hidden_size, .. }) => Some(hidden_size.clone()),
                    _ => None,
                };
                if let Some(hidden_size) = cell_hidden_size {
                    if n_targets != 2 {
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

            let resolved = resolve_call_target(target, ctx.import_map);
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
                let init_node = find_keyword_arg_node(args_node, "init", ctx.text.as_bytes())
                    .and_then(|kw| kw.child_by_field_name("value"))
                    .or_else(|| find_positional_arg_node(args_node, 1))?;
                let init_shape = shape_of_expression(init_node, ctx)?;
                let xs_node = find_keyword_arg_node(args_node, "xs", ctx.text.as_bytes())
                    .and_then(|kw| kw.child_by_field_name("value"))
                    .or_else(|| find_positional_arg_node(args_node, 2));

                let ys_shape = match (body_node, xs_node) {
                    (Some(body_node), Some(xs_node)) => {
                        scan_body_ys_shape(body_node, &init_shape, xs_node, rhs.start_byte(), ctx)
                    }
                    _ => None,
                };

                return Some(vec![Some(init_shape), ys_shape]);
            }

            // `values, indices = jax.lax.top_k(operand, k)` — both outputs
            // share `operand`'s shape with the last axis replaced by `k`.
            if matches!(known, Some(KnownFunction::LaxTopK)) {
                if n_targets != 2 {
                    return None;
                }
                let args_node = rhs.child_by_field_name("arguments")?;
                let args = extract_call_arguments(args_node, ctx.text).ok()?;
                let mut positionals = args.iter().filter_map(|a| match a {
                    CallArgument::Positional { value } => Some(value.as_str()),
                    _ => None,
                });
                let operand = positionals.next()?;
                let k = positionals.next().or_else(|| {
                    args.iter().find_map(|a| match a {
                        CallArgument::Keyword { name, value } if name == "k" => {
                            Some(value.as_str())
                        }
                        _ => None,
                    })
                })?;
                let mut out = ctx.resolve_shape(operand, scope_idx)?;
                let last = out.len().checked_sub(1)?;
                out[last] = k.to_string();
                return Some(vec![Some(out.clone()), Some(out)]);
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
                        | KnownFunction::Chunk
                        | KnownFunction::Unbind
                        | KnownFunction::KthValue
                        | KnownFunction::MedianDim
                        | KnownFunction::TorchSplit
                )
            ) {
                let args_node = rhs.child_by_field_name("arguments")?;
                let args = extract_call_arguments(args_node, ctx.text).ok()?;
                return tuple_multi_output_shapes(&known?, &args, args_node, n_targets, scope_idx, ctx);
            }

            if !matches!(known, Some(KnownFunction::Split)) {
                return None;
            }
            let args_node = rhs.child_by_field_name("arguments")?;
            let args = extract_call_arguments(args_node, ctx.text).ok()?;
            match compute_split_shapes(&args, &ctx.scope_shapes(scope_idx)) {
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
        KnownFunction::MedianDim => {
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
            if !has_dim {
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
    _text: &str,
    out: &mut Vec<Node<'a>>,
) -> Result<(), String> {
    for i in 0..node.named_child_count() {
        let child = node.named_child(i as u32).unwrap();
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
                collect_value_statements_recursive(child, _text, out)?;
            }
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
    scopes: &[FunctionShapeScope],
) -> Result<Option<Vec<String>>, String> {
    apply_vmap_call_chain(
        &[(info.in_axes, info.out_axes)],
        &info.wrapped,
        args,
        caller_shapes,
        scopes,
    )
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
    args: &[CallArgument],
    caller_shapes: &dyn ShapeLookup,
    scopes: &[FunctionShapeScope],
) -> Result<Option<Vec<String>>, String> {
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

    // Find the wrapped function's FunctionShapeScope.
    let callee = match find_callee_scope(wrapped, None, scopes) {
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
    let callable = callable_node.utf8_text(ctx.text.as_bytes()).ok()?.to_string();

    let outer_args = extract_call_arguments(outer_args_node, ctx.text).ok()?;

    // Case A: the vmapped callable is `self.<attr>` resolving to a known layer.
    if let Some(attr) = callable.strip_prefix("self.")
        && let Some(layer) = ctx
            .self_attr_layer_at(attr, outer_args_node.start_byte())
            .cloned()
    {
        return apply_inline_vmap_layer_chain(
            (&callable, &layer),
            &outer_args,
            &axes,
            scope_idx,
            outer_args_node.range(),
            ctx,
        );
    }

    // Case B: the callable is a bare identifier — a user function in this file.
    if !callable.contains('.') {
        return match apply_vmap_call_chain(
            &axes,
            &callable,
            &outer_args,
            &ctx.scope_shapes(scope_idx),
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
    let resolved = resolve_call_target(target, ctx.import_map);
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
fn shape_of_inline_layer(
    ctor_call: Node,
    outer_args_node: Node,
    scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let kind = classify_inline_constructor(ctor_call, ctx.text, ctx.import_map)?;
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

/// Search for a `FunctionShapeScope` whose `function_name` matches `target`,
/// excluding the module scope (index 0) and optionally excluding scopes
/// whose byte range contains `call_byte`.
///
/// - `Some(call_byte)`: exclude scopes whose byte range contains this byte
///   (used by user-function calls to avoid self-recursive binding).
/// - `None`: don't exclude any scope (used by vmap where the call is not
///   inside the wrapped function).
///
/// Returns the index of the best (smallest-scope) matching scope, or `None`.
fn find_callee_scope(
    target: &str,
    call_byte: Option<usize>,
    scopes: &[FunctionShapeScope],
) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None; // (scope index, scope size)
    for (i, scope) in scopes.iter().enumerate() {
        if i == 0 {
            continue; // skip module scope
        }
        if scope.function_name.as_deref() != Some(target) {
            continue;
        }
        // Don't bind a call to itself (recursive call in the same function body).
        if let Some(byte) = call_byte
            && scope.start_byte <= byte
            && byte < scope.end_byte
        {
            continue;
        }
        let size = scope.end_byte - scope.start_byte;
        match best {
            None => best = Some((i, size)),
            Some((_, prev_size)) if size < prev_size => best = Some((i, size)),
            _ => {} // keep first on tie
        }
    }
    best.map(|(i, _)| i)
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
    text: &str,
) -> Result<Option<Vec<String>>, String> {
    let binding = build_dim_binding(callee, target_name, positional_arg_shapes)?;
    let Some(return_name) = trace_bare_return_identifier(text, callee.start_byte, callee.end_byte)
    else {
        return Ok(None);
    };
    let Some(return_shape) = callee.shapes.get(&return_name) else {
        return Ok(None);
    };
    Ok(Some(substitute_dims(return_shape, &binding)))
}

/// Re-parse `text[start_byte..end_byte]` (a single function definition's own
/// byte range, from `FunctionShapeScope`) as a standalone snippet and find
/// its one `return <identifier>` statement, not descending into nested
/// function/class definitions (a `return` there belongs to a different
/// scope). Returns `None` for zero, multiple distinct, or non-identifier
/// return targets — deliberately conservative (v1).
fn trace_bare_return_identifier(text: &str, start_byte: usize, end_byte: usize) -> Option<String> {
    let snippet = text.get(start_byte..end_byte)?;
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_python::LANGUAGE.into()).ok()?;
    let tree = parser.parse(snippet, None)?;
    let func_node = tree.root_node().named_child(0)?;
    if func_node.kind() != "function_definition" {
        return None;
    }
    let body = func_node.child_by_field_name("body")?;
    let mut found: Option<String> = None;
    collect_bare_return_identifiers(body, snippet, &mut found)?;
    found
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
// The mechanism above (`apply_user_function` / `bind_and_substitute` /
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
/// more conservative `trace_bare_return_identifier` convention) — genuinely
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
/// Recursion/cycles: re-entering a scope already being specialized (a
/// direct or mutual recursive call) returns `None` immediately, and the
/// active-specialization stack is capped at `MAX_SPECIALIZATION_DEPTH`.
fn specialize_callee_call(
    callee_idx: usize,
    seeded_params: HashMap<String, Vec<String>>,
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
        specialized_shapes.entry(name).or_insert(shape);
    }

    let is_first = !ctx.specialized_scopes.contains(&callee_idx);
    let saved_shapes = std::mem::replace(&mut ctx.scopes[callee_idx].shapes, specialized_shapes);
    let errors_before = ctx.errors.len();
    let assignments_before = ctx.assignment_shapes.len();

    ctx.active_specializations.push(callee_idx);

    // Re-collect just this callee's own assignments (cheap — bounded by its
    // own body size, not the whole file) and replay them through the same
    // per-assignment logic the top-level pass uses. Items whose innermost
    // scope isn't exactly `callee_idx` belong to a nested closure defined
    // inside this body — that closure gets its own scope and is evaluated
    // lazily, on its own, if/when it is itself called.
    let body_assignments = collect_assignment_items(func_node, ctx.text).unwrap_or_default();
    for (lhs, rhs_node, assignment_node) in body_assignments {
        if scope_index_for_byte(ctx.scopes, rhs_node.start_byte()) == Some(callee_idx) {
            process_assignment_item(lhs, rhs_node, assignment_node, ctx);
        }
    }

    let ret = match func_node.child_by_field_name("body") {
        Some(body) => trace_specialized_return(body, callee_idx, ctx),
        None => SpecializedReturn::default(),
    };

    ctx.active_specializations.pop();

    if is_first {
        ctx.specialized_scopes.insert(callee_idx);
    } else {
        ctx.scopes[callee_idx].shapes = saved_shapes;
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
    let all_params = &ctx.scopes[callee_idx].all_params;
    let params: &[String] = match all_params.first().map(String::as_str) {
        Some("self") | Some("cls") => &all_params[1..],
        _ => all_params,
    };
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
) -> HashMap<String, Vec<String>> {
    let params: &[String] = match all_params.first().map(String::as_str) {
        Some("self") | Some("cls") => &all_params[1..],
        _ => all_params,
    };
    let mut seeded = HashMap::new();
    let mut positional_idx = 0usize;
    for arg in args {
        match arg {
            CallArgument::Positional { value } => {
                if let Some(param) = params.get(positional_idx)
                    && let Some(shape) = ctx.resolve_shape(value, caller_scope_idx)
                {
                    seeded.insert(param.clone(), shape);
                }
                positional_idx += 1;
            }
            CallArgument::Keyword { name, value } => {
                if params.iter().any(|p| p == name)
                    && let Some(shape) = ctx.resolve_shape(value, caller_scope_idx)
                {
                    seeded.insert(name.clone(), shape);
                }
            }
        }
    }
    seeded
}

/// Fallback for a same-file call whose callee (a top-level function, a
/// nested closure, or a `self.<method>`) has at least one un-annotated
/// parameter: seed the un-annotated params from this call site's resolved
/// argument shapes and evaluate the callee's body on demand. Tried only
/// after `apply_user_function`'s existing annotation/bare-return-trace path
/// has already run and found nothing — when every param is already
/// annotated, that path already covers the call (or correctly found
/// nothing to propagate), so this returns `None` up front rather than
/// re-doing the same work.
fn apply_seeded_user_function(
    target: &str,
    call_byte: usize,
    args: &[CallArgument],
    caller_scope_idx: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let callee_idx = find_callee_scope(target, Some(call_byte), ctx.scopes)?;
    let all_params = ctx.scopes[callee_idx].all_params.clone();
    let seeded = seed_params_from_call(&all_params, args, caller_scope_idx, ctx);
    let result = specialize_callee_call(callee_idx, seeded, ctx)?;
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
    xs_node: Node,
    scan_call_byte: usize,
    ctx: &mut ShapeCtx,
) -> Option<Vec<String>> {
    let body_name = body_node.utf8_text(ctx.text.as_bytes()).ok()?;
    if body_name.contains('.') {
        return None;
    }
    let xs_shape = shape_of_expression(xs_node, ctx)?;
    let (leading, rest) = xs_shape.split_first()?;
    let leading = leading.clone();
    let rest = rest.to_vec();

    let callee_idx = find_callee_scope(body_name, Some(scan_call_byte), ctx.scopes)?;
    let all_params = ctx.scopes[callee_idx].all_params.clone();
    let mut seeded = HashMap::new();
    if let Some(carry_param) = all_params.first() {
        seeded.insert(carry_param.clone(), init_shape.to_vec());
    }
    if let Some(elem_param) = all_params.get(1) {
        seeded.insert(elem_param.clone(), rest);
    }

    let result = specialize_callee_call(callee_idx, seeded, ctx)?;
    let mut elems = result.tuple?;
    if elems.len() != 2 {
        return None;
    }
    let y_shape = elems.remove(1)?;
    let mut out = vec![leading];
    out.extend(y_shape);
    Some(out)
}

/// Attempt to resolve `target` as a user-defined function in the same file
/// and propagate its declared (or traced) return shape to the call site.
///
/// Returns:
/// - `Some(Ok(Some(shape)))` if a matching function was found and its
///   return shape could be computed after binding param dims to arg dims —
///   either from a `-> ReturnType` annotation, or (when absent) traced from
///   a bare `return <name>` statement against the callee's own body shapes
///   (see `trace_user_function_return`).
/// - `Some(Ok(None))` if a matching function was found but no shape could
///   be produced (no annotation and no traceable bare-identifier return) —
///   argument validation still ran but nothing to propagate.
/// - `Some(Err(msg))` if argument shapes don't unify with declared param shapes.
/// - `None` if no matching user-defined function was found (fall through to
///   the known-function branch).
///
/// v1 limitations (documented in PR):
/// - Only positional arguments are matched. Keyword args that match a param
///   name are honoured; otherwise the call is skipped with Ok(None).
/// - No cross-file resolution (the tracing fallback is same-file only —
///   `apply_imported_user_function` passes `text: None`, disabling it).
/// - Qualified names ("module.func") are excluded at the call site.
/// - Fresh output dims (not in the binding) pass through unchanged.
fn apply_user_function(
    target: &str,
    call_byte: usize,
    args: &[CallArgument],
    caller_shapes: &dyn ShapeLookup,
    scopes: &[FunctionShapeScope],
    text: &str,
) -> Option<Result<Option<Vec<String>>, String>> {
    let scope_idx = find_callee_scope(target, Some(call_byte), scopes)?;
    bind_user_function_args(&scopes[scope_idx], target, args, caller_shapes, Some(text))
}

/// Bind a callee's declared parameter shapes to the caller's argument shapes
/// and substitute into its return shape. Shared by `apply_user_function`
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
    // `Some(source_text)` for the same-file path (enables the no-return-
    // annotation tracing fallback); `None` for the cross-file path
    // (`apply_imported_user_function`), where tracing isn't supported yet.
    text: Option<&str>,
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
    // callee has no `-> ReturnType` annotation and source text is available
    // (same-file path), the body-tracing fallback (see
    // `trace_user_function_return`).
    if callee.return_shape.is_some() {
        Some(bind_and_substitute(callee, target, &arg_shapes))
    } else if let Some(text) = text {
        Some(trace_user_function_return(callee, target, &arg_shapes, text))
    } else {
        Some(Ok(None))
    }
}

/// Cross-file counterpart of `apply_user_function`: resolve `target` through
/// the import map to a function defined in another file on disk, extract its
/// jaxtyping parameter + return shape annotations, and apply the same
/// bind-and-substitute logic. `None` means "not an imported function we can
/// resolve, or it has no jaxtyping annotations" — the caller falls through to
/// the known-function branch, same contract as `apply_user_function`.
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
        );
    }

    false
}

#[cfg(test)]
mod shape_of_expression_tests {
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

    fn analyze_simple(code: &str) -> LayerShapeAnalysis {
        let tree = parse(code);
        let import_map = build_import_map(tree.root_node(), code).unwrap();
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
        collect_function_definition_nodes(tree.root_node(), &mut function_nodes);
        let mut scope_function_nodes: Vec<Option<Node>> = std::iter::once(None)
            .chain(function_nodes.into_iter().map(Some))
            .collect();
        scope_function_nodes.resize(scopes.len(), None);
        let original_shapes: Vec<HashMap<String, Vec<String>>> =
            scopes.iter().map(|s| s.shapes.clone()).collect();

        let mut ctx = ShapeCtx {
            text: code,
            import_map: &import_map,
            layer_records: &layer_records,
            self_attr_layers: &self_attr_layers,
            aliases: &aliases,
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
            scope_function_nodes: &scope_function_nodes,
            active_specializations: Vec::new(),
            specialized_scopes: std::collections::HashSet::new(),
            original_shapes: &original_shapes,
        };

        for (lhs, rhs_node, _assignment_node) in assignments {
            let Lhs::Single(lhs_name) = lhs else {
                continue;
            };
            let scope_idx = match scope_index_for_byte(ctx.scopes, rhs_node.start_byte()) {
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
        // single token
        assert_eq!(normalize_dim("self.dt_rank", &aliases, 0), "dt_rank");
        // inside an expression, multiple tokens
        assert_eq!(
            normalize_dim("self.dt_rank + self.d_state", &aliases, 0),
            "dt_rank + d_state"
        );
        // unaliased self.attr is left untouched
        assert_eq!(normalize_dim("self.unknown", &aliases, 0), "self.unknown");
        // prefix collision: self.dt_rank must not match self.dt_rank2
        assert_eq!(normalize_dim("self.dt_rank2", &aliases, 0), "self.dt_rank2");
        // no self. → unchanged
        assert_eq!(normalize_dim("seq_length", &aliases, 0), "seq_length");
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

        assert_eq!(normalize_dim("self.rank", &aliases, 50), "dt_rank");
        assert_eq!(normalize_dim("self.rank", &aliases, 150), "other_rank");
        // Outside every class range and not a lone binding: left untouched.
        assert_eq!(normalize_dim("self.rank", &aliases, 250), "self.rank");
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
}
