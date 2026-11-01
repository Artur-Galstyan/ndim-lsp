use std::collections::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use tree_sitter::{InputEdit, Node, Parser, Point, Range, Tree};

mod names;

use names::{Dsu, NameSet, link_units, name_hash, vmap_names};

use crate::layers::{extract_layer_assignments_scoped, extract_self_attr_layers_by_class};
use crate::python_ast::{
    ClassHierarchy, ClassSummary, SelfAttrWrite, build_import_map, dim_symbols,
    extract_jaxtyping_shapes, extract_self_attr_writes,
};
use crate::resolution::{Deps, ResolutionCache, search_roots_fingerprint};
use crate::types::*;

fn point_at(text: &str, byte: usize) -> Point {
    let before = &text.as_bytes()[..byte];
    let row = before.iter().filter(|&&b| b == b'\n').count();
    let line_start = before
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |i| i + 1);
    Point::new(row, byte - line_start)
}

/// The one edit that turns `old` into `new`: the bytes between their common
/// prefix and common suffix.
pub fn diff_edit(old: &str, new: &str) -> InputEdit {
    let (o, n) = (old.as_bytes(), new.as_bytes());
    let prefix = o.iter().zip(n).take_while(|(a, b)| a == b).count();
    let suffix = o[prefix..]
        .iter()
        .rev()
        .zip(n[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let (old_end, new_end) = (o.len() - suffix, n.len() - suffix);
    InputEdit {
        start_byte: prefix,
        old_end_byte: old_end,
        new_end_byte: new_end,
        start_position: point_at(old, prefix),
        old_end_position: point_at(old, old_end),
        new_end_position: point_at(new, new_end),
    }
}

/// Parse `new_text` reusing `old_tree`, the tree of `old_text`. tree-sitter
/// recovers from a syntax error differently when it reuses a tree than when it
/// parses from scratch, so a tree with an error, old or new, is parsed again
/// from scratch: the result always equals a fresh parse.
pub fn reparse(
    parser: &mut Parser,
    old_text: &str,
    old_tree: &Tree,
    new_text: &str,
) -> Option<Tree> {
    if old_tree.root_node().has_error() {
        return parser.parse(new_text, None);
    }
    let mut tree = old_tree.clone();
    tree.edit(&diff_edit(old_text, new_text));
    let new = parser.parse(new_text, Some(&tree))?;
    if new.root_node().has_error() {
        return parser.parse(new_text, None);
    }
    Some(new)
}

/// Description of the first node where two trees differ in kind, byte range,
/// or error flags, or `None` if they are equal.
pub fn first_tree_difference(a: &Tree, b: &Tree) -> Option<String> {
    let (a, b) = (flatten(a), flatten(b));
    let i = (0..a.len().max(b.len())).find(|&i| a.get(i) != b.get(i))?;
    Some(format!("item {i}: {:?} vs {:?}", a.get(i), b.get(i)))
}

type Item = (usize, &'static str, usize, usize, bool, bool, bool);

fn flatten(tree: &Tree) -> Vec<Item> {
    let mut items = Vec::new();
    let mut cursor = tree.walk();
    let mut depth = 0;
    loop {
        let n = cursor.node();
        items.push((
            depth,
            n.kind(),
            n.start_byte(),
            n.end_byte(),
            n.is_error(),
            n.is_missing(),
            n.is_extra(),
        ));
        if cursor.goto_first_child() {
            depth += 1;
            continue;
        }
        while !cursor.goto_next_sibling() {
            if !cursor.goto_parent() {
                return items;
            }
            depth -= 1;
        }
    }
}

/// A shift of positions: bytes and rows. Columns never move, because a
/// record is reused only for a unit that starts in the same column. The shift
/// is one amount for every position, or the amount of the unit that holds
/// each position (`Remap`).
#[derive(Clone, Copy)]
pub struct Delta<'a> {
    pub bytes: isize,
    pub rows: isize,
    table: Option<&'a Remap>,
}

impl<'a> Delta<'a> {
    pub const fn new(bytes: isize, rows: isize) -> Self {
        Self {
            bytes,
            rows,
            table: None,
        }
    }

    fn by_unit(table: &'a Remap) -> Self {
        Self {
            bytes: 0,
            rows: 0,
            table: Some(table),
        }
    }

    /// The shift that moves a unit-relative position to its place in a file
    /// where the unit starts at `start_byte` on `start_row`.
    fn from_relative(start_byte: usize, start_row: usize) -> Self {
        Self::new(start_byte as isize, start_row as isize)
    }

    fn inverse(self) -> Self {
        assert!(self.table.is_none(), "a remap has no inverse");
        Self::new(-self.bytes, -self.rows)
    }

    fn byte(self, at: &mut usize) {
        if let Some(table) = self.table {
            table.byte(at);
            return;
        }
        *at = at
            .checked_add_signed(self.bytes)
            .expect("invariant: a shifted position stays inside the file");
    }

    fn row(self, row: &mut usize) {
        if let Some(table) = self.table {
            table.row(row);
            return;
        }
        *row = row
            .checked_add_signed(self.rows)
            .expect("invariant: a shifted row stays inside the file");
    }

    /// The range of the class around a binding. `(0, usize::MAX)` means no
    /// class, and stays as it is.
    fn class(self, start: &mut usize, end: &mut usize) {
        if (*start, *end) != (0, usize::MAX) {
            self.byte(start);
            self.byte(end);
        }
    }
}

/// Where each unit of an old file went in the new file. A position moves by
/// the amount of the unit that holds it, which is not the same for the units
/// of one record: an edit between them moves one and not the other.
pub(crate) struct Remap {
    units: Vec<Moved>,
    /// A position fell where no unit that has a place in the new file is.
    bad: std::cell::Cell<bool>,
}

#[derive(Clone, Copy)]
struct Moved {
    start: usize,
    end: usize,
    first_row: usize,
    last_row: usize,
    bytes: isize,
    rows: isize,
}

impl Remap {
    /// Only units that have a place in the new file can hold a position of a
    /// record that is moved. `None` when two of them share a byte or a row:
    /// the holder of a position is then not known.
    fn new(old: &[Span], new_of_old: &[Option<usize>], new: &[Span]) -> Option<Self> {
        let mut units = Vec::new();
        for (o, span) in old.iter().enumerate().filter(|(_, s)| s.real) {
            let Some(n) = new_of_old[o] else {
                continue;
            };
            units.push(Moved {
                start: span.start,
                end: span.end,
                first_row: span.row,
                last_row: span.end_row,
                bytes: new[n].start as isize - span.start as isize,
                rows: new[n].row as isize - span.row as isize,
            });
        }
        let crowded = units
            .windows(2)
            .any(|w| w[0].end >= w[1].start || w[0].last_row >= w[1].first_row);
        (!crowded).then_some(Self {
            units,
            bad: std::cell::Cell::new(false),
        })
    }

    /// Positions stay where they are. Checks that each one has a unit.
    fn identity(spans: &[Span]) -> Option<Self> {
        let placed: Vec<Option<usize>> = (0..spans.len()).map(Some).collect();
        Self::new(spans, &placed, spans)
    }

    fn unit_of_byte(&self, at: usize) -> Option<&Moved> {
        let i = self.units.partition_point(|u| u.start <= at);
        let unit = &self.units[i.checked_sub(1)?];
        (at <= unit.end).then_some(unit)
    }

    fn unit_of_row(&self, row: usize) -> Option<&Moved> {
        let i = self.units.partition_point(|u| u.first_row <= row);
        let unit = &self.units[i.checked_sub(1)?];
        (row <= unit.last_row).then_some(unit)
    }

    fn byte(&self, at: &mut usize) {
        match self.unit_of_byte(*at) {
            Some(unit) => {
                *at = at
                    .checked_add_signed(unit.bytes)
                    .expect("invariant: a shifted position stays inside the file");
            }
            None => self.bad.set(true),
        }
    }

    fn row(&self, row: &mut usize) {
        match self.unit_of_row(*row) {
            Some(unit) => {
                *row = row
                    .checked_add_signed(unit.rows)
                    .expect("invariant: a shifted row stays inside the file");
            }
            None => self.bad.set(true),
        }
    }

    fn is_bad(&self) -> bool {
        self.bad.get()
    }
}

/// A value that holds positions of a file. Each implementation destructures
/// its type without `..`, so a new field forces a decision.
pub trait Shift {
    fn shift(&mut self, by: Delta<'_>);
}

impl<T: Shift> Shift for Vec<T> {
    fn shift(&mut self, by: Delta<'_>) {
        self.iter_mut().for_each(|item| item.shift(by));
    }
}

impl<T: Shift> Shift for Option<T> {
    fn shift(&mut self, by: Delta<'_>) {
        if let Some(item) = self {
            item.shift(by);
        }
    }
}

impl Shift for Point {
    fn shift(&mut self, by: Delta<'_>) {
        let Point { row, column: _ } = self;
        by.row(row);
    }
}

impl Shift for Range {
    fn shift(&mut self, by: Delta<'_>) {
        let Range {
            start_byte,
            end_byte,
            start_point,
            end_point,
        } = self;
        by.byte(start_byte);
        by.byte(end_byte);
        start_point.shift(by);
        end_point.shift(by);
    }
}

impl Shift for ShapeDimensionSite {
    fn shift(&mut self, by: Delta<'_>) {
        let ShapeDimensionSite {
            binding: _,
            axis: _,
            value: _,
            range,
        } = self;
        range.shift(by);
    }
}

impl Shift for FunctionShapeScope {
    fn shift(&mut self, by: Delta<'_>) {
        let FunctionShapeScope {
            function_name: _,
            start_byte,
            end_byte,
            shapes: _,
            return_shape: _,
            param_order: _,
            all_params: _,
            dimension_sites,
        } = self;
        by.byte(start_byte);
        by.byte(end_byte);
        dimension_sites.shift(by);
    }
}

impl Shift for LayerAssignment {
    fn shift(&mut self, by: Delta<'_>) {
        let LayerAssignment {
            name: _,
            kind: _,
            byte_position,
        } = self;
        by.byte(byte_position);
    }
}

impl Shift for ScopedSelfAttrLayer {
    fn shift(&mut self, by: Delta<'_>) {
        let ScopedSelfAttrLayer {
            class_start,
            class_end,
            kind: _,
        } = self;
        by.class(class_start, class_end);
    }
}

impl Shift for ScopedSelfAttrAlias {
    fn shift(&mut self, by: Delta<'_>) {
        let ScopedSelfAttrAlias {
            class_start,
            class_end,
            value: _,
        } = self;
        by.class(class_start, class_end);
    }
}

impl Shift for AssignmentShape {
    fn shift(&mut self, by: Delta<'_>) {
        let AssignmentShape {
            line,
            name: _,
            shape: _,
        } = self;
        let mut row = *line as usize;
        by.row(&mut row);
        *line = row as u32;
    }
}

impl Shift for ShapeFix {
    fn shift(&mut self, by: Delta<'_>) {
        match self {
            ShapeFix::AppendTranspose {
                expression_range,
                operand_range,
            } => {
                expression_range.shift(by);
                operand_range.shift(by);
            }
        }
    }
}

impl Shift for ShapeError {
    fn shift(&mut self, by: Delta<'_>) {
        let ShapeError {
            variable: _,
            message: _,
            range,
            kind: _,
            related,
            fix,
        } = self;
        range.shift(by);
        if let Some((related_range, _)) = related {
            related_range.shift(by);
        }
        fix.shift(by);
    }
}

impl Shift for LayerApplication {
    fn shift(&mut self, by: Delta<'_>) {
        let LayerApplication {
            variable: _,
            layer: _,
            input: _,
            extra_inputs: _,
            kind: _,
            range,
        } = self;
        range.shift(by);
    }
}

/// A node of the tree, named by what finds it again: the smallest node over
/// a byte range, and the kind that picks one of several with that range.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NodeRef {
    kind: &'static str,
    start_byte: usize,
    end_byte: usize,
}

impl NodeRef {
    pub(crate) fn of(node: Node) -> Self {
        Self {
            kind: node.kind(),
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        }
    }

    /// The node under `within` that this names, if the tree still has it.
    pub(crate) fn find<'t>(&self, within: Node<'t>) -> Option<Node<'t>> {
        let mut node = within.descendant_for_byte_range(self.start_byte, self.end_byte)?;
        while node.kind() != self.kind || node.byte_range() != (self.start_byte..self.end_byte) {
            node = node
                .parent()
                .filter(|p| p.byte_range() == (self.start_byte..self.end_byte))?;
        }
        Some(node)
    }
}

impl Shift for NodeRef {
    fn shift(&mut self, by: Delta<'_>) {
        let NodeRef {
            kind: _,
            start_byte,
            end_byte,
        } = self;
        by.byte(start_byte);
        by.byte(end_byte);
    }
}

/// A `SelfAttrWrite` that holds no node, so it can outlive its tree.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WriteRecord {
    attr: String,
    class: (usize, usize),
    value: Option<NodeRef>,
}

impl WriteRecord {
    fn of(write: &SelfAttrWrite) -> Self {
        Self {
            attr: write.attr.clone(),
            class: write.class,
            value: write.value.map(NodeRef::of),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        attr: &str,
        class: (usize, usize),
        value: Option<(usize, usize)>,
    ) -> Self {
        Self {
            attr: attr.to_string(),
            class,
            value: value.map(|(start_byte, end_byte)| NodeRef {
                kind: "identifier",
                start_byte,
                end_byte,
            }),
        }
    }

    fn resolve<'t>(&self, unit: Node<'t>) -> Option<SelfAttrWrite<'t>> {
        let value = match &self.value {
            Some(node_ref) => Some(node_ref.find(unit)?),
            None => None,
        };
        Some(SelfAttrWrite {
            attr: self.attr.clone(),
            class: self.class,
            value,
        })
    }
}

impl Shift for WriteRecord {
    fn shift(&mut self, by: Delta<'_>) {
        let WriteRecord {
            attr: _,
            class,
            value,
        } = self;
        by.class(&mut class.0, &mut class.1);
        value.shift(by);
    }
}

impl Shift for ClassSummary {
    fn shift(&mut self, by: Delta<'_>) {
        let ClassSummary {
            name: _,
            range,
            bases: _,
            fields: _,
        } = self;
        by.class(&mut range.0, &mut range.1);
    }
}

/// Largest text a memo is kept for, and the most units in it. A larger file
/// is analyzed without one.
const MAX_MEMO_TEXT: usize = 4 << 20;
const MAX_MEMO_UNITS: usize = 100_000;

/// What the pre-passes found in a file, before the propagation pass.
pub(crate) struct PrePass<'t> {
    pub(crate) scopes: Vec<FunctionShapeScope>,
    pub(crate) import_map: HashMap<String, ImportPath>,
    pub(crate) layer_records: Vec<LayerAssignment>,
    pub(crate) self_attr_layers: HashMap<String, Vec<ScopedSelfAttrLayer>>,
    pub(crate) writes: Vec<SelfAttrWrite<'t>>,
    pub(crate) hierarchy: ClassHierarchy,
    /// What the propagation pass can take from the last run. `None` for a
    /// pass over the whole tree.
    pub(crate) plan: Option<PropPlan>,
}

impl<'t> PrePass<'t> {
    /// The pre-passes over the whole tree.
    pub(crate) fn whole_tree<F>(
        root: Node<'t>,
        text: &str,
        search_roots: &[PathBuf],
        read_file: &F,
        max_depth: usize,
        cache: Option<&ResolutionCache>,
    ) -> Result<Self, String>
    where
        F: Fn(&PathBuf) -> Option<String>,
    {
        let scopes = extract_jaxtyping_shapes(root, text)?;
        let import_map = build_import_map(root, text)?;
        let layer_records = extract_layer_assignments_scoped(
            root,
            text,
            &import_map,
            search_roots,
            read_file,
            max_depth,
            cache,
        )?;
        let self_attr_layers = extract_self_attr_layers_by_class(
            root,
            text,
            &import_map,
            search_roots,
            read_file,
            max_depth,
            cache,
        )?;
        Ok(Self {
            scopes,
            import_map,
            layer_records,
            self_attr_layers,
            writes: extract_self_attr_writes(root, text),
            hierarchy: ClassHierarchy::new(root, text),
            plan: None,
        })
    }
}

/// How much of the last run the memo reused.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoStats {
    pub units: usize,
    /// Units whose syntax record was reused.
    pub syntax_hits: usize,
    /// Units whose layer record was reused.
    pub layer_hits: usize,
    /// Units whose propagation output was replayed instead of computed.
    pub replayed_units: usize,
    /// The propagation ran over every unit: nothing could be replayed.
    pub propagate_full: bool,
}

/// Where one top-level statement sits in the text.
#[derive(Debug, Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
    row: usize,
    end_row: usize,
    col: usize,
    /// The unit has a syntax error. Its parse may depend on the text after
    /// it, so it is never reused.
    error: bool,
    /// A statement, not a comment or a `;`. Only these hold the positions of
    /// an analysis.
    real: bool,
}

impl Span {
    fn of(node: Node) -> Self {
        let start = node.start_position();
        Self {
            start: node.start_byte(),
            end: node.end_byte(),
            row: start.row,
            end_row: node.end_position().row,
            col: start.column,
            error: node.has_error(),
            real: node.is_named() && node.kind() != "comment",
        }
    }

    fn to_new_file(self) -> Delta<'static> {
        Delta::from_relative(self.start, self.row)
    }
}

/// What depends on the text of one unit only, in unit-relative positions.
struct SyntaxRecord {
    /// The first scope is the unit's own copy of the module scope.
    scopes: Vec<FunctionShapeScope>,
    imports: HashMap<String, ImportPath>,
    writes: Vec<WriteRecord>,
    classes: Vec<ClassSummary>,
    names: NameSet,
    /// Hashes of the names the annotations of the unit use as dims.
    symbols: Vec<u64>,
}

impl SyntaxRecord {
    fn extract(unit: Node, text: &str, span: Span) -> Result<Self, String> {
        let scopes = extract_jaxtyping_shapes(unit, text)?;
        let mut symbols: Vec<u64> = scopes
            .iter()
            .flat_map(|s| s.shapes.values().flatten().chain(s.return_shape.iter().flatten()))
            .flat_map(|dim| dim_symbols(dim))
            .map(|symbol| name_hash(symbol.as_bytes()))
            .collect();
        symbols.sort_unstable();
        symbols.dedup();
        let mut record = Self {
            scopes,
            imports: build_import_map(unit, text)?,
            writes: extract_self_attr_writes(unit, text)
                .iter()
                .map(WriteRecord::of)
                .collect(),
            classes: ClassSummary::collect(unit, text),
            names: NameSet::of(unit, text),
            symbols,
        };
        record.shift(span.to_new_file().inverse());
        Ok(record)
    }
}

impl Shift for SyntaxRecord {
    fn shift(&mut self, by: Delta<'_>) {
        let SyntaxRecord {
            scopes,
            imports: _,
            writes,
            classes,
            names: _,
            symbols: _,
        } = self;
        scopes.shift(by);
        writes.shift(by);
        classes.shift(by);
    }
}

/// What depends on the text of one unit, the import map, and the files on
/// disk the unit resolved names to.
struct LayerRecord {
    import_digest: u64,
    assignments: Vec<LayerAssignment>,
    self_attrs: HashMap<String, Vec<ScopedSelfAttrLayer>>,
    /// `None` when the disk state the unit looked at is not tracked: the
    /// record serves this run only.
    deps: Option<Deps>,
}

impl Shift for LayerRecord {
    fn shift(&mut self, by: Delta<'_>) {
        let LayerRecord {
            import_digest: _,
            assignments,
            self_attrs,
            deps: _,
        } = self;
        assignments.shift(by);
        self_attrs.values_mut().for_each(|v| v.shift(by));
    }
}

#[derive(Clone)]
struct Unit {
    span: Span,
    hash: u64,
    syntax: Arc<SyntaxRecord>,
    layers: Option<Arc<LayerRecord>>,
    /// What the propagation pass made of the unit, in the positions of the
    /// text of the memo.
    prop: Option<Arc<PropRecord>>,
    /// The first unit of the group of units that share names with this one.
    comp: u32,
}

/// What a memo is valid for. Another value drops the whole memo.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MemoKey {
    cache: u64,
    search_roots: u64,
    max_depth: usize,
}

/// The pre-pass results of the top-level statements ("units") of one file,
/// to reuse for the units of the next version that have the same text.
#[derive(Clone, Default)]
pub struct UnitMemo {
    key: Option<MemoKey>,
    text: Arc<str>,
    units: Vec<Unit>,
    stats: MemoStats,
    /// What the propagation records depend on besides their own unit.
    env: u64,
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn import_digest(import_map: &HashMap<String, ImportPath>) -> u64 {
    let mut entries: Vec<_> = import_map.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut hasher = DefaultHasher::new();
    for (alias, ImportPath { dots, module, name }) in entries {
        (alias, dots, module, name).hash(&mut hasher);
    }
    hasher.finish()
}

pub(crate) fn unit_nodes(root: Node) -> Vec<Node> {
    let mut cursor = root.walk();
    root.children(&mut cursor).collect()
}

impl UnitMemo {
    /// What the last run reused.
    pub fn stats(&self) -> MemoStats {
        self.stats
    }

    /// Whether the memo holds nothing to reuse.
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    fn same_text(&self, old: &Unit, new_text: &str, new: &Span) -> bool {
        !old.span.error
            && !new.error
            && old.span.col == new.col
            && self.text.as_bytes()[old.span.start..old.span.end]
                == new_text.as_bytes()[new.start..new.end]
    }

    /// For each new unit, the old unit with the same text and start column.
    /// The units before the first change and after the last match by place,
    /// the rest by text. A unit that moved, or that appears twice, matches
    /// too.
    fn match_units(
        &self,
        text: &str,
        spans: &[Span],
        hashes: &mut [Option<u64>],
    ) -> Vec<Option<usize>> {
        let old = &self.units;
        let mut found = vec![None; spans.len()];
        let common = old.len().min(spans.len());
        let prefix = (0..common)
            .take_while(|&i| self.same_text(&old[i], text, &spans[i]))
            .count();
        let suffix = (0..common - prefix)
            .take_while(|&j| {
                self.same_text(&old[old.len() - 1 - j], text, &spans[spans.len() - 1 - j])
            })
            .count();
        for (i, slot) in found.iter_mut().enumerate().take(prefix) {
            *slot = Some(i);
        }
        for j in 0..suffix {
            found[spans.len() - 1 - j] = Some(old.len() - 1 - j);
        }
        if prefix + suffix >= spans.len() {
            return found;
        }
        let mut by_text: HashMap<(u64, usize), Vec<usize>> = HashMap::new();
        for (i, unit) in old.iter().enumerate() {
            if !unit.span.error {
                by_text
                    .entry((unit.hash, unit.span.col))
                    .or_default()
                    .push(i);
            }
        }
        for i in prefix..spans.len() - suffix {
            let span = &spans[i];
            if span.error {
                continue;
            }
            let hash = *hashes[i]
                .get_or_insert_with(|| hash_bytes(&text.as_bytes()[span.start..span.end]));
            found[i] = by_text
                .get(&(hash, span.col))
                .and_then(|candidates| {
                    candidates
                        .iter()
                        .find(|&&o| self.same_text(&old[o], text, span))
                })
                .copied();
        }
        found
    }

    /// The pre-passes over `root`, from the units of this memo that did not
    /// change, and the memo for the next run. Any failure is an `Err`: run
    /// `PrePass::whole_tree`, which gives the real result or the real error.
    pub(crate) fn prepass<'t, F>(
        &self,
        root: Node<'t>,
        text: &str,
        search_roots: &[PathBuf],
        read_file: &F,
        max_depth: usize,
        cache: &ResolutionCache,
    ) -> Result<(PrePass<'t>, UnitMemo), String>
    where
        F: Fn(&PathBuf) -> Option<String>,
    {
        let key = MemoKey {
            cache: cache.id(),
            search_roots: search_roots_fingerprint(search_roots),
            max_depth,
        };
        let old = if self.key.as_ref() == Some(&key) {
            self
        } else {
            &UnitMemo::default()
        };
        let nodes = unit_nodes(root);
        if text.len() > MAX_MEMO_TEXT || nodes.len() > MAX_MEMO_UNITS {
            return Err("file too large for a memo".to_string());
        }
        let spans: Vec<Span> = nodes.iter().map(|n| Span::of(*n)).collect();
        let mut hashes = vec![None; spans.len()];
        let found = old.match_units(text, &spans, &mut hashes);

        let mut units: Vec<Unit> = Vec::with_capacity(spans.len());
        let mut stats = MemoStats {
            units: spans.len(),
            ..MemoStats::default()
        };
        for (i, span) in spans.iter().enumerate() {
            let unit = match found[i] {
                Some(o) => {
                    stats.syntax_hits += 1;
                    let reused = &old.units[o];
                    Unit {
                        span: *span,
                        hash: reused.hash,
                        syntax: reused.syntax.clone(),
                        layers: reused.layers.clone(),
                        prop: reused.prop.clone(),
                        comp: 0,
                    }
                }
                None => Unit {
                    span: *span,
                    hash: *hashes[i]
                        .get_or_insert_with(|| hash_bytes(&text.as_bytes()[span.start..span.end])),
                    syntax: Arc::new(SyntaxRecord::extract(nodes[i], text, *span)?),
                    layers: None,
                    prop: None,
                    comp: 0,
                },
            };
            units.push(unit);
        }

        let mut scopes = vec![FunctionShapeScope {
            function_name: None,
            start_byte: root.start_byte(),
            end_byte: root.end_byte(),
            shapes: HashMap::new(),
            return_shape: None,
            param_order: Vec::new(),
            all_params: Vec::new(),
            dimension_sites: Vec::new(),
        }];
        let mut import_map = HashMap::new();
        let mut writes = Vec::new();
        let mut classes = Vec::new();
        for (unit, node) in units.iter().zip(&nodes) {
            let by = unit.span.to_new_file();
            let record = &*unit.syntax;
            let (module, functions) = record
                .scopes
                .split_first()
                .ok_or("a unit has no module scope")?;
            scopes[0].shapes.extend(module.shapes.clone());
            for mut site in module.dimension_sites.clone() {
                site.shift(by);
                scopes[0].dimension_sites.push(site);
            }
            for mut scope in functions.iter().cloned() {
                scope.shift(by);
                scopes.push(scope);
            }
            import_map.extend(record.imports.clone());
            for write in &record.writes {
                let mut write = write.clone();
                write.shift(by);
                writes.push(write.resolve(*node).ok_or("a write lost its value node")?);
            }
            for class in &record.classes {
                let mut class = class.clone();
                class.shift(by);
                classes.push(class);
            }
        }

        let digest = import_digest(&import_map);
        let mut layer_records = Vec::new();
        let mut self_attr_layers: HashMap<String, Vec<ScopedSelfAttrLayer>> = HashMap::new();
        let mut layer_hit = Vec::with_capacity(units.len());
        for (unit, node) in units.iter_mut().zip(&nodes) {
            let valid = unit.layers.as_ref().is_some_and(|rec| {
                rec.import_digest == digest
                    && rec.deps.as_ref().is_some_and(|d| cache.deps_valid(d))
            });
            layer_hit.push(valid);
            if valid {
                stats.layer_hits += 1;
            } else {
                let (found, deps) = cache.record_deps(|| {
                    Ok::<_, String>((
                        extract_layer_assignments_scoped(
                            *node,
                            text,
                            &import_map,
                            search_roots,
                            read_file,
                            max_depth,
                            Some(cache),
                        )?,
                        extract_self_attr_layers_by_class(
                            *node,
                            text,
                            &import_map,
                            search_roots,
                            read_file,
                            max_depth,
                            Some(cache),
                        )?,
                    ))
                });
                let (assignments, self_attrs) = found?;
                let mut record = LayerRecord {
                    import_digest: digest,
                    assignments,
                    self_attrs,
                    deps,
                };
                record.shift(unit.span.to_new_file().inverse());
                unit.layers = Some(Arc::new(record));
            }
            let by = unit.span.to_new_file();
            let record = unit.layers.as_ref().expect("invariant: set above");
            for assignment in &record.assignments {
                let mut assignment = assignment.clone();
                assignment.shift(by);
                layer_records.push(assignment);
            }
            for (attr, entries) in &record.self_attrs {
                let merged = self_attr_layers.entry(attr.clone()).or_default();
                for entry in entries {
                    let mut entry = entry.clone();
                    entry.shift(by);
                    merged.push(entry);
                }
            }
        }

        let mut env = 0;
        let plan = if propagate_memo_enabled() {
            let built = PropPlan::build(old, &mut units, &found, &layer_hit, &import_map, cache);
            stats.replayed_units = built.plan.replay.iter().flatten().count();
            stats.propagate_full = stats.replayed_units == 0;
            env = built.env;
            Some(built.plan)
        } else {
            None
        };

        let next = UnitMemo {
            key: Some(key),
            text: Arc::from(text),
            units,
            stats,
            env,
        };
        let pre = PrePass {
            scopes,
            import_map,
            layer_records,
            self_attr_layers,
            writes,
            hierarchy: ClassHierarchy::from_summaries(&classes),
            plan,
        };
        Ok((pre, next))
    }
}

/// Whether the propagation pass reuses the records of the last run.
fn propagate_memo_enabled() -> bool {
    std::env::var_os("NDIM_NO_PROPAGATE_MEMO").is_none()
}

/// The output of the propagation pass for the statements of one phase.
#[derive(Clone, Default)]
pub(crate) struct Segment {
    pub(crate) errors: Vec<ShapeError>,
    pub(crate) applications: Vec<LayerApplication>,
    pub(crate) assignments: Arc<Vec<AssignmentShape>>,
}

impl Segment {
    fn moved(&self, by: Delta<'_>, rows: bool) -> Self {
        let mut moved = Self {
            errors: self.errors.clone(),
            applications: self.applications.clone(),
            assignments: self.assignments.clone(),
        };
        moved.errors.shift(by);
        moved.applications.shift(by);
        if rows {
            let mut assignments = (*self.assignments).clone();
            assignments.shift(by);
            moved.assignments = Arc::new(assignments);
        }
        moved
    }

    /// Checks that every position has a unit, without moving any.
    fn check(&mut self, by: Delta<'_>) {
        self.errors.shift(by);
        self.applications.shift(by);
        if let Some(assignments) = Arc::get_mut(&mut self.assignments) {
            assignments.shift(by);
        }
    }
}

/// What the propagation pass made of one unit: its output in two phases
/// (assignments, then value statements), the shapes it left in the scopes it
/// owns, and the files on disk it looked at.
pub(crate) struct PropRecord {
    phases: [Segment; 2],
    /// The final shapes of the function scopes of the unit.
    scopes: Arc<Vec<HashMap<String, Vec<String>>>>,
    /// The final shape, if any, of each module name the unit assigns.
    module: Arc<Vec<(String, Option<Vec<String>>)>>,
    deps: [Option<Deps>; 2],
}

impl PropRecord {
    fn usable(&self, cache: &ResolutionCache) -> bool {
        self.deps
            .iter()
            .all(|d| d.as_ref().is_some_and(|d| cache.deps_valid(d)))
    }

    fn moved(&self, by: Delta<'_>, rows: bool) -> Self {
        Self {
            phases: [self.phases[0].moved(by, rows), self.phases[1].moved(by, rows)],
            scopes: self.scopes.clone(),
            module: self.module.clone(),
            deps: self.deps.clone(),
        }
    }
}

/// What the propagation pass made of one unit it ran.
pub(crate) struct UnitRun {
    pub(crate) phases: [Segment; 2],
    pub(crate) deps: [Option<Deps>; 2],
}

/// Output of `propagate_calls`: layer applications, shape errors, and the
/// per-assignment shape records used for inlay hints.
pub(crate) type PropagateOutput = (Vec<LayerApplication>, Vec<ShapeError>, Vec<AssignmentShape>);

/// Which units the propagation pass has to run, and the records of the rest.
pub(crate) struct PropPlan {
    pub(crate) dirty: Vec<bool>,
    spans: Vec<Span>,
    /// The index in the scopes of the first function scope of each unit.
    first_scope: Vec<usize>,
    syntax: Vec<Arc<SyntaxRecord>>,
    /// For each unit that is not dirty, its record in the new positions.
    replay: Vec<Option<Arc<PropRecord>>>,
}

struct BuiltPlan {
    plan: PropPlan,
    env: u64,
}

/// Dirty bytes above this share of the file run everything: the replay would
/// save too little.
const MAX_DIRTY_SHARE: f64 = 0.6;

impl PropPlan {
    fn run_all(units: &[Unit]) -> Self {
        let mut first_scope = vec![1];
        for unit in units {
            first_scope.push(first_scope.last().unwrap() + unit.syntax.scopes.len() - 1);
        }
        Self {
            dirty: vec![true; units.len()],
            spans: units.iter().map(|u| u.span).collect(),
            first_scope,
            syntax: units.iter().map(|u| u.syntax.clone()).collect(),
            replay: vec![None; units.len()],
        }
    }

    fn build(
        old: &UnitMemo,
        units: &mut [Unit],
        found: &[Option<usize>],
        layer_hit: &[bool],
        import_map: &HashMap<String, ImportPath>,
        cache: &ResolutionCache,
    ) -> BuiltPlan {
        let n = units.len();
        let m = old.units.len();
        let sets: Vec<&NameSet> = units.iter().map(|u| &u.syntax.names).collect();
        let mut globals: Vec<u64> = sets.iter().flat_map(|s| s.globals.iter().copied()).collect();
        globals.sort_unstable();
        globals.dedup();
        // A name that a plain assignment gives an integer and an annotation
        // uses as a dim is not read as the integer (`integer_constant`).
        let ints: HashSet<u64> = sets.iter().flat_map(|s| s.ints.iter().copied()).collect();
        let mut clashes: Vec<u64> = units
            .iter()
            .flat_map(|u| u.syntax.symbols.iter().copied())
            .filter(|symbol| ints.contains(symbol))
            .collect();
        clashes.sort_unstable();
        clashes.dedup();
        let env = {
            let mut hasher = DefaultHasher::new();
            (
                import_digest(import_map),
                &globals,
                sets.iter().any(|s| s.wildcard),
                &clashes,
            )
                .hash(&mut hasher);
            hasher.finish()
        };

        // The groups of units that share names, in the new text.
        let mut dsu = Dsu::new(n + m);
        link_units(&sets, &vmap_names(import_map), 0, &mut dsu);
        let comp: Vec<u32> = (0..n).map(|i| dsu.find(i) as u32).collect();
        drop(sets);
        for (unit, comp) in units.iter_mut().zip(&comp) {
            unit.comp = *comp;
        }

        let everything = |units: &[Unit]| BuiltPlan {
            plan: Self::run_all(units),
            env,
        };
        let usable_memo =
            m > 0 && old.env == env && old.units.iter().all(|u| u.prop.is_some());
        if !usable_memo {
            return everything(units);
        }

        // Each old unit has one place in the new text, or none.
        let mut new_of_old: Vec<Option<usize>> = vec![None; m];
        let mut twice = vec![false; m];
        for (i, o) in found.iter().enumerate() {
            if let Some(o) = *o {
                twice[o] |= new_of_old[o].is_some();
                new_of_old[o].get_or_insert(i);
            }
        }
        for (slot, twice) in new_of_old.iter_mut().zip(&twice) {
            if *twice {
                *slot = None;
            }
        }
        let old_spans: Vec<Span> = old.units.iter().map(|u| u.span).collect();
        let new_spans: Vec<Span> = units.iter().map(|u| u.span).collect();
        let Some(remap) = Remap::new(&old_spans, &new_of_old, &new_spans) else {
            return everything(units);
        };

        // The groups of the old text join the groups of the new text.
        for (o, old_unit) in old.units.iter().enumerate() {
            let at = |u: usize| new_of_old[u].unwrap_or(n + u);
            dsu.union(at(o), at(old_unit.comp as usize));
        }
        let group: Vec<usize> = (0..n).map(|i| dsu.find(i)).collect();
        let mut dirty_group = vec![false; n + m];
        for i in 0..n {
            let reusable = found[i].is_some_and(|o| {
                new_of_old[o] == Some(i)
                    && layer_hit[i]
                    && !new_spans[i].error
                    && old.units[o].prop.as_ref().is_some_and(|p| p.usable(cache))
            });
            if !reusable {
                dirty_group[group[i]] = true;
            }
        }
        for o in (0..m).filter(|&o| new_of_old[o].is_none()) {
            dirty_group[dsu.find(n + o)] = true;
        }
        // A group is replayed only if its units come in the order they ran.
        let mut last_old: Vec<isize> = vec![-1; n + m];
        for (i, o) in found.iter().enumerate() {
            if let Some(o) = *o {
                if o as isize <= last_old[group[i]] {
                    dirty_group[group[i]] = true;
                }
                last_old[group[i]] = o as isize;
            }
        }
        let dirty: Vec<bool> = group.iter().map(|&g| dirty_group[g]).collect();
        let bytes_of = |want: bool| -> usize {
            (0..n)
                .filter(|&i| dirty[i] == want)
                .map(|i| new_spans[i].end - new_spans[i].start)
                .sum()
        };
        let (dirty_bytes, clean_bytes) = (bytes_of(true), bytes_of(false));
        if dirty_bytes as f64 > MAX_DIRTY_SHARE * (dirty_bytes + clean_bytes) as f64 {
            return everything(units);
        }

        // Positions in a clean group move with the units that hold them.
        let mut bytes_moved = vec![false; n + m];
        let mut rows_moved = vec![false; n + m];
        for (i, o) in found.iter().enumerate() {
            if let Some(o) = o.filter(|_| !dirty[i]) {
                bytes_moved[group[i]] |= old_spans[o].start != new_spans[i].start;
                rows_moved[group[i]] |= old_spans[o].row != new_spans[i].row;
            }
        }
        let by = Delta::by_unit(&remap);
        let mut replay = Vec::with_capacity(n);
        for (i, o) in found.iter().enumerate() {
            let Some(o) = o.filter(|_| !dirty[i]) else {
                replay.push(None);
                continue;
            };
            let record = old.units[o].prop.as_ref().unwrap();
            let group = group[i];
            replay.push(Some(if bytes_moved[group] || rows_moved[group] {
                Arc::new(record.moved(by, rows_moved[group]))
            } else {
                record.clone()
            }));
        }
        if remap.is_bad() {
            return everything(units);
        }
        let mut plan = Self::run_all(units);
        plan.dirty = dirty;
        plan.replay = replay;
        BuiltPlan { plan, env }
    }

    /// The output of the whole pass: the records of the units that were not
    /// run, in their places among the output of those that were. The scopes
    /// of the replayed units get their final shapes. `next` gets the records
    /// of all units.
    pub(crate) fn finish(
        self,
        runs: Vec<Option<UnitRun>>,
        scopes: &mut [FunctionShapeScope],
        next: &mut UnitMemo,
    ) -> Result<PropagateOutput, String> {
        let n = self.dirty.len();
        let identity = Remap::identity(&self.spans);
        let mut records: Vec<Arc<PropRecord>> = Vec::with_capacity(n);
        for (i, run) in runs.into_iter().enumerate() {
            if let Some(record) = &self.replay[i] {
                records.push(record.clone());
                continue;
            }
            let run = run.ok_or("a unit was not run")?;
            let (first, last) = (self.first_scope[i], self.first_scope[i + 1]);
            let mut phases = run.phases;
            let mut deps = run.deps;
            // A record whose positions no unit holds could not be moved to
            // the next text.
            match &identity {
                Some(table) => {
                    phases.iter_mut().for_each(|p| p.check(Delta::by_unit(table)));
                    if table.is_bad() {
                        deps = [None, None];
                    }
                }
                None => deps = [None, None],
            }
            records.push(Arc::new(PropRecord {
                phases,
                scopes: Arc::new(scopes[first..last].iter().map(|s| s.shapes.clone()).collect()),
                module: Arc::new(
                    self.syntax[i]
                        .names
                        .assigned
                        .iter()
                        .map(|name| (name.clone(), scopes[0].shapes.get(name).cloned()))
                        .collect(),
                ),
                deps,
            }));
        }
        for i in (0..n).filter(|&i| !self.dirty[i]) {
            let (first, last) = (self.first_scope[i], self.first_scope[i + 1]);
            for (scope, shapes) in scopes[first..last].iter_mut().zip(records[i].scopes.iter()) {
                scope.shapes = shapes.clone();
            }
            for (name, shape) in records[i].module.iter() {
                match shape {
                    Some(shape) => scopes[0].shapes.insert(name.clone(), shape.clone()),
                    None => scopes[0].shapes.remove(name),
                };
            }
        }
        let mut applications = Vec::new();
        let mut errors = Vec::new();
        let mut assignments = Vec::new();
        for phase in 0..2 {
            for record in &records {
                let segment = &record.phases[phase];
                errors.extend(segment.errors.iter().cloned());
                applications.extend(segment.applications.iter().cloned());
                assignments.extend(segment.assignments.iter().cloned());
            }
        }
        for (unit, record) in next.units.iter_mut().zip(records) {
            unit.prop = Some(record);
        }
        Ok((applications, errors, assignments))
    }
}

/// The first part of two analyses that differs, or `None` if they are equal.
/// Applications compare with their ranges, which `LayerApplication`'s own
/// equality leaves out.
pub fn first_analysis_difference(a: &LayerShapeAnalysis, b: &LayerShapeAnalysis) -> Option<String> {
    let LayerShapeAnalysis {
        scopes,
        layers,
        applications,
        errors,
        assignment_shapes,
    } = a;
    let part = if *scopes != b.scopes {
        "scopes"
    } else if *layers != b.layers {
        "layers"
    } else if *errors != b.errors {
        "errors"
    } else if *assignment_shapes != b.assignment_shapes {
        "assignment_shapes"
    } else if format!("{applications:?}") != format!("{:?}", b.applications) {
        "applications"
    } else {
        return None;
    };
    Some(part.to_string())
}
