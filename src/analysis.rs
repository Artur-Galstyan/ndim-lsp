use std::path::PathBuf;

use tree_sitter::Node;

use crate::layers::{
    apply_layer_applications, extract_layer_applications, extract_layer_assignments,
};
use crate::python_ast::extract_jaxtyping_shapes;

use crate::types::*;

pub fn analyze_layer_shapes<F>(
    node: Node,
    text: &str,
    search_roots: &[PathBuf],
    read_file: F,
    max_depth: usize,
) -> Result<LayerShapeAnalysis, String>
where
    F: Fn(&PathBuf) -> Option<String>,
{
    let mut shapes = extract_jaxtyping_shapes(node, text)?;
    let layers = extract_layer_assignments(node, text, search_roots, read_file, max_depth)?;
    let applications = extract_layer_applications(node, text, &layers)?;
    let errors = apply_layer_applications(&applications, &mut shapes);

    Ok(LayerShapeAnalysis {
        shapes,
        layers,
        applications,
        errors,
    })
}
