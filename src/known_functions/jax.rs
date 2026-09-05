//! Shape rules for public array APIs added between JAX 0.9.2 and 0.11.1.
//! See bench/jax_compat/README.md for the release audit and runtime oracle.
//! Reuse the parent module's dimension, argument, and broadcasting rules.

use super::*;

pub(super) fn classify(module: &[String], name: &str) -> Option<KnownFunction> {
    if module == ["jax", "numpy"] && name == "top_k" {
        return Some(KnownFunction::LaxTopK);
    }
    if module == ["jax", "lax"] {
        return match name {
            "broadcast_like" => Some(KnownFunction::BroadcastLike),
            "stack" => Some(KnownFunction::Stack),
            "unstack" => Some(KnownFunction::Unbind),
            "stage" => Some(KnownFunction::Elementwise {
                parameters: &["x"],
                rank_promotion: false,
            }),
            "mulhi" => Some(KnownFunction::Elementwise {
                parameters: &["x", "y"],
                rank_promotion: false,
            }),
            _ => None,
        };
    }
    if module == ["jax", "scipy", "special"] {
        let parameters: &'static [&'static str] = match name {
            "dawsn" | "erfcx" | "loggamma" => &["x"],
            "wofz" => &["z"],
            "boxcox" | "boxcox1p" => &["x", "lmbda"],
            "comb" => &["N", "k"],
            "owens_t" => &["h", "a"],
            _ => return None,
        };
        return Some(KnownFunction::Elementwise {
            parameters,
            rank_promotion: true,
        });
    }
    if module == ["jax", "scipy", "linalg"] {
        return match name {
            "hadamard" => Some(KnownFunction::Hadamard),
            "dft" => Some(KnownFunction::Dft),
            "invhilbert" => Some(KnownFunction::InvHilbert),
            "invpascal" => Some(KnownFunction::InvPascal),
            "helmert" => Some(KnownFunction::Helmert),
            "circulant" => Some(KnownFunction::Circulant),
            "fiedler" => Some(KnownFunction::Fiedler),
            "companion" => Some(KnownFunction::Companion),
            "fiedler_companion" => Some(KnownFunction::FiedlerCompanion),
            "leslie" => Some(KnownFunction::Leslie),
            "convolution_matrix" => Some(KnownFunction::ConvolutionMatrix),
            "qr_multiply" => Some(KnownFunction::QrMultiply),
            _ => None,
        };
    }
    None
}

fn array_shape(value: &str, shapes: &dyn ShapeLookup) -> Option<Vec<String>> {
    shapes.shape(value).cloned().or_else(|| {
        // Scalar literals need no entry in the scope's array-shape map.
        // Rust also parses "inf"/"NaN", which are identifiers in Python.
        ((value.bytes().any(|b| b.is_ascii_digit()) && value.parse::<f64>().is_ok())
            || matches!(value, "True" | "False"))
        .then(Vec::new)
    })
}

pub(super) fn apply_elementwise(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
    parameters: &[&str],
    rank_promotion: bool,
) -> Result<Option<Vec<String>>, String> {
    let mut output = Vec::new();
    for (i, parameter) in parameters.iter().enumerate() {
        let Some(value) = nth_positional_or_keyword(args, i, &[*parameter]) else {
            return Ok(None);
        };
        let Some(shape) = array_shape(value, shapes) else {
            return Ok(None);
        };
        if !rank_promotion && !output.is_empty() && !shape.is_empty() && output.len() != shape.len()
        {
            return Err("lax elementwise operands must have equal rank unless scalar".into());
        }
        output = broadcast_two_shapes(&output, &shape)?;
    }
    Ok(Some(output))
}

pub(super) fn apply_broadcast_like(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(arr) =
        nth_positional_or_keyword(args, 0, &["arr"]).and_then(|v| array_shape(v, shapes))
    else {
        return Ok(None);
    };
    let Some(like) =
        nth_positional_or_keyword(args, 1, &["like_arr"]).and_then(|v| array_shape(v, shapes))
    else {
        return Ok(None);
    };
    let broadcast = broadcast_two_shapes(&arr, &like)?;
    if broadcast.len() != like.len()
        || !broadcast
            .iter()
            .zip(&like)
            .all(|(a, b)| dims_canonically_equal(a, b))
    {
        return Err("broadcast_like cannot broadcast input to the reference shape".into());
    }
    Ok(Some(like))
}

fn size_arg(
    args: &[CallArgument],
    index: usize,
    name: &str,
    shapes: &dyn ShapeLookup,
) -> Option<String> {
    let value = nth_positional_or_keyword(args, index, &[name])?.trim();
    if let Some(dim) = resolve_shape_index(value, shapes) {
        return Some(dim);
    }
    if let Ok(n) = value.parse::<i64>() {
        return Some(n.to_string());
    }
    // Keep symbolic integer parameters, but not arbitrary Python expressions
    // or synthetic array bindings introduced by the recursive evaluator.
    let mut chars = value.chars();
    if chars.next().is_some_and(|c| c.is_alphabetic() || c == '_')
        && chars.all(|c| c.is_alphanumeric() || c == '_')
        && shapes.shape(value).is_none_or(Vec::is_empty)
        && !value.starts_with("__synth_")
        && !matches!(value, "None" | "True" | "False")
    {
        return Some(value.to_string());
    }
    None
}

fn require_min_size(dim: &str, min: i64, context: &str) -> Result<(), String> {
    if dim.parse::<i64>().is_ok_and(|n| n < min) {
        return Err(format!("{context} requires size >= {min}, got {dim}"));
    }
    Ok(())
}

pub(super) fn apply_matrix(
    function: &KnownFunction,
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    if matches!(
        function,
        KnownFunction::Hadamard
            | KnownFunction::Dft
            | KnownFunction::InvHilbert
            | KnownFunction::InvPascal
            | KnownFunction::Helmert
    ) {
        let Some(mut n) = size_arg(args, 0, "n", shapes) else {
            return Ok(None);
        };
        if matches!(function, KnownFunction::Hadamard) {
            require_min_size(&n, 1, "hadamard")?;
            if n.parse::<u64>().is_ok_and(|n| !n.is_power_of_two()) {
                return Err(format!("hadamard requires a power-of-two size, got {n}"));
            }
        } else if matches!(function, KnownFunction::Helmert) {
            require_min_size(&n, 1, "helmert")?;
            let Some(full) =
                parse_bool(nth_positional_or_keyword(args, 1, &["full"]).unwrap_or("False"))
            else {
                return Ok(None);
            };
            let rows = if full { n.clone() } else { add_to_dim(&n, -1) };
            return Ok(Some(vec![rows, n]));
        } else if let Ok(value) = n.parse::<i64>() {
            // These constructors use arange(n), including its empty result
            // for non-positive n. Their second argument is NOT a column count.
            n = value.max(0).to_string();
        }
        return Ok(Some(vec![n.clone(), n]));
    }

    let keyword = match function {
        KnownFunction::Circulant => "c",
        KnownFunction::Leslie => "f",
        _ => "a",
    };
    let Some(mut input) =
        nth_positional_or_keyword(args, 0, &[keyword]).and_then(|v| array_shape(v, shapes))
    else {
        return Ok(None);
    };
    if input.is_empty() {
        if matches!(function, KnownFunction::ConvolutionMatrix) {
            return Err("convolution_matrix requires rank >= 1".into());
        }
        // These JAX constructors explicitly promote scalars to length one.
        input.push("1".into());
    }
    let n = input.pop().expect("scalar inputs were promoted above");
    match function {
        KnownFunction::Companion | KnownFunction::FiedlerCompanion => {
            let min = if matches!(function, KnownFunction::Companion) {
                2
            } else {
                1
            };
            require_min_size(&n, min, "companion")?;
            let order = add_to_dim(&n, -1);
            input.extend([order.clone(), order]);
        }
        KnownFunction::Leslie => {
            require_min_size(&n, 2, "leslie")?;
            let Some(mut survival) =
                nth_positional_or_keyword(args, 1, &["s"]).and_then(|v| array_shape(v, shapes))
            else {
                return Ok(None);
            };
            let survival_n = survival.pop().unwrap_or_else(|| "1".into());
            check_dim_match(&add_to_dim(&n, -1), &survival_n, "leslie")?;
            input = broadcast_prefix_shapes(&input, &survival, "leslie")?;
            input.extend([n.clone(), n]);
        }
        KnownFunction::ConvolutionMatrix => {
            require_min_size(&n, 1, "convolution_matrix input")?;
            let Some(columns) = size_arg(args, 1, "n", shapes) else {
                return Ok(None);
            };
            require_min_size(&columns, 1, "convolution_matrix columns")?;
            let mode = nth_positional_or_keyword(args, 2, &["mode"]).unwrap_or("'full'");
            let rows = match mode {
                "'full'" | "\"full\"" => match (n.parse::<i64>(), columns.parse::<i64>()) {
                    (Ok(m), Ok(n)) => (m as i128 + n as i128 - 1).to_string(),
                    _ => format!("({n})+({columns})-1"),
                },
                "'same'" | "\"same\"" | "'valid'" | "\"valid\"" => {
                    // Ordering unrelated symbolic dimensions is not known.
                    let (Ok(m), Ok(n)) = (n.parse::<i64>(), columns.parse::<i64>()) else {
                        return Ok(None);
                    };
                    if mode.contains("same") {
                        m.max(n).to_string()
                    } else {
                        (m.max(n) - m.min(n) + 1).to_string()
                    }
                }
                _ if mode.starts_with(['\'', '"']) => {
                    return Err("convolution_matrix mode must be full, same, or valid".into());
                }
                _ => return Ok(None),
            };
            input.extend([rows, columns]);
        }
        KnownFunction::Circulant | KnownFunction::Fiedler => input.extend([n.clone(), n]),
        _ => return Ok(None),
    }
    Ok(Some(input))
}

/// JAX NumPy and LAX top_k share axis/k shape semantics. Unlike torch.topk,
/// axis is keyword-only and named `axis`, not `dim`.
pub(crate) fn compute_top_k_shape(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<String>>, String> {
    let Some(mut output) =
        nth_positional_or_keyword(args, 0, &["a", "operand"]).and_then(|v| array_shape(v, shapes))
    else {
        return Ok(None);
    };
    if output.is_empty() {
        return Err("top_k requires rank >= 1".into());
    }
    let axis = args
        .iter()
        .find_map(|arg| match arg {
            CallArgument::Keyword { name, value } if name == "axis" => Some(value.as_str()),
            _ => None,
        })
        .unwrap_or("-1");
    let Some(axis) = parse_axis(axis) else {
        return Ok(None);
    };
    let axis = normalize_axis(axis, output.len(), "top_k")?;
    let Some(k) = size_arg(args, 1, "k", shapes) else {
        return Ok(None);
    };
    require_min_size(&k, 0, "top_k")?;
    if let (Ok(k), Ok(size)) = (k.parse::<i64>(), output[axis].parse::<i64>())
        && k > size
    {
        return Err(format!("top_k k={k} exceeds axis size {size}"));
    }
    output[axis] = k;
    Ok(Some(output))
}

pub(crate) fn compute_qr_multiply_shapes(
    args: &[CallArgument],
    shapes: &dyn ShapeLookup,
) -> Result<Option<Vec<Vec<String>>>, String> {
    let Some(a) = nth_positional_or_keyword(args, 0, &["a"]).and_then(|v| array_shape(v, shapes))
    else {
        return Ok(None);
    };
    let Some(mut c) =
        nth_positional_or_keyword(args, 1, &["c"]).and_then(|v| array_shape(v, shapes))
    else {
        return Ok(None);
    };
    if a.len() < 2 || c.is_empty() {
        return Err("qr_multiply requires rank >= 2 for a and rank >= 1 for c".into());
    }
    let mode = nth_positional_or_keyword(args, 2, &["mode"]).unwrap_or("'right'");
    let left = match mode {
        "'left'" | "\"left\"" => true,
        "'right'" | "\"right\"" => false,
        _ if mode.starts_with(['\'', '"']) => {
            return Err("qr_multiply mode must be left or right".into());
        }
        _ => return Ok(None),
    };
    let Some(pivoting) =
        parse_bool(nth_positional_or_keyword(args, 3, &["pivoting"]).unwrap_or("False"))
    else {
        return Ok(None);
    };
    let m = &a[a.len() - 2];
    let n = &a[a.len() - 1];
    let k = numeric_min_dim(m, n);
    let vector = c.len() == 1;
    if vector {
        if left {
            c.push("1".into());
        } else {
            c.insert(0, "1".into());
        }
    }
    let rows = &c[c.len() - 2];
    let cols = &c[c.len() - 1];
    if left {
        check_dim_match(&k, rows, "qr_multiply Q @ c")?;
    } else {
        check_dim_match(m, cols, "qr_multiply c @ Q")?;
    }
    let batch = broadcast_prefix_shapes(&a[..a.len() - 2], &c[..c.len() - 2], "qr_multiply")?;
    let mut product = batch.clone();
    if left {
        product.extend([m.clone(), cols.clone()]);
    } else {
        product.extend([rows.clone(), k.clone()]);
    }
    if vector {
        // JAX uses ravel(), not squeeze(): even batch axes collapse for 1-D c.
        product = vec![flattened_dim(&product)];
    }
    let mut r = batch.clone();
    r.extend([k, n.clone()]);
    let mut outputs = vec![product, r];
    if pivoting {
        let mut p = batch;
        p.push(n.clone());
        outputs.push(p);
    }
    Ok(Some(outputs))
}
