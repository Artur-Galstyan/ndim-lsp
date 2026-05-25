"""End-to-end visual test file for ndim-lsp shape analysis.

Open this file in VSCode with the ndim-lsp server running.
Lines marked  # expect: ...  should show a red squiggle diagnostic.
All other lines should be clean.
"""

from jaxtyping import Float, Array
import equinox as eqx
import torch
import jax.numpy as jnp
import numpy as np

# ── Section 1: jaxtyping annotations + function scope isolation ──

def annotated_fn(x: Float[Array, "batch features"], y: Float[Array, "batch 1"]):
    # x has shape ["batch", "features"], y has shape ["batch", "1"]
    # No diagnostics expected — shapes are valid annotations.
    return x, y


def another_fn(x: Float[Array, "seq_len dim"]):
    # Same param name 'x' as above, but different shape — no cross-contamination.
    # No diagnostics expected.
    return x


def scope_collision_demo(x: Float[Array, "n m"]):
    # 'x' here is ["n", "m"], independent of the 'x' in annotated_fn.
    # No diagnostics expected.
    return x


# ── Section 2: equinox Linear layer (success + mismatch) ──

def eqx_linear_ok(x: Float[Array, "batch 64"]):
    layer = eqx.nn.Linear(64, 128)
    y = layer(x)  # OK: last dim 64 matches in_features=64 → shape ["batch", "128"]
    return y


def eqx_linear_mismatch(x: Float[Array, "batch 32"]):
    layer = eqx.nn.Linear(64, 128)
    y = layer(x)  # expect: Linear layer expected input last dim 64, got 32
    return y


# ── Section 3: torch Linear layer ──

def torch_linear_ok(x: Float[Array, "batch 128"]):
    layer = torch.nn.Linear(128, 256)
    y = layer(x)  # OK: last dim 128 matches in_features=128 → shape ["batch", "256"]
    return y


def torch_linear_mismatch(x: Float[Array, "batch 64"]):
    layer = torch.nn.Linear(128, 256)
    y = layer(x)  # expect: Linear layer expected input last dim 128, got 64
    return y


# ── Section 4: equinox Conv2d (success + channels mismatch) ──

def eqx_conv2d_ok(x: Float[Array, "batch 3 32 32"]):
    layer = eqx.nn.Conv2d(3, 16, 3)
    y = layer(x)  # OK: in_channels=3 matches, output shape ["batch", "16", "30", "30"]
    return y


def eqx_conv2d_channels_mismatch(x: Float[Array, "batch 1 32 32"]):
    layer = eqx.nn.Conv2d(3, 16, 3)
    y = layer(x)  # expect: Conv2d expected 3 input channels, got 1
    return y


def eqx_conv2d_stride_padding(x: Float[Array, "batch 3 32 32"]):
    layer = eqx.nn.Conv2d(3, 16, 3, stride=2, padding=1)
    y = layer(x)  # OK: (32+2*1-3)/2+1 = 16 → shape ["batch", "16", "16", "16"]
    return y


# ── Section 5: torch Conv2d ──

def torch_conv2d_ok(x: Float[Array, "batch 3 32 32"]):
    layer = torch.nn.Conv2d(3, 16, 3)
    y = layer(x)  # OK: in_channels=3 matches → shape ["batch", "16", "30", "30"]
    return y


def torch_conv2d_channels_mismatch(x: Float[Array, "batch 8 32 32"]):
    layer = torch.nn.Conv2d(3, 16, 3)
    y = layer(x)  # expect: Conv2d expected 3 input channels, got 8
    return y


def torch_conv2d_stride_padding(x: Float[Array, "batch 3 64 64"]):
    layer = torch.nn.Conv2d(3, 32, 5, stride=2, padding=2)
    y = layer(x)  # OK: (64+2*2-5)/2+1 = 32 → shape ["batch", "32", "32", "32"]
    return y


# ── Section 6: known free-function calls ──

def free_fn_sum(x: Float[Array, "batch features"]):
    y = jnp.sum(x, axis=0)  # OK: reduces axis 0 → shape ["features"]
    return y


def free_fn_reshape_ok(x: Float[Array, "6 4"]):
    y = jnp.reshape(x, (3, 8))  # OK: 6*4 == 3*8
    return y


def free_fn_reshape_bad(x: Float[Array, "6 4"]):
    y = jnp.reshape(x, (3, 9))  # expect: reshape size mismatch (24 vs 27)
    return y


def free_fn_np_reshape_ok(x: Float[Array, "2 3 4"]):
    y = np.reshape(x, (6, 4))  # OK: 2*3*4 == 6*4
    return y


# ── Section 7: method calls ──

def method_flatten(x: Float[Array, "batch height width channels"]):
    y = x.flatten()  # OK: flattens to 1D → shape ["batch*height*width*channels"]
    return y


def method_sum(x: Float[Array, "batch features"]):
    z = x.sum(axis=0)  # OK: reduces axis 0 → shape ["features"]
    return z


def method_reshape_ok(x: Float[Array, "6 4"]):
    w = x.reshape(3, 8)  # OK: 6*4 == 3*8
    return w


def method_reshape_bad(x: Float[Array, "6 4"]):
    bad = x.reshape(3, 9)  # expect: reshape size mismatch (24 vs 27)
    return bad


def method_squeeze_unsqueeze(x: Float[Array, "batch 1 features"]):
    y = x.squeeze(1)  # OK: removes axis 1 (size 1) → shape ["batch", "features"]
    z = y.unsqueeze(0)  # OK: inserts axis at 0 → shape ["1", "batch", "features"]
    return z


def method_transpose(x: Float[Array, "batch features"]):
    y = x.transpose()  # OK: reverses dims → shape ["features", "batch"]
    return y


# ── Section 8: binary operators ──

def matmul_success(a: Float[Array, "batch m k"], b: Float[Array, "batch k n"]):
    c = a @ b  # OK: batch dims match, inner k==k → shape ["batch", "m", "n"]
    return c


def matmul_inner_mismatch(a: Float[Array, "batch 3 5"], b: Float[Array, "batch 4 2"]):
    c = a @ b  # expect: matmul dimension mismatch — last dim 5 != second-to-last dim 4
    return c


def matmul_batch_mismatch(a: Float[Array, "8 3 5"], b: Float[Array, "7 5 2"]):
    c = a @ b  # expect: matmul batch dimension mismatch — [8] != [7]
    return c


def elementwise_add_ok(x: Float[Array, "batch features"], y: Float[Array, "batch features"]):
    z = x + y  # OK: shapes match exactly → shape ["batch", "features"]
    return z


def elementwise_add_mismatch(x: Float[Array, "batch 32"], y: Float[Array, "batch 64"]):
    z = x + y  # expect: elementwise + expected equal shapes, got [batch, 32] and [batch, 64]
    return z


def elementwise_mul_ok(x: Float[Array, "h w"], y: Float[Array, "h w"]):
    z = x * y  # OK: shapes match → shape ["h", "w"]
    return z


def elementwise_sub_mismatch(x: Float[Array, "n"], y: Float[Array, "m"]):
    z = x - y  # expect: elementwise - expected equal shapes, got [n] and [m]
    return z


# ── Section 9: chained / interleaved features ──

def chain_layer_method_binary(x: Float[Array, "batch 64"]):
    # 1. Apply a linear layer
    layer = eqx.nn.Linear(64, 32)
    h = layer(x)  # shape ["batch", "32"]

    # 2. Method call on the result
    h_flat = h.flatten()  # OK: flattens to 1D

    # 3. Binary op with matching shapes
    y = h + h  # OK: both ["batch", "32"] → shape ["batch", "32"]
    return y


def chain_reshape_then_sum(x: Float[Array, "4 6"]):
    y = x.reshape(2, 12)  # OK: 4*6 == 2*12 → shape ["2", "12"]
    z = jnp.sum(y, axis=1)  # OK: reduces axis 1 → shape ["2"]
    return z


def chain_conv_reshape_linear(x: Float[Array, "batch 3 32 32"]):
    conv = eqx.nn.Conv2d(3, 16, 3, padding=0)
    h = conv(x)  # OK: shape ["batch", "16", "30", "30"]
    h_flat = h.flatten()  # OK: flattens all dims
    # Note: linear layer in_features must match the total flattened size
    # 16*30*30 = 14400 — but we use a symbolic name for readability
    return h_flat
