import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import Linear
from jaxtyping import Array, Float

"""
# valid matmul: b matches b
def valid_matmul(x: Float[Array, "a b"], y: Float[Array, "b c"]):
    z = x @ y
    return z


# invalid matmul: b != c
def invalid_matmul(x: Float[Array, "a b"], y: Float[Array, "c a"]):
    return x @ y


# shape tracing: hover z should show (a, c)
def trace_shape(x: Float[Array, "a b"], y: Float[Array, "b c"]):
    z = x @ y
    return z


# same param names, different shapes across functions
def scoped_a(x: Float[Array, "a b"], y: Float[Array, "b c"]):
    return x @ y




def scoped_b(x: Float[Array, "m n"], y: Float[Array, "p q"]):
    return x @ y

# chained matmul: z = x @ y, then z @ w
def chained(x: Float[Array, "a b"], y: Float[Array, "b c"], w: Float[Array, "c d"]):
    z = x @ y
    return z @ w




# sequential: z traced, then used
def sequential(x: Float[Array, "a b"], y: Float[Array, "b c"], v: Float[Array, "c d"]):
    z = x @ y
    w = z @ v
    return w


"""


def chaos(
    a: Float[Array, "i j"],
    b: Float[Array, "j k"],
    c: Float[Array, "k l"],
    d: Float[Array, "l m"],
    e: Float[Array, "m n"],
    f: Float[Array, "x y"],
):
    # (i, j) @ (j, k) = (i, k) ✓
    ab = a @ b
    # (i, k) @ (k, l) = (i, l) ✓
    abc = ab @ c
    # (i, l) @ (l, m) = (i, m) ✓
    abcd = abc @ d
    # (i, m) @ (m, n) = (i, n) ✓
    abcde = abcd @ e
    # (i, n) @ (x, y) → n != x ✗ squiggle
    bad = abcde @ f
    # (x, y) @ (i, j) → y != i ✗ squiggle
    worse = f @ a
    # (i, j) @ (j, k) = (i, k) ✓
    ok = a @ b
    # (i,j)@(j,k)=(i,k) → (i,k)@(k,l)=(i,l) → (i,l)@(l,m)=(i,m) → (i,m)@(m,n)=(i,n) ✓
    chain = a @ b @ c @ d @ e
    # same as chain but then (i,n)@(x,y) → n != x ✗ squiggle
    bad_chain = a @ b @ c @ d @ e @ f
    return abcde


def elementwise(x: Float[Array, "a b"], y: Float[Array, "a b"], z: Float[Array, "c d"]):
    # (a, b) + (a, b) = (a, b) ✓
    good = x + y
    # (a, b) - (c, d) → a != c ✗ squiggle
    bad = x - z
    # (a, b) * (a, b) = (a, b) ✓ then (a, b) + (c, d) ✗ squiggle
    mixed = (x * y) + z


def parens_and_transpose(
    x: Float[Array, "a b"],
    y: Float[Array, "b c"],
    w: Float[Array, "c a"],
):
    # parens: (a,b) @ (b,c) = (a,c) ✓
    p = x @ y
    # transpose: w is (c,a), w.T is (a,c) ✓
    wt = w.T
    # parens + matmul: (a,c) @ (c,a) = (a,a) ✓
    combo = (x @ y) @ w
    # transpose in matmul: (a,b) @ (a,b).T = (a,b) @ (b,a) = (a,a) ✓
    xtx = x @ x.T
    # chained with transpose: (a,b) @ (b,c) = (a,c), then (a,c) @ (a,c).T = (a,c) @ (c,a) = (a,a) ✓
    chain_t = (x @ y) @ (x @ y).T
    # invalid: x is (a,b), y.T is (c,b), b != c ✗ squiggle
    bad_t = x @ y.T
    # invalid parens: (a,b) @ (b,c) = (a,c), (a,c) + (a,b) → c != b ✗ squiggle
    bad_parens = (x @ y) + x


def transpose_tests(
    x: Float[Array, "a b"],
    y: Float[Array, "a b c"],
):
    # (a, b) → (b, a) ✓
    t1 = jnp.transpose(x, (1, 0))
    # (a, b, c) → (c, a, b) ✓
    t2 = jnp.transpose(y, (2, 0, 1))
    # (b, a) @ (a, b) = (b, b) ✓
    valid = t1 @ x
    # (b, a) + (a, b) → b != a ✗ squiggle
    bad = t1 + x


from jaxtyping import Array, Float


def sum_tests(
    x: Float[Array, "a b"],
    y: Float[Array, "a b c"],
):
    # (a, b) sum axis=0 → (b) ✓
    s1 = jnp.sum(x, axis=0)
    # (a, b) sum axis=1 → (a) ✓
    s2 = jnp.sum(x, axis=1)
    # (a, b, c) sum axis=1 → (a, c) ✓
    s3 = jnp.sum(y, axis=1)
    # (a, c) @ (c, b) — but we don't have (c, b), so let's use transpose
    # (a, b, c) sum axis=0 → (b, c), transpose → (c, b), then (a, c) @ (c, b) = (a, b) ✓
    s4 = jnp.sum(y, axis=0)
    s4t = jnp.transpose(s4, (1, 0))
    valid = s3 @ s4t
    # (b) + (a) → b != a ✗ squiggle
    bad = s1 + s2


def squeeze_tests(
    x: Float[Array, "a b"],
    y: Float[Array, "a b c"],
):
    # expand then squeeze: (a, b) → (1, a, b) → (a, b) ✓
    expanded = jnp.expand_dims(x, axis=0)
    back = jnp.squeeze(expanded, axis=0)
    # expand at end then squeeze: (a, b) → (a, b, 1) → (a, b) ✓
    expanded2 = jnp.expand_dims(x, axis=2)
    back2 = jnp.squeeze(expanded2, axis=2)
    # double expand then squeeze all: (a, b) → (1, a, b) → (1, a, 1, b) → (a, b) ✓
    double = jnp.expand_dims(jnp.expand_dims(x, axis=0), axis=2)
    squeezed_all = jnp.squeeze(double)
    # squeeze symbolic dim: (a, b) squeeze axis=0 → a is not "1" ✗ squiggle
    bad = jnp.squeeze(x, axis=0)
    # squeeze out of bounds: (a, b) squeeze axis=5 ✗ squiggle
    oob = jnp.squeeze(x, axis=5)
    # keepdims then squeeze: (a, b, c) sum axis=1 keepdims → (a, 1, c) → squeeze axis=1 → (a, c) ✓
    kept = jnp.sum(y, axis=1, keepdims=True)
    unsqueezed = jnp.squeeze(kept, axis=1)


def linear_tests(
    x: Float[Array, "batch in_dim"],
    y: Float[Array, "batch other_dim"],
    key: Float[Array, "2"],
):
    # Linear: replaces last dim in_dim → out_dim
    linear = eqx.nn.Linear(in_features=in_dim, out_features=out_dim, key=key)
    # (batch, in_dim) → (batch, out_dim) ✓
    z = linear(x)
    # (batch, other_dim) → last dim is other_dim, not in_dim ✗ squiggle

    bad = linear(y)


def concat_tests(
    x: Float[Array, "a b"],
    y: Float[Array, "a c"],
    z: Float[Array, "a b"],
    w: Float[Array, "d b"],
):
    # (a, b) and (a, c) along axis=1 → (a, b+c) ✓
    c1 = jnp.concatenate([x, y], axis=1)
    # (a, b) and (a, b) along axis=1 → (a, b+b) ✓
    c2 = jnp.concatenate([x, z], axis=1)
    # (a, b) and (a, b) along axis=0 → (a+a, b) ✓
    c3 = jnp.concatenate([x, z], axis=0)
    # three arrays: (a, b), (a, c), (a, b) along axis=1 → (a, b+c+b) ✓
    c4 = jnp.concatenate([x, y, z], axis=1)
    # dim mismatch: (a, b) and (d, b) along axis=1 → a != d ✗ squiggle
    bad = jnp.concatenate([x, w], axis=1)
    # axis out of bounds: (a, b) axis=5 ✗ squiggle
    oob = jnp.concatenate([x, y], axis=5)
