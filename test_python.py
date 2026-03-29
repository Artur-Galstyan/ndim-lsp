from jaxtyping import Array, Float
import jax.numpy as jnp


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
    p = (x @ y)
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